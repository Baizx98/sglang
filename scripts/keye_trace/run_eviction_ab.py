"""Compare two online Keye LRU paths on a frozen request matrix."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from run_eviction_agent import GPUS, MODEL, SEED, inventory, post, read_jsonl, save


def freeze(source, output):
    cohorts = {8192: {}, 32768: {}, 131072: {}}
    tasks = ("niah_single_1", "niah_multikey_1", "vt", "qa_1")
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            length = row.get("length_bucket_upper")
            if (row["dataset"] != "RULER" or length not in cohorts or row["task"] not in tasks
                    or row["split"] != "test"):
                continue
            cohorts[length].setdefault(row["task"], row)
    frozen = {
        str(length): [
            dict(rid=f"ab__ruler__{length}__{task}", source_rid=r["rid"],
                 prompt_tokens=r["prompt_tokens"], input_ids=r["input_ids"])
            for task in tasks for r in [cohort[task]]
        ] for length, cohort in cohorts.items()
    }
    save(output, frozen)
    return frozen


def run_mode(root, rows, capacity, steps, mode, port, batch_size):
    root.mkdir()
    mode_env = dict(
        CUDA_VISIBLE_DEVICES=",".join(GPUS), KEYE_SM80_DSA="1", KEYE_SM80_EXACT_TOPK="1",
        KEYE_HORIZON_STORAGE="lru", KEYE_HORIZON_CACHE_TOKENS=str(capacity),
        KEYE_EVICTION_AB_DIR=str(root), KEYE_EVICTION_AB_MODE=mode,
        KEYE_EVICTION_AB_MAX_REQUESTS=str(batch_size), KEYE_SM80_PREFILL_SCORE_CHUNK="64",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="false",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KEYE_", "SGLANG_"))}
    env.update(mode_env)
    max_prompt = max(row["prompt_tokens"] for row in rows)
    waves = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    total_tokens = max(sum(row["prompt_tokens"] + steps + 128 for row in wave) for wave in waves)
    # The checkpoint uses four KV heads by 128 BF16 features across 48 layers.
    host_main_bytes = (total_tokens + 2112) * 4 * 128 * 2 * 2 * 48
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text()
                         .splitlines() if line.startswith("MemAvailable:"))
    if host_main_bytes + (12 << 30) > available_kib * 1024:
        raise RuntimeError(f"Insufficient available DRAM for {host_main_bytes} pinned KV bytes")
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
        "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", "1", "--pp-size", "2", "--pp-max-micro-batch-size", str(batch_size),
        "--attention-backend", "keye_sa",
        "--context-length", str(max_prompt + steps + 128),
        "--max-total-tokens", str(total_tokens + 2048),
        "--max-running-requests", str(batch_size), "--chunked-prefill-size", "4096",
        "--mem-fraction-static", "0.90", "--disable-cuda-graph",
        "--disable-custom-all-reduce", "--disable-radix-cache", "--disable-overlap-schedule",
        "--dtype", "bfloat16", "--random-seed", str(SEED), "--skip-server-warmup",
    ]
    save(root / "launch.json", dict(devices=inventory(GPUS, require_idle=True),
         command=command, environment=mode_env, mode=mode, per_request_capacity=capacity,
         planned_pinned_main_kv_bytes=host_main_bytes,
         request_ids=[r["rid"] for r in rows], commit=subprocess.check_output(
             ["git", "rev-parse", "HEAD"], text=True).strip()))
    with (root / "server.log").open("x") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(400):
            if server.poll() is not None:
                raise RuntimeError(f"{mode} server stopped during load: {server.returncode}")
            try:
                with urlopen(url + "/health", timeout=2):
                    break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError(f"{mode} Keye server readiness")
        response = []
        start = time.perf_counter()
        for wave in waves:
            response.extend(post(url + "/generate", dict(
                input_ids=[r["input_ids"] for r in wave], rid=[r["rid"] for r in wave],
                sampling_params=dict(temperature=0, max_new_tokens=steps + 1, ignore_eos=True))))
        elapsed = time.perf_counter() - start
        save(root / "response.json", dict(response=response, wall_seconds=elapsed))
        return response
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def read_pair(root, rows, steps, batch_size, output_paths):
    metadata = {row["rid"]: row for row in rows}
    totals, by_layer, actual_active_batch = {}, [], Counter()
    expected = {(rid, step) for rid in metadata for step in range(steps)}
    excluded = Counter()
    for layer in range(48):
        layer_events = {}
        for mode in ("baseline", "prefetch"):
            path = output_paths[mode] / f"layer{layer:02d}.jsonl"
            by_key = {}
            for event in read_jsonl(path):
                rid, step = event["rid"], event["step"]
                key = (rid, step)
                assert key not in by_key and event["layer"] == layer
                assert event["context"] == metadata[rid]["prompt_tokens"] + step + 1
                by_key[key] = event
                actual_active_batch[event["active_batch"]] += 1
            assert by_key.keys() == expected, (mode, layer, len(by_key), len(expected))
            layer_events[mode] = by_key
        matched = set()
        for key in expected:
            a = layer_events["baseline"][key]
            b = layer_events["prefetch"][key]
            if a["active_batch"] != batch_size or b["active_batch"] != batch_size:
                excluded["outside_requested_batch_in_either_path"] += 1
                continue
            demands = [sorted(pos for pos, _, _ in event["demand"]) for event in (a, b)]
            assert all(len(d) == len(set(d)) for d in demands)
            if demands[0] != demands[1] or a["current_selected"] != b["current_selected"]:
                raise ValueError(f"Exact Top-k differs between A and B at layer {layer}, key={key}")
            matched.add(key)
        per_request = Counter(rid for rid, _ in matched)
        if any(per_request[rid] < min(8, steps) for rid in metadata):
            raise ValueError(f"Too few paired B{batch_size} frames at layer {layer}: {per_request}")
        counters = []
        for mode, by_key in layer_events.items():
            counts = Counter()
            for key in sorted(matched):
                event = by_key[key]
                step = event["step"]
                counts.update(state for _, _, state in event["demand"])
                counts["predicted_eviction_wasted"] += sum(e[-1] for e in event["prefetch_evictions"])
                counts["prefetch_evicted_demand"] += len(
                    {e[0] for e in event["prefetch_evictions"]} &
                    {phys for _, phys, state in event["demand"] if state == "absent"})
                counts["demand_eviction_wasted"] += sum(e[-1] for e in event["demand_evictions"])
                for field in ("demand_bytes", "prefetch_bytes", "repeat_h2d"):
                    counts[field] += event[field]
                counts["exposed_transfer_ms"] += event["exposed_transfer_ms"] or 0.0
                counts["prefetch_events"] += int(bool(event["prefetch"]))
                counts["demand_rows"] += len(event["demand"])
                if step == steps - 1:
                    counts["unused_at_end"] += event["unused_resident"]
                if mode == "baseline":
                    assert not event["prefetch"] and not event["prefetch_bytes"]
            counters.append(counts)
            totals.setdefault(mode, Counter()).update(counts)
        by_layer.append(dict(layer=layer, baseline=counters[0], prefetch=counters[1],
                             matched_frames=len(matched)))
    if not totals["prefetch"]["prefetch_events"]:
        raise ValueError("Predictive path issued no actual H2D transfers")
    for count in totals.values():
        count["hit_rate"] = (count["resident"] + count["prefetch_ready"]) / count["demand_rows"]
    save(root / "comparison.json", dict(totals=totals, by_layer=by_layer, steps=steps,
         requested_batch=batch_size, active_batch_counts=actual_active_batch,
         excluded_frames=excluded, exact_demand_equal=True, output_ids_equal=True))
    return totals


def wait_gpu():
    while True:
        try:
            inventory(GPUS, require_idle=True)
            return
        except RuntimeError as exc:
            if "another compute process" not in str(exc):
                raise
            time.sleep(20)


def compare(args, root, rows, capacity, steps, batch_size):
    if (root / "comparison.json").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    results, output_paths = {}, {}
    for mode in ("baseline", "prefetch"):
        if args.wait_gpu:
            wait_gpu()
        mode_run = root / f"{mode}-{time.time_ns()}"
        response = run_mode(mode_run, rows, capacity, steps, mode, args.port, batch_size)
        results[mode] = {r["meta_info"]["id"]: r["output_ids"] for r in response}
        output_paths[mode] = mode_run
        assert len(results[mode]) == len(rows) and all(
            len(ids) == steps + 1 for ids in results[mode].values())
    if results["baseline"] != results["prefetch"]:
        raise ValueError("A/B generated token trajectories differ")
    save(root / "evidence.json", dict(paths={k: str(v) for k, v in output_paths.items()},
         input_ids_sha256={r["rid"]: hashlib.sha256(json.dumps(r["input_ids"]).encode()).hexdigest()
                           for r in rows}))
    totals = read_pair(root, rows, steps, batch_size, output_paths)
    print(root, json.dumps(totals), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ruler-source", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31327)
    parser.add_argument("--wait-gpu", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--agent-inputs", type=Path)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    frozen = args.run / "ruler-requests.json"
    requests = json.loads(frozen.read_text()) if frozen.is_file() else freeze(args.ruler_source, frozen)
    if args.freeze_only:
        return
    compare(args, args.run / "pilot-8k-b1-c4096", requests["8192"][:1], 4096, 8, 1)
    if not args.pilot_only:
        conditions = []
        for length in (32768, 131072):
            for batch in (1, 4):
                for capacity in (2048, 4096):
                    # Overlapping PP prefill can leave only a short full-batch
                    # decode window. Preserve all ramp rows for auditing.
                    conditions.append((f"{length}-b{batch}-c{capacity}",
                                       requests[str(length)][:batch], capacity,
                                       128 if batch > 1 else 32, batch))
        if args.agent_inputs:
            bridge = [dict(row, rid="ab__" + row["rid"]) for row in read_jsonl(args.agent_inputs)
                      if row.get("split") == "pilot"]
            conditions.append(("agent-bridge-b1-c4096", bridge, 4096, 32, 1))
        failures = {}
        for name, rows, capacity, steps, batch in conditions:
            try:
                compare(args, args.run / name, rows, capacity, steps, batch)
            except Exception as exc:
                failures[name] = repr(exc)
                print(f"{name}: {exc!r}", file=sys.stderr, flush=True)
        save(args.run / "formal-status.json", dict(
            status="partial" if failures else "complete", failures=failures,
            verified=[name for name, *_ in conditions if name not in failures]))


if __name__ == "__main__":
    main()
