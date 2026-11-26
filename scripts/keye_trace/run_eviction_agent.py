"""Run resumable Keye Agent trace collection and validate each committed turn."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np

from horizonkv_predictive_runtime import inventory, post
from prepare_eviction_agent import MODEL, read_jsonl

GPUS = (
    "GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822",
    "GPU-79886749-8f57-63a5-8c8e-8a68a085565a",
)
SEED = 20261126


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def validate_ids(a, length, width):
    assert a.shape == (width,) and a.dtype == np.int32
    valid = a[a >= 0]
    assert len(valid) == min(length, width), (len(valid), length, width)
    assert np.all(valid < length) and len(np.unique(valid)) == len(valid)
    assert np.all(a[len(valid):] == -1)
    return valid


def validate(row, events, output_ids):
    rid, prompt = row["rid"], row["prompt_tokens"]
    steps = max(0, len(output_ids) - 1)
    manifest, activation = [], []
    labels = np.full(prompt + steps, len(row["segments"]), dtype=np.int32)
    for i, span in enumerate(row["segments"]):
        labels[span["start"]:span["end"]] = i
    segments = row["segments"] + [
        dict(role="current_turn_generated", source_turn=row["turn"], start=prompt, end=prompt + steps)
    ]
    for layer in range(48):
        cursor = 0
        counts = np.zeros(len(segments), dtype=np.int64)
        eligible = np.zeros(len(segments), dtype=np.int64)
        for path in sorted(events.glob(f"{rid}__layer{layer:02d}__*.npz")):
            with np.load(path, allow_pickle=False) as chunk:
                for i, context in enumerate(chunk["context_len"]):
                    assert chunk["step"][i] == cursor and context == prompt + cursor + 1
                    exact = validate_ids(chunk["true_topk"][i], context, 2048)
                    if layer:
                        candidates = validate_ids(chunk["predicted_topm"][i], context, 3072)
                        assert candidates[0] == context - 1
                    else:
                        assert "predicted_topm" not in chunk.files
                    counts += np.bincount(labels[exact[exact < context - 1]],
                                          minlength=len(segments))
                    eligible += [max(0, min(context - 1, s["end"]) - s["start"])
                                 for s in segments]
                    cursor += 1
            manifest.append(dict(layer=layer, file=path.name, bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if cursor != steps:
            raise ValueError(f"Incomplete trace {rid} layer {layer}: {cursor}/{steps}")
        activation.extend(dict(layer=layer, segment=i, **span,
                               selected=int(counts[i]), eligible=int(eligible[i]))
                          for i, span in enumerate(segments))
    return dict(decode_steps=steps, chunks=manifest, activation=activation)


def collect(args, rows):
    root = args.run
    root.mkdir(parents=True, exist_ok=True)
    events = root / "events"
    events.mkdir(exist_ok=True)
    completed = root / "turns"
    completed.mkdir(exist_ok=True)
    total = len(rows)
    pending = []
    for row in rows:
        rid = row["rid"]
        if (completed / f"{rid}.json").exists():
            continue
        # Keep partial captures from interrupted turns for diagnosis.
        partial = list(events.glob(f"{rid}__layer*"))
        if partial:
            archive = root / "partial" / f"{rid}-{time.time_ns()}"
            archive.mkdir(parents=True)
            for file in partial:
                file.rename(archive / file.name)
        pending.append(row)
    if not pending:
        return
    existing = total - len(pending)
    max_prompt = max(r["prompt_tokens"] for r in rows)
    devices = inventory(GPUS, require_idle=True)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is occupied")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KEYE_", "SGLANG_"))}
    options = dict(
        CUDA_VISIBLE_DEVICES=",".join(GPUS), KEYE_SM80_DSA="1", KEYE_SM80_EXACT_TOPK="1",
        KEYE_AGENT_TRACE_DIR=str(events), KEYE_LOOKAHEAD_TRACE_DIR=str(events),
        KEYE_LOOKAHEAD_RID_PREFIX="unused-agent-shadow-prefix",
        KEYE_SM80_PREFILL_SCORE_CHUNK="64", OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
    )
    env.update(options)
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
        "--trust-remote-code", "--host", "127.0.0.1", "--port", str(args.port),
        "--tp-size", "1", "--pp-size", "2", "--attention-backend", "keye_sa",
        "--context-length", str(max(8192, max_prompt + args.max_new_tokens + 2)),
        "--max-total-tokens", str(max(16384, max_prompt + args.max_new_tokens + 2048)),
        "--max-running-requests", "1", "--chunked-prefill-size", "1024",
        "--mem-fraction-static", "0.90", "--disable-cuda-graph",
        "--disable-custom-all-reduce", "--disable-radix-cache", "--disable-overlap-schedule",
        "--dtype", "bfloat16", "--random-seed", str(SEED), "--skip-server-warmup",
    ]
    attempt = time.time_ns()
    save(root / f"launch-{attempt}.json", dict(command=command, options=options,
         devices=devices, source_commit=subprocess.check_output(
             ["git", "rev-parse", "HEAD"], text=True).strip(),
         source_inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
         prediction="same-state BF16 shadow, sorted history and reserved current, M3072"))
    url = f"http://127.0.0.1:{args.port}"
    save(root / "state.json", dict(status="loading", completed=existing, total=total))
    with (root / f"server-{attempt}.log").open("x") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        for _ in range(400):
            if server.poll() is not None:
                raise RuntimeError(f"Server stopped during load: {server.returncode}")
            try:
                with urlopen(url + "/health", timeout=2):
                    break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("Keye server readiness")
        warmup = post(url + "/generate", dict(
            input_ids=[151644, 872, 198, 9707], rid="trace-health",
            sampling_params=dict(max_new_tokens=3, temperature=0, ignore_eos=True)))
        if len(warmup.get("output_ids", [])) != 3:
            raise RuntimeError("Decode health request failed")
        for i, row in enumerate(pending, start=existing):
            rid = row["rid"]
            save(root / "state.json", dict(status="running", completed=i, total=total, rid=rid))
            post(url + "/flush_cache", {})
            response = post(url + "/generate", dict(
                input_ids=row["input_ids"], rid=rid,
                sampling_params=dict(max_new_tokens=args.max_new_tokens, temperature=0)))
            finish = response["meta_info"]["finish_reason"]
            if not finish or finish["type"] not in ("length", "stop"):
                raise RuntimeError(f"Unexpected finish {rid}: {finish}")
            # A subsequent prefill flushes the final short chunk on both PP ranks.
            post(url + "/generate", dict(input_ids=[151644, 872, 198, 9707],
                 rid="trace-flush", sampling_params=dict(max_new_tokens=1, temperature=0)))
            ids = response["output_ids"]
            checked = validate(row, events, ids)
            save(completed / f"{rid}.json", dict(
                rid=rid, dataset=row["dataset"], session_id=row["session_id"],
                turn=row["turn"], finish_reason=finish, generation_ids=ids, **checked))
            print(f"{i+1}/{total} {rid}: {checked['decode_steps']} validated steps", flush=True)
        summary = Counter()
        for row in rows:
            result = json.loads((completed / f"{row['rid']}.json").read_text())
            summary["steps"] += result["decode_steps"]
            summary["chunks"] += len(result["chunks"])
            summary["finish_" + result["finish_reason"]["type"]] += 1
        save(root / "state.json", dict(status="complete", completed=total, total=total, **summary))
    except BaseException as exc:
        save(root / "failure.json", dict(error=str(exc), attempt=attempt))
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--port", type=int, default=31326)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.run = args.run.resolve()
    rows = [r for path in args.inputs for r in read_jsonl(path)]
    if args.pilot:
        rows = [r for r in rows if r.get("split") == "pilot"]
    if not rows or len(rows) != len({r["rid"] for r in rows}):
        raise ValueError("Expected a nonempty cohort with unique request IDs")
    collect(args, rows)


if __name__ == "__main__":
    main()
