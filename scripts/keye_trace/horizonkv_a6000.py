"""Frozen dual-A6000 rerun queue, with independent 20-minute ntfy monitoring.

Raw requests, selection traces and Kineto captures remain under ignored data/.
The replay is explicitly a CPU simulation of traces collected on this GPU pair.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from run_horizonkv_motivation import run_wave, save

REPO = Path(__file__).resolve().parents[2]
GPU_IDS = ["GPU-43f0e479-a737-da10-0aa1-565600c916c4", "GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822"]
SOURCE = REPO / "data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl"
CAPACITIES = [0, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144]


def compile_collector(root):
    cuda = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13"
    target = root / "cupti_collector.so"
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC",
        str(Path(__file__).with_name("horizonkv_cupti.cpp")),
        "-I", str(cuda / "include"), "-I", "/usr/local/cuda-12.4/include",
        "-L", str(cuda / "lib"), "-Wl,-rpath," + str(cuda / "lib"),
        "-l:libcupti.so.13", "-o", str(target)], check=True)
    return target


def inventory(require_idle=False):
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,pci.bus_id,driver_version,gpu_recovery_action",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    rows = [dict(zip(["name", "uuid", "memory_mib", "free_mib", "pci_bus", "driver", "recovery_action"],
                     [v.strip() for v in row])) for row in csv.reader(text.splitlines())]
    selected = [next(r for r in rows if r["uuid"] == uid) for uid in GPU_IDS]
    if len({r["uuid"] for r in selected}) != 2 or any(r["name"] != "NVIDIA RTX A6000" for r in selected):
        raise RuntimeError("Paper evidence requires exactly the frozen two RTX A6000 GPUs")
    if any(r["recovery_action"] != "None" for r in selected):
        raise RuntimeError("An A6000 requires driver recovery; no GPU workload may start")
    if require_idle:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        if any(line.split(",")[0].strip() in GPU_IDS for line in apps.splitlines()):
            raise RuntimeError("A selected A6000 has another compute process")
        if any(int(r["free_mib"]) < 45000 for r in selected):
            raise RuntimeError("A selected A6000 has insufficient free memory")
    return selected


def notify(message):
    for attempt in range(3):
        try:
            config = json.loads(Path("/home10T/bzx/.codex/ntfy.json").read_text())
            request = urllib.request.Request(config["url"], data=message.encode(), method="POST",
                headers={"Authorization": "Bearer " + config["token"],
                         "Content-Type": "text/plain; charset=utf-8", "Title": "HorizonKV dual A6000"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status == 200
        except (OSError, ValueError, KeyError):
            if attempt < 2:
                time.sleep(5)
    return False


def monitor(root, owner):
    next_notice = 0
    while True:
        state = json.loads((root / "state.json").read_text())
        proc = Path(f"/proc/{owner}/cmdline")
        alive = proc.exists() and b"horizonkv_a6000.py" in proc.read_bytes()
        terminal = state["status"] in ("completed", "failed", "interrupted")
        if not alive and not terminal:
            state = dict(state, status="abnormal_exit")
            terminal = True
        if terminal or time.time() >= next_notice:
            detail = ""
            log = root / state.get("server_log", "missing")
            if log.is_file():
                detail = f"; log_age={int(time.time()-log.stat().st_mtime)}s"
            message = (f"HorizonKV 双 A6000: {state['status']}; {state.get('phase', '-')}; "
                f"完成 wave={state.get('completed_waves', 0)}/{state.get('total_waves', '?')}; "
                f"当前={state.get('current', '-')}{detail}")
            item = {"checked_unix_s": time.time(), "state": state, "ntfy_sent": notify(message)}
            try:
                item["gpu_inventory"] = inventory()
            except (OSError, subprocess.SubprocessError, StopIteration, RuntimeError) as exc:
                item["gpu_inventory_error"] = type(exc).__name__
            save(root / "monitor-status.json", item)
            with (root / "monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            next_notice = time.time() + 1200
        if terminal:
            return
        time.sleep(15)


def primary(row):
    return row["dataset"] == "RULER-v1" and row["task"] == "niah_multikey_1"


def pool_class(rows):
    needed = sum(r["prompt_tokens"] + 256 + 64 for r in rows)
    return next(p for p in (140000, 280000, 530000) if needed <= p // 64 * 64)


def freeze(root):
    groups = defaultdict(list)
    for line in SOURCE.read_text().splitlines():
        row = json.loads(line)
        groups[(row["dataset"], row["task"], row["length_bucket_upper"])].append(row)
    waves, skipped, breakdown = [], [], []
    for cell_index, (key, rows) in enumerate(sorted(groups.items())):
        rows = sorted(rows, key=lambda r: r["case_id"])
        chosen = rows if primary(rows[0]) else rows[:4]
        concurrency = 4 if len(chosen) >= 4 else 1
        for start in range(0, len(chosen), concurrency):
            batch = chosen[start:start+concurrency]
            if len(batch) != concurrency:
                skipped.append({"cell": key, "reason": "incomplete_distinct_request_wave", "cases": [r["case_id"] for r in batch]})
                continue
            waves.append({"cell": list(key), "cell_id": cell_index, "rows": batch, "concurrency": concurrency, "pool": pool_class(batch)})
        if primary(rows[0]):
            for concurrency in (1, 2):
                for start in range(0, 4, concurrency):
                    batch = rows[start:start+concurrency]
                    waves.append({"cell": list(key), "cell_id": cell_index, "rows": batch, "concurrency": concurrency, "pool": pool_class(batch)})
        # One fixed outcome-independent wave per breakdown cell. Longest C1
        # adds a feasible HBM counterpart if the C4 long pool cannot fit.
        if primary(rows[0]) or (key[2] == 32768 and key[1] in ("Single-Document QA", "gov_report", "repobench-p")):
            batch = rows[:4]
            breakdown.append({"cell": list(key), "cell_id": cell_index, "rows": batch, "concurrency": 4, "pool": pool_class(batch)})
            if primary(rows[0]) and key[2] == 131072:
                breakdown.append({"cell": list(key), "cell_id": cell_index, "rows": rows[:1], "concurrency": 1, "pool": 140000})
    for i, wave in enumerate(waves):
        wave["wave_id"] = i
    for i, wave in enumerate(breakdown):
        wave["wave_id"] = i
    requests = {r["case_id"]: r for w in waves + breakdown for r in w["rows"]}
    jobs = []
    # All-HBM is attempted first for each pool. A diagnosed OOM is recorded,
    # and collection falls back to exact DRAM storage on the same GPU pair.
    for pool in (140000, 280000, 530000):
        jobs.append({"id": f"trace-p{pool}", "phase": "trace", "mode": "hbm", "pool": pool,
                     "waves": [w for w in waves if w["pool"] == pool]})
    for pool in (140000, 280000, 530000):
        matching = [w for w in breakdown if w["pool"] == pool]
        for mode in ("hbm", "dram", "lru8192", "lru2048", "lru32768"):
            selected = matching if mode in ("hbm", "dram", "lru8192") else [w for w in matching if w["cell"][0] == "RULER-v1"]
            tasks = [(w, rep, phase) for w in selected for rep in range(5) for phase in ("control", "kernel")]
            random.Random(20261014).shuffle(tasks)
            jobs.append({"id": f"breakdown-p{pool}-{mode}", "phase": "breakdown", "mode": mode,
                         "pool": pool, "tasks": tasks})
    # Online trace/replay identity check at small/medium/large LRU budgets.
    for capacity in (2048, 8192, 32768):
        for pool in (140000, 280000, 530000):
            selected = [w for w in breakdown if primary(w["rows"][0]) and w["concurrency"] == 4 and w["pool"] == pool]
            jobs.append({"id": f"audit-p{pool}-lru{capacity}", "phase": "audit", "mode": f"lru{capacity}", "pool": pool, "waves": selected})
    plan = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "requests_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "gpu_uuids": GPU_IDS,
        "seed": 20261014, "topk": 2048, "layers": 48, "tp": 2, "capacities": CAPACITIES,
        "trace_max_new_tokens": 256, "breakdown_max_new_tokens": 64,
        "profile_start_decode": 32, "profile_steps": 8, "timing_repetitions": 5,
        "waves": [{k: v for k, v in w.items() if k != "rows"} | {"cases": [r["case_id"] for r in w["rows"]]} for w in waves],
        "skipped": skipped, "breakdown_cells": [{k: v for k, v in w.items() if k != "rows"} | {"cases": [r["case_id"] for r in w["rows"]]} for w in breakdown],
        "jobs": [{k: v for k, v in j.items() if k not in ("tasks", "waves")} | {"wave_count": len(j.get("waves", j.get("tasks", [])))} for j in jobs],
        "evidence_rule": "Only this dual-A6000 capture is eligible; simulation and measurement remain distinct; validation still required"}
    save(root / "plan.json", plan)
    with (root / "selected-requests.jsonl").open("w") as stream:
        for row in requests.values():
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return jobs, plan


class StartupError(RuntimeError):
    pass


def start_server(root, job, mode, state, port):
    hardware = inventory(require_idle=True)
    live = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if live != state["source_commit"] or dirty:
        raise RuntimeError("Frozen source checkout changed; refusing to mix versions")
    directory = root / job["id"]
    directory.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KEYE_", "SGLANG_"))}
    env.update(CUDA_VISIBLE_DEVICES=",".join(GPU_IDS), CUDA_DEVICE_ORDER="PCI_BUS_ID", KEYE_SM80_DSA="1",
        KEYE_SM80_EXACT_TOPK="1", KEYE_HORIZON_COLD_WAVES="1", PYTHONUNBUFFERED="1",
        KEYE_HORIZON_STORAGE="lru" if mode.startswith("lru") else mode,
        KEYE_HORIZON_CACHE_TOKENS=mode[3:] if mode.startswith("lru") else "0")
    if job["phase"] in ("trace", "audit"):
        env["KEYE_HORIZON_CAPACITY_DIR"] = str(directory / "trace")
    else:
        env["KEYE_HORIZON_KERNEL_DIR"] = str(directory / "profiles")
        env["KEYE_HORIZON_CUPTI_LIBRARY"] = str(root / "cupti_collector.so")
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", "/Tan/model/Keye-VL-2.0-30B-A3B",
        "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port), "--tp-size", "2",
        "--attention-backend", "keye_sa", "--context-length", "131328", "--max-total-tokens", str(job["pool"]),
        "--max-running-requests", "4", "--chunked-prefill-size", "1024", "--mem-fraction-static", "0.90",
        "--disable-cuda-graph", "--disable-custom-all-reduce", "--disable-radix-cache", "--random-seed", "20261014"]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            raise RuntimeError("Requested server port is already occupied")
    except ConnectionRefusedError:
        pass
    log = directory / f"server-{mode}.log"
    state.update(current=job["id"], server_log=str(log.relative_to(root)))
    save(root / "state.json", state)
    output = log.open("w")
    proc = subprocess.Popen(command, env=env, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    output.close()
    save(directory / f"startup-{mode}.json", {"pid": proc.pid, "command": command,
        "environment": {k: v for k, v in env.items() if k.startswith(("KEYE_", "CUDA_"))},
        "source_commit": live, "gpu_inventory": inventory()})
    try:
        deadline = time.time() + 1200
        while time.time() < deadline:
            if proc.poll() is not None:
                raise StartupError(log.read_text()[-12000:])
            if "The server is fired up and ready to roll!" in log.read_text()[-12000:]:
                return proc, directory
            time.sleep(2)
        raise StartupError("readiness timeout")
    except BaseException:
        stop_server(proc)
        raise


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def execute(root, jobs, plan, state, port):
    infeasible = set()
    reference = {}
    for job in jobs:
        directory = root / job["id"]
        if (directory / "completed.json").exists():
            continue
        tasks = job.get("tasks") or [(w, 0, "capacity") for w in job.get("waves", [])]
        if not tasks:
            continue
        mode = job["mode"]
        state.update(phase=job["phase"], current=job["id"], updated_unix_s=time.time())
        save(root / "state.json", state)
        if mode == "hbm" and job["pool"] in infeasible and job["phase"] == "breakdown":
            directory.mkdir(exist_ok=True)
            save(directory / "infeasible.json", {"reason": "same pool HBM startup OOM", "pool": job["pool"]})
            state["skipped_waves"] += len(tasks)
            continue
        try:
            proc, directory = start_server(root, job, mode, state, port)
        except StartupError as exc:
            text = str(exc)
            oom = "out of memory" in text.lower() or "not enough memory" in text.lower() or "Not enough GPU memory" in text
            if mode != "hbm" or not oom:
                raise
            infeasible.add(job["pool"])
            save(directory / "infeasible-hbm.json", {"reason": "observed_startup_OOM", "pool": job["pool"], "log": f"server-{mode}.log"})
            notify(f"HorizonKV 双 A6000: 全 HBM pool={job['pool']} 启动 OOM，已记录不可行点；统计采集改走精确 DRAM 路径。")
            if job["phase"] != "trace":
                state["skipped_waves"] += len(tasks)
                continue
            mode = "dram"
            proc, directory = start_server(root, job, mode, state, port)
        try:
            with (directory / "requests.jsonl").open("a", buffering=1) as stream:
                for wave, rep, phase in tasks:
                    key = f"{phase}__{job['id']}__wave{wave['wave_id']:03d}-r{rep}"
                    state.update(current=key, updated_unix_s=time.time())
                    save(root / "state.json", state)
                    result = run_wave(f"http://127.0.0.1:{port}", wave["rows"], key,
                        plan["trace_max_new_tokens"] if job["phase"] in ("trace", "audit") else plan["breakdown_max_new_tokens"], 14400)
                    for value in result:
                        value.update(wave_index=wave["wave_id"], cell=wave["cell"], concurrency=wave["concurrency"],
                            repetition=rep, phase=phase, storage=mode, pool=job["pool"], gpu_uuids=GPU_IDS)
                        ref_key = (value["case_id"], wave["concurrency"])
                        tokens = value["response"]["output_ids"]
                        if job["phase"] == "trace":
                            reference[ref_key] = tokens
                        else:
                            expected = reference.get(ref_key)
                            value["output_ids_equal_trace_prefix"] = expected is not None and tokens == expected[:len(tokens)]
                        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if any(v.get("output_ids_equal_trace_prefix") is False for v in result):
                        raise RuntimeError("Exact output prefix gate failed; retain evidence and investigate")
                    state["completed_waves"] += 1
                    state["updated_unix_s"] = time.time()
                    save(root / "state.json", state)
                    print(json.dumps({"completed_waves": state["completed_waves"], "task": key}), flush=True)
            time.sleep(5)
        finally:
            stop_server(proc)
        save(directory / "completed.json", {"finished_unix_s": time.time(), "mode": mode, "tasks": len(tasks)})
    state.update(phase="replay-and-audit", current="CPU simulation of new dual-A6000 traces", updated_unix_s=time.time())
    save(root / "state.json", state)
    subprocess.run([sys.executable, str(Path(__file__).with_name("horizonkv_a6000_analyze.py")), "--run-dir", str(root)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--monitor", type=int)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--port", type=int, default=31016)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.monitor:
        monitor(root, args.monitor)
        return
    root.mkdir(parents=True, exist_ok=True)
    if (root / "state.json").exists():
        raise RuntimeError("Run directory already has execution state; use a new version to preserve raw evidence")
    hardware = inventory()
    compile_collector(root)
    jobs, plan = freeze(root)
    if args.smoke:
        # A real-model end-to-end gate, separate from formal evidence.
        trace = next(j for j in jobs if j["id"] == "trace-p140000")
        wave = next(w for w in trace["waves"] if primary(w["rows"][0]) and w["concurrency"] == 4)
        trace["waves"] = [wave]
        smoke_jobs = [trace]
        for mode in ("hbm", "dram", "lru8192"):
            job = next(j for j in jobs if j["id"] == f"breakdown-p140000-{mode}")
            job["tasks"] = [t for t in job["tasks"] if primary(t[0]["rows"][0]) and t[0]["cell"][2] == 32768 and t[1] == 0]
            smoke_jobs.append(job)
        smoke_jobs.append(next(j for j in jobs if j["id"] == "audit-p140000-lru8192"))
        jobs = smoke_jobs
        plan.update(smoke_only=True, trace_max_new_tokens=64,
                    jobs=[{"id": j["id"], "wave_count": len(j.get("tasks", j.get("waves", [])))} for j in jobs])
        save(root / "plan.json", plan)
    if args.prepare:
        print(json.dumps({k: plan[k] for k in ("jobs", "skipped")}))
        return
    state = {"status": "running", "phase": "startup", "source_commit": plan["source_commit"],
        "completed_waves": 0, "skipped_waves": 0, "total_waves": sum(j["wave_count"] for j in plan["jobs"]),
        "updated_unix_s": time.time(), "gpu_inventory": hardware}
    save(root / "state.json", state)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--run-dir", str(root), "--monitor", str(os.getpid())],
                     stdout=(root / "monitor.log").open("a"), stderr=subprocess.STDOUT, start_new_session=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        execute(root, jobs, plan, state, args.port)
        state.update(status="completed", phase="finished", updated_unix_s=time.time())
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=type(exc).__name__ + ": " + str(exc)[-300:], updated_unix_s=time.time())
        raise
    finally:
        save(root / "state.json", state)


if __name__ == "__main__":
    main()
