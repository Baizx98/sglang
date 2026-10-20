"""Collect real layer inputs on the healthy A6000 + L40S; no timing claims."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from horizonkv_single_runtime import INPUT_GPUS, REPO, notify, save, update
from run_horizonkv_motivation import run_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--port", type=int, default=31120)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--wave-start", type=int, default=0)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    save(root / "collection-source.json", {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": sys.executable, "input_gpus": INPUT_GPUS,
        "wave_start": args.wave_start, "samples_per_task": args.samples,
        "max_output_tokens": args.steps, "timing_evidence": False,
    })
    source = REPO / "data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl"
    groups = defaultdict(list)
    tasks = {"niah_multikey_1", "Single-Document QA", "gov_report", "repobench-p"}
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["task"] in tasks and row["length_bucket_upper"] == 32768:
            groups[row["task"]].append(row)
    waves = []
    for task in ("niah_multikey_1", "Single-Document QA", "gov_report", "repobench-p"):
        rows = sorted(groups[task], key=lambda r: r["case_id"])[:args.samples]
        waves.extend([task, rows[start:start + 4]] for start in range(0, len(rows), 4))
    save(root / "input-manifest.json", {"source": str(source), "waves": waves,
                                      "gpus": INPUT_GPUS, "purpose": "model-derived inputs only"})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(INPUT_GPUS),
               KEYE_SM80_DSA="1", KEYE_SM80_EXACT_TOPK="1",
               KEYE_LAYER_REPLAY_DIR=str(root / "fixtures"),
               KEYE_LAYER_REPLAY_LAYERS="11,12,23,24",
               KEYE_LAYER_REPLAY_STEPS=str(args.steps),
               OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", "/Tan/model/Keye-VL-2.0-30B-A3B",
               "--trust-remote-code", "--host", "127.0.0.1", "--port", str(args.port),
               "--tp-size", "2", "--attention-backend", "keye_sa",
               "--context-length", "131328", "--max-total-tokens", "140000",
               "--max-running-requests", "4", "--chunked-prefill-size", "1024",
               "--mem-fraction-static", "0.85", "--disable-cuda-graph",
               "--disable-custom-all-reduce", "--disable-radix-cache",
               "--disable-overlap-schedule", "--dtype", "bfloat16"]
    save(root / "capture-command.json", {"command": command, "gpus": INPUT_GPUS})
    server = None
    try:
        with (root / "capture-server.log").open("w") as log:
            server = subprocess.Popen(command, env=env, cwd=REPO, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        update(root, phase="loading_input_model", child_pid=server.pid,
               detail="健康 A6000+L40S 仅采集真实输入，非性能结果。")
        url = f"http://127.0.0.1:{args.port}"
        deadline = time.time() + 900
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"input server exited {server.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("input server startup")
        if not args.skip_smoke:
            smoke = dict(waves[0][1][0])
            smoke["input_ids"] = smoke["input_ids"][:512]
            smoke["case_id"] = "input-smoke-512"
            results = run_wave(url, [smoke], "replay__smoke", 2, 300)
            save(root / "input-smoke-response.json", results)
            paths = list((root / "fixtures").glob("replay__smoke*history.pt"))
            if len(paths) != 8:
                raise RuntimeError(f"smoke history count {len(paths)}, expected four layers x two ranks")
            notify("HorizonKV：真实输入采集 smoke 通过，四层×两 rank 的历史 KV 与层输入已保存；开始32K四类负载。")
        if args.smoke_only:
            update(root, phase="input_smoke_ready", detail="真实512-token smoke完成；性能测量尚未开始。")
            return
        for index, (task, rows) in enumerate(waves):
            if index < args.wave_start:
                continue
            update(root, phase="collecting_real_inputs", completed_waves=index,
                   total_waves=len(waves), detail=f"{task} wave {index + 1}/{len(waves)}")
            result = run_wave(url, rows, f"replay__wave{index:03d}", args.steps, 7200)
            save(root / f"input-wave{index:03d}.json", result)
            if index == 0 or index + 1 == len(waves):
                notify(f"HorizonKV：真实32K输入采集 {index + 1}/{len(waves)} wave 完成；{task}。")
        update(root, phase="real_inputs_ready", completed_waves=len(waves),
               detail="真实输入采集完成；下一阶段仅健康 A6000 回放计时。")
        save(root / "input-collection-complete.json", {"wave_start": args.wave_start,
                                                      "completed_waves": len(waves),
                                                      "total_waves": len(waves)})
    except Exception as exc:
        update(root, phase="input_capture_error", detail=f"{type(exc).__name__}: {exc}")
        notify(f"HorizonKV 输入采集失败：{type(exc).__name__}: {exc}；日志已保存，检查后继续。")
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        update(root, child_pid=None)


if __name__ == "__main__":
    main()
