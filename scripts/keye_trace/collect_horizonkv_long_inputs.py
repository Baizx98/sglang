"""Collect real C1 long-context fixtures on A6000+L40S, excluding all timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from horizonkv_single_runtime import GPU, INPUT_GPUS, REPO, notify, save, update
from run_horizonkv_motivation import run_wave


def freeze(root, samples):
    source = REPO / "data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    ruler = lambda bucket: sorted([r for r in rows if r["task"] == "niah_multikey_1" and
                                   r["length_bucket_upper"] == bucket], key=lambda r: r["case_id"])[:samples]
    qa = lambda target, lo, hi: sorted([r for r in rows if r["task"] == "Single-Document QA" and
                                      lo <= r["prompt_tokens"] <= hi],
                                     key=lambda r: (abs(r["prompt_tokens"]-target), r["case_id"]))[:samples]
    groups = [("QA16K", qa(16384, 8192, 16384)), ("RULER32K", ruler(32768)),
              ("RULER64K", ruler(65536)), ("QA100K", qa(100000, 90000, 110000)),
              ("RULER128K", ruler(131072))]
    waves = []
    group_records = []
    for label, selected in groups:
        assert len(selected) == samples, (label, len(selected))
        first = len(waves)
        for row in selected:
            assert row["prompt_tokens"] + 48 <= 131328
            waves.append(dict(wave=len(waves), group=label, rows=[row]))
        group_records.append(dict(label=label, first_wave=first, waves=list(range(first, len(waves))),
                                  case_ids=[r["case_id"] for r in selected],
                                  actual_input_tokens=[r["prompt_tokens"] for r in selected]))
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    selection="Outcome-independent first case IDs for RULER; closest actual input length for QA; no truncation/repetition",
                    seed=20261014, groups=group_records, waves=waves,
                    input_gpus=INPUT_GPUS, capture_concurrency=1,
                    note="C1 collection avoids full-model HBM OOM; any merged C4 layer replay is not C4 end-to-end serving")
    save(root / "long-input-manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--start-wave", type=int, default=0)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest_path = root / "long-input-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze(root, args.samples)
    if args.freeze_only:
        print(json.dumps(manifest["groups"]))
        return
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", "/Tan/model/Keye-VL-2.0-30B-A3B",
               "--trust-remote-code", "--host", "127.0.0.1", "--port", "31121", "--tp-size", "2",
               "--attention-backend", "keye_sa", "--context-length", "131328", "--max-total-tokens", "140000",
               "--max-running-requests", "1", "--chunked-prefill-size", "1024", "--mem-fraction-static", "0.85",
               "--disable-cuda-graph", "--disable-custom-all-reduce", "--disable-radix-cache",
               "--disable-overlap-schedule", "--dtype", "bfloat16", "--random-seed", "20261014"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(INPUT_GPUS), KEYE_SM80_DSA="1",
               KEYE_SM80_EXACT_TOPK="1", KEYE_LAYER_REPLAY_DIR=str(root / "fixtures"),
               KEYE_LAYER_REPLAY_LAYERS="1,12,24,36,47", KEYE_LAYER_REPLAY_STEPS="40",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    save(root / "long-capture-command.json", dict(command=command, gpus=INPUT_GPUS,
         source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
         timing_evidence=False))
    server = None
    try:
        with (root / "long-capture-server.log").open("w") as log:
            server = subprocess.Popen(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(root, phase="loading_long_input_model", child_pid=server.pid, total_waves=len(manifest["waves"]),
               detail="健康A6000+L40S只采集输入；C1，最多约131K，无故障卡。")
        deadline = time.time() + 900
        url = "http://127.0.0.1:31121"
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"long input server exited {server.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(3)
        else:
            raise TimeoutError("long input server startup")
        for ordinal, item in enumerate(manifest["waves"]):
            wave = item["wave"]
            if (root / f"input-wave{wave:03d}.json").exists():
                continue
            if wave < args.start_wave:
                continue
            update(root, phase="collecting_long_inputs", completed_waves=ordinal,
                   detail=f"{item['group']} C1 wave {ordinal+1}/{len(manifest['waves'])}; L={item['rows'][0]['prompt_tokens']}")
            result = run_wave(url, item["rows"], f"replay__wave{wave:03d}", 48, 1800)
            assert len(result) == 1 and result[0]["response"]["output_ids"]
            save(root / f"input-wave{wave:03d}.json", result)
            if (ordinal+1) % args.samples == 0:
                notify(f"HorizonKV长输入：{item['group']}真实输入{ordinal+1}/{len(manifest['waves'])}完成；采集耗时不作性能结果。")
        save(root / "long-input-complete.json", dict(waves=len(manifest["waves"])))
        update(root, phase="long_inputs_completed", completed_waves=len(manifest["waves"]),
               detail="长输入采集完成，释放两卡后单A6000层计时。")
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait()
        update(root, child_pid=None)


if __name__ == "__main__":
    main()
