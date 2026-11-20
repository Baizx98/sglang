"""Collect Keye layers for exact-vs-approximate prediction replay with PP2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.request

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/keye_trace"))

from horizonkv_predictive_runtime import inventory, post
from horizonkv_single_runtime import INPUT_GPUS


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def evaluate_quality(url, rows, args):
    import re
    from statistics import mean
    think_closed = [151667, 271, 151668, 271]
    rng = random.Random(args.seed)
    with (args.output / "task-quality.jsonl").open("x") as stream:
        for number, row in enumerate(rows, 1):
            modes = ["native", "bf16", "compact"]
            rng.shuffle(modes)
            for mode in modes:
                request = f"predict-eval__{mode}__{row['rid']}"
                print(f"quality {number}/{len(rows)} {mode} {row['rid']}", flush=True)
                result = post(url + "/generate", dict(
                    input_ids=row["input_ids"] + think_closed, rid=request,
                    sampling_params=dict(temperature=0, max_new_tokens=args.quality_max_tokens,
                                         ignore_eos=False),
                ))
                finish = result["meta_info"]["finish_reason"]["type"]
                tokens = result["output_ids"]
                if not tokens or finish not in {"stop", "length"}:
                    raise RuntimeError(f"incomplete quality request {request}: {finish}")
                text = re.sub(r"[\x00-\x1f]", "\n", result["text"]).strip()
                if "<think>" in text and "</think>" not in text:
                    answer = ""
                else:
                    answer = text.rsplit("</think>", 1)[-1].strip()
                hits = [entry.lower() in answer.lower() for entry in row["expected_answer"]]
                score = max(hits) if row["task"].startswith("qa_") else mean(hits)
                stream.write(json.dumps(dict(
                    rid=row["rid"], request=request, mode=mode, task=row["task"],
                    prompt_tokens=len(row["input_ids"]) + len(think_closed), finish=finish,
                    final_score=score, output_tokens=len(tokens), final_answer=answer,
                    output_ids=tokens,
                ), ensure_ascii=False) + "\n")
                stream.flush()


def run(args):
    rows = [json.loads(line) for line in args.requests.read_text().splitlines()]
    assert rows and len({row["rid"] for row in rows}) == len(rows)
    quality = args.quality_policy is not None
    for row in rows:
        assert row["rid"].startswith("predict-full-") and Path(row["rid"]).name == row["rid"]
        if quality:
            assert row["dataset"] == "RULER" and row["expected_answer"]
        room = args.quality_max_tokens + 4 if quality else args.steps + 1
        assert len(row["input_ids"]) + room <= args.context
    devices = inventory(INPUT_GPUS, require_idle=True)
    args.output.mkdir(parents=True, exist_ok=False)
    traces = args.output / "fixtures"
    traces.mkdir()
    env = dict(
        CUDA_VISIBLE_DEVICES=",".join(INPUT_GPUS),
        KEYE_SM80_DSA="1", KEYE_SM80_EXACT_TOPK="1",
        KEYE_LOOKAHEAD_TRACE_DIR=str(traces),
        KEYE_PREDICT_CAPTURE_LAYERS=",".join(map(str, args.layers)),
        KEYE_PREDICT_CAPTURE_STEPS=str(args.steps),
        KEYE_SM80_PREFILL_SCORE_CHUNK=str(args.score_chunk),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
    )
    if quality:
        policy = args.output / "policy.json"
        policy.write_bytes(args.quality_policy.read_bytes())
        env["KEYE_PREDICT_QUALITY_POLICY"] = str(policy)
        fi = ROOT / "data/indexer-optimization/fi074"
        dsl = ROOT / "data/indexer-optimization/fi-dsl"
        env["PYTHONPATH"] = os.pathsep.join(map(str, (
            fi, dsl / "nvidia_cutlass_dsl/dsl_packages", dsl,
        )))
        cuda = ROOT / "data/indexer-optimization/cuda130/nvidia/cu13"
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = f"{cuda / 'bin'}{os.pathsep}{os.environ['PATH']}"
        env["CPATH"] = os.pathsep.join(map(str, (
            cuda / "include/cccl", ROOT / ".venv/lib/python3.12/site-packages/nvidia/cu13/include",
        )))
        env["LIBRARY_PATH"] = str(cuda / "lib")
        env["FLASHINFER_CUDA_ARCH_LIST"] = "8.6;8.9"
    launcher = "lookahead_quality_server.py" if quality else "capture_server.py"
    command = [
        sys.executable, str(HERE / launcher),
        "--model-path", "/Tan/model/Keye-VL-2.0-30B-A3B",
        "--trust-remote-code", "--host", "127.0.0.1", "--port", str(args.port),
        "--tp-size", "1", "--pp-size", "2", "--attention-backend", "keye_sa",
        "--context-length", str(args.context), "--max-total-tokens", str(args.pool),
        "--max-running-requests", "1", "--chunked-prefill-size", str(args.prefill_chunk),
        "--mem-fraction-static", str(args.mem_fraction), "--disable-cuda-graph",
        "--disable-custom-all-reduce", "--disable-radix-cache", "--disable-overlap-schedule",
        "--skip-server-warmup", "--dtype", "bfloat16", "--random-seed", str(args.seed),
    ]
    save(args.output / "environment.json", dict(
        command=command, environment=env, devices=devices,
        scope="PP2 TP1 task quality only" if quality else "PP2 TP1 input capture only",
        layers=args.layers, steps=args.steps, seed=args.seed,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    ))
    (args.output / "requests.jsonl").write_bytes(args.requests.read_bytes())
    with (args.output / "server.log").open("x") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=os.environ | env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Input server exited with code {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/model_info", timeout=3):
                        break
                except OSError:
                    time.sleep(3)
            else:
                raise TimeoutError("Input server readiness")
            health = post(url + "/generate", dict(
                input_ids=rows[0]["input_ids"][:512], rid="predict-capture-health",
                sampling_params=dict(temperature=0, max_new_tokens=3, ignore_eos=True),
            ))
            assert len(health["output_ids"]) == 3
            save(args.output / "health.json", health)
            if quality:
                evaluate_quality(url, rows, args)
                return
            for count, row in enumerate(rows):
                rid = row["rid"]
                print(f"capture {count + 1}/{len(rows)} {rid} length={len(row['input_ids'])}", flush=True)
                result = post(url + "/generate", dict(
                    input_ids=row["input_ids"], rid=rid,
                    sampling_params=dict(temperature=0, max_new_tokens=args.steps + 1, ignore_eos=True),
                ))
                save(args.output / f"response-{rid}.json", result)
                if len(result["output_ids"]) != args.steps + 1:
                    raise RuntimeError(f"Incomplete input capture: {rid}: {result['meta_info']}")
                for layer in args.layers:
                    prefix = traces / f"{rid}-l{layer:02}"
                    keys = torch.load(f"{prefix}-keys.pt", weights_only=True, mmap=True)
                    assert keys.shape == (len(row["input_ids"]) + 1, 64)
                    assert keys.dtype == torch.bfloat16
                    for step in range(args.steps):
                        frame = torch.load(f"{prefix}-s{step:02}.pt", weights_only=True)
                        assert frame["length"] == keys.shape[0] + step
                        assert frame["request"] == rid and frame["layer"] == layer and frame["step"] == step
                        assert frame["source_hidden"].shape == frame["target_hidden"].shape
                        chosen = frame["target_topk"]
                        assert chosen.shape == (1, 2048) and chosen.unique().numel() == 2048
                        assert int(chosen.min()) >= 0 and int(chosen.max()) < frame["length"]
                        if step == 0:
                            assert torch.equal(keys[-1:].view(torch.int16), frame["target_key"].view(torch.int16))
                with (args.output / "completed.jsonl").open("a") as done:
                    done.write(json.dumps(dict(rid=rid, prompt_tokens=len(row["input_ids"]))) + "\n")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--pool", type=int, default=263168)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--score-chunk", type=int, default=16)
    parser.add_argument("--mem-fraction", type=float, default=0.98)
    parser.add_argument("--port", type=int, default=31248)
    parser.add_argument("--quality-policy", type=Path)
    parser.add_argument("--quality-max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261120)
    run(parser.parse_args())
