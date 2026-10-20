"""Bounded dual-A6000 compute, NCCL and real Keye TP2 functional validation."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from horizonkv_a6000 import GPU_IDS, REPO, inventory, notify, save


def gpu_check():
    import torch
    torch.cuda.set_device(0)
    memory = torch.ones(64 * 1024 * 1024, device="cuda", dtype=torch.float32)
    assert memory.sum().item() == memory.numel()
    matrix = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        result = matrix @ matrix
        assert torch.all(result == 1024).item()
    torch.cuda.synchronize()
    print(json.dumps({"gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                      "memory_bytes_checked": memory.numel() * memory.element_size(),
                      "gemm_repetitions": 20, "passed": True}), flush=True)


def nccl_check():
    import torch
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    for _ in range(20):
        value = torch.full((1024 * 1024,), float(rank + 1), device="cuda")
        dist.all_reduce(value)
        assert torch.all(value == 3).item()
    gathered = [torch.empty(1, device="cuda") for _ in range(2)]
    dist.all_gather(gathered, torch.tensor([float(rank)], device="cuda"))
    assert [x.item() for x in gathered] == [0, 1]
    torch.cuda.synchronize()
    print(json.dumps({"rank": rank, "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
                      "all_reduce_repetitions": 20, "elements_per_rank": 1024*1024,
                      "all_gather_passed": True}), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.single:
        return gpu_check()
    if args.worker:
        return nccl_check()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "phase": "inventory", "gpus": GPU_IDS}
    child = None
    last_notice = time.time()

    def update(phase):
        state.update(phase=phase, updated_unix_s=time.time())
        save(root / "state.json", state)

    def tick():
        nonlocal last_notice
        if time.time() - last_notice >= 1200:
            cards = inventory()
            notify(f"双A6000 TP验证仍在运行：{state['phase']}；子进程存活={child.poll() is None}；" +
                   "; ".join(f"{c['uuid']}: recovery={c['recovery_action']}" for c in cards))
            last_notice = time.time()

    def run(command, env, name, timeout):
        nonlocal child
        update(name)
        with (root / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while child.poll() is None and time.time() < deadline:
            tick()
            time.sleep(2)
        if child.poll() is None:
            raise TimeoutError(name)
        if child.returncode:
            raise RuntimeError(f"{name} exited {child.returncode}; see log")

    try:
        cards = inventory(require_idle=True)
        save(root / "environment.json", {"cards": cards,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "python": sys.executable, "started_unix_s": time.time(),
             "scope": "functional validation only; no long-run stability or performance claim"})
        notify("开始双A6000 TP验证：当前两卡空闲且recovery=None；先单卡计算、NCCL，再真实Keye TP2短/32K推理。L40S不参与。")
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        for rank, uid in enumerate(GPU_IDS):
            run([sys.executable, __file__, "--single"], dict(env, CUDA_VISIBLE_DEVICES=uid), f"single_gpu_{rank}", 120)
        both = dict(env, CUDA_VISIBLE_DEVICES=",".join(GPU_IDS), NCCL_DEBUG="INFO")
        run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
             __file__, "--worker"], both, "nccl_pair", 180)
        notify("双A6000：两卡计算与20轮NCCL all-reduce/all-gather通过，开始加载真实Keye TP2模型。")
        port = 31220
        both.update(KEYE_SM80_DSA="1", KEYE_SM80_EXACT_TOPK="1")
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", "/Tan/model/Keye-VL-2.0-30B-A3B",
                   "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port), "--tp-size", "2",
                   "--attention-backend", "keye_sa", "--context-length", "40000", "--max-total-tokens", "140000",
                   "--max-running-requests", "4", "--chunked-prefill-size", "1024", "--mem-fraction-static", "0.85",
                   "--disable-cuda-graph", "--disable-custom-all-reduce", "--disable-radix-cache",
                   "--disable-overlap-schedule", "--dtype", "bfloat16"]
        save(root / "server-command.json", {"command": command, "gpus": GPU_IDS,
                                            "mode": "TP2 eager; NCCL; exact Top-K; text-only"})
        update("loading_tp2_model")
        with (root / "server.log").open("w") as log:
            child = subprocess.Popen(command, cwd=REPO, env=both, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 900
        while time.time() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"TP2 server exited {child.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            tick()
            time.sleep(3)
        else:
            raise TimeoutError("model startup")
        from run_horizonkv_motivation import run_wave
        source = REPO / "data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl"
        requests = [json.loads(s) for s in source.read_text().splitlines()]
        rows = sorted([r for r in requests if r["task"] == "niah_multikey_1" and
                       r["length_bucket_upper"] == 32768], key=lambda r: r["case_id"])[:4]
        assert len(rows) == 4
        smoke = dict(rows[0], input_ids=rows[0]["input_ids"][:512], case_id="tp-smoke-512")
        for label, batch, steps in [("short512", [smoke], 16), ("c4_32k_a", rows, 32), ("c4_32k_b", rows, 32)]:
            update(label)
            result = run_wave(url, batch, label, steps, 600)
            assert len(result) == len(batch)
            assert all(r["response"]["output_ids"] for r in result)
            save(root / (label + ".json"), result)
        a = json.loads((root / "c4_32k_a.json").read_text())
        b = json.loads((root / "c4_32k_b.json").read_text())
        identical = [x["response"]["output_ids"] == y["response"]["output_ids"] for x, y in zip(a, b)]
        save(root / "repeat-check.json", {"output_ids_equal": identical,
                                          "note": "repeat determinism within this TP2 run, not TP1 or quality equivalence"})
        state.update(status="completed", repeat_output_ids_equal=identical, cards_after=inventory())
        notify(f"双A6000 TP2验证通过：两卡计算、NCCL、512短请求及两轮C4/约32K推理完成；重复token一致={all(identical)}。只代表本次功能验证，未启动正式矩阵。")
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        notify("双A6000 TP验证停止：" + state["error"] + "；日志保留，未启动正式矩阵。")
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["updated_unix_s"] = time.time()
        save(root / "state.json", state)


if __name__ == "__main__":
    main()
