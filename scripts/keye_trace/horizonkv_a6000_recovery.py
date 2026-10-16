"""Wait read-only for repaired GPUs, then run gates and the frozen rerun once.

This process survives a GPU reset, not a machine reboot. It never resets a GPU
and never substitutes the L40S. Failure of either gate stops the queue.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from horizonkv_a6000 import GPU_IDS, REPO, inventory, notify, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "state.json").exists():
        raise RuntimeError("Supervisor directory already exists; use a new version")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    state = {"status": "waiting_for_gpu_recovery", "source_commit": commit,
             "updated_unix_s": time.time(), "owner_pid": os.getpid(),
             "smoke_dir": str(args.smoke_dir.resolve()), "formal_dir": str(args.formal_dir.resolve())}
    next_notice, healthy_reads = 0, 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    child = None
    try:
        while healthy_reads < 3:
            try:
                cards = inventory(require_idle=True)
                healthy = all(int(r["free_mib"]) >= 45000 for r in cards)
                state.update(gpu_inventory=cards, last_error=None)
            except (OSError, subprocess.SubprocessError, StopIteration, RuntimeError) as exc:
                healthy = False
                state["last_error"] = type(exc).__name__ + ": " + str(exc)
            healthy_reads = healthy_reads + 1 if healthy else 0
            state.update(updated_unix_s=time.time(), healthy_consecutive_checks=healthy_reads)
            if time.time() >= next_notice:
                state["ntfy_sent"] = notify("HorizonKV 双 A6000：等待 GPU 恢复，正式测量尚未启动。" +
                    ("GPU 状态初步恢复，连续检查中。" if healthy else "GPU0 驱动要求 Reset；后台每分钟检查，每20分钟通知。"))
                next_notice = time.time() + 1200
            save(root / "state.json", state)
            if healthy_reads < 3:
                time.sleep(60)
        live = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
        if live != commit or dirty:
            raise RuntimeError("Source changed while waiting; refusing to use an unfrozen version")
        notify("HorizonKV：两张 A6000 已连续3次健康且空闲，先重跑正确性检查与真实模型预检，通过后自动启动正式矩阵。")
        state.update(status="running", phase="gpu_correctness")
        save(root / "state.json", state)
        checks = [(GPU_IDS[0], "test_horizonkv_capacity.py", ["--gpu"]),
                  (GPU_IDS[1], "test_horizonkv_host_store.py", []),
                  (GPU_IDS[0], "test_horizonkv_cupti.py", []),
                  (GPU_IDS[1], "test_horizonkv_cupti.py", [])]
        for gpu, script, extra in checks:
            with (root / f"{script}-{GPU_IDS.index(gpu)}.log").open("w") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name(script)), *extra],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if child.wait(timeout=600):
                    raise RuntimeError(f"GPU correctness gate failed: {script}")
        for label, directory, flags in [("smoke", args.smoke_dir, ["--smoke"]), ("formal", args.formal_dir, [])]:
            state.update(phase=label, child_run_dir=str(directory.resolve()), updated_unix_s=time.time())
            save(root / "state.json", state)
            with (root / f"{label}-runner.log").open("w") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("horizonkv_a6000.py")),
                    "--run-dir", str(directory.resolve()), *flags], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if child.wait():
                    raise RuntimeError(f"{label} failed; formal run cannot bypass the smoke gate")
        state.update(status="completed", phase="finished")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=type(exc).__name__+": "+str(exc))
        notify("HorizonKV 恢复队列停止：" + state["error"])
        raise
    finally:
        state["updated_unix_s"] = time.time()
        save(root / "state.json", state)


if __name__ == "__main__":
    main()
