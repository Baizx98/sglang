"""Continue the frozen 32K v05 queue after its representative loop.

Finish CPU cache replay, collect the remaining QA/summary/code inputs, release
the input GPUs, then measure only on the healthy A6000 and replay all caches.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from horizonkv_single_runtime import GPU, REPO, notify, save, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.wait_pid:
        while Path(f"/proc/{args.wait_pid}/cmdline").exists():
            cmdline = Path(f"/proc/{args.wait_pid}/cmdline").read_bytes()
            if b"replay_horizonkv_layer_capacity.py" not in cmdline:
                break
            time.sleep(10)
        assert (root / "m4-capacity-v01/audit.json").exists(), "initial CPU replay did not finish"
    # Retain the interrupted, unreturned input wave outside the active manifest.
    old = root / "interrupted-input-wave007"
    old.mkdir(exist_ok=True)
    for path in (root / "fixtures").glob("replay__wave007__*.pt"):
        path.rename(old / path.name)
    for rank in (0, 1):
        path = root / f"fixtures/manifest_rank{rank:02d}.jsonl"
        lines = path.read_text().splitlines()
        removed = [s for s in lines if "replay__wave007__" in s]
        if removed:
            (old / f"manifest_rank{rank:02d}.jsonl").write_text("\n".join(removed) + "\n")
        path.write_text("\n".join(s for s in lines if "replay__wave007__" not in s) + "\n")
    for name in ("capture-server.log", "capture-command.json", "input-manifest.json"):
        shutil.copy2(root / name, root / ("part1-" + name))
    python = str(REPO / ".venv/bin/python")
    jobs = [
        ("remaining_32k_inputs", [python, "scripts/keye_trace/collect_horizonkv_layer_inputs.py",
                                 "--run-dir", str(root), "--wave-start", "7", "--skip-smoke"], None),
        ("complete_32k_a6000_replay", [python, "scripts/keye_trace/run_horizonkv_layer_replay.py",
                                   "--run-dir", str(root), "--output-dir", str(root / "four-workloads-v01"),
                                   "--extended", "--waves", *map(str, range(20))], GPU),
        ("complete_32k_cache_replay", [python, "scripts/keye_trace/replay_horizonkv_layer_capacity.py",
                                      "--run-dir", str(root), "--output-dir", str(root / "m4-capacity-v02")], ""),
    ]
    save(root / "continuation-jobs.json", [{"phase": phase, "command": cmd} for phase, cmd, _ in jobs])
    child = None
    try:
        for phase, command, devices in jobs:
            update(root, phase=phase, detail="继续v05的32K四类负载；16/64/128K及全48层随后扩展。")
            notify(f"HorizonKV 进入下一阶段：{phase}。32K代表点已完成，继续四类负载，关键节点及20分钟通知保持运行。")
            env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
            if devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = devices
            with (root / f"{phase}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            save(root / "continuation-current.json", {"phase": phase, "pid": child.pid})
            code = child.wait()
            if code:
                raise RuntimeError(f"{phase} exit {code}")
            if phase == "remaining_32k_inputs":
                assert (root / "input-collection-complete.json").exists()
                assert len(list(root.glob("input-wave*.json"))) == 20
            elif phase == "complete_32k_a6000_replay":
                files = list((root / "four-workloads-v01/results").glob("*.json"))
                state = json.loads((root / "state.json").read_text())
                assert len(files) == state["total_cases"]
                assert all(json.loads(f.read_text())["output_matches"] for f in files)
            else:
                assert json.loads((root / "m4-capacity-v02/audit.json").read_text())["status"] == "completed"
        update(root, status="completed", phase="32k_four_workloads_completed", child_pid=None,
               detail="32K四类负载的本轮队列完成；更长上下文、全48层和强基线进一步核验尚待执行。")
        notify("HorizonKV：32K四类负载队列完成，计时仅来自健康A6000；LRU/Belady离线结果和原始时间线已保存。更长上下文与全48层仍待后续扩展。")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        update(root, status="failed", phase="continuation_failed", child_pid=None,
               detail=f"{type(exc).__name__}: {exc}")
        notify(f"HorizonKV 后续队列停止：{type(exc).__name__}: {exc}。已完成结果保留。")
        raise


if __name__ == "__main__":
    main()
