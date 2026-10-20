"""Run-state and ntfy support for the single-A6000 motivation experiments."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

GPU = "GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822"
INPUT_GPUS = [GPU, "GPU-79886749-8f57-63a5-8c8e-8a68a085565a"]
REPO = Path(__file__).resolve().parents[2]


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def notify(message):
    try:
        cfg = json.loads(Path("/home10T/bzx/.codex/ntfy.json").read_text())
        request = urllib.request.Request(
            cfg["url"], data=message.encode(), method="POST",
            headers={"Authorization": "Bearer " + cfg["token"],
                     "Content-Type": "text/plain; charset=utf-8",
                     "Title": "HorizonKV single A6000"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except (OSError, ValueError, KeyError):
        return False


def inventory():
    value = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = [line.strip() for line in value.splitlines()]
    return [row for row in rows if any(uid in row for uid in INPUT_GPUS)]


def update(root, **changes):
    path = Path(root) / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(changes, updated_unix_s=time.time())
    save(path, state)
    return state


def monitor(root):
    root = Path(root)
    next_notice = time.time() + 1200
    while True:
        state = json.loads((root / "state.json").read_text())
        terminal = state.get("status") in {"completed", "failed", "interrupted"}
        if time.time() >= next_notice or terminal:
            record = {"checked_unix_s": time.time(), "state": state,
                      "gpu_inventory": inventory()}
            pid = state.get("child_pid")
            record["child_alive"] = bool(pid and Path(f"/proc/{pid}").exists())
            # File counts and timestamps supplement process liveness.
            record["fixture_files"] = len(list((root / "fixtures").glob("*.pt")))
            record["result_files"] = len(list((root / "results").glob("*.json")))
            age = int(time.time() - state["updated_unix_s"])
            message = (f"HorizonKV v05：{state.get('status')}；{state.get('phase')}；"
                       f"输入文件={record['fixture_files']}，结果文件={record['result_files']}；"
                       f"状态更新距今 {age}s。{state.get('detail', '')}")
            record["ntfy_sent"] = notify(message)
            save(root / "monitor-status.json", record)
            with (root / "monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            next_notice = time.time() + 1200
        if terminal:
            return
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=Path, required=True)
    monitor(parser.parse_args().monitor)
