"""Attribute actual CUDA activities through their CPU launch correlations.

Profiler record-function device totals can include launch gaps. Sum GPU kernel,
copy and memset durations instead; keep these separate from elapsed spans.
"""

import json
from collections import defaultdict
from pathlib import Path


def kernel_work_from_trace(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    ranges = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("layer")]
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                and e.get("args", {}).get("correlation") is not None}
    work = defaultdict(float)
    count = defaultdict(int)
    for event in events:
        if event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        parent = None
        if launch is not None:
            matching = [r for r in ranges if r["tid"] == launch["tid"]
                        and r["ts"] <= launch["ts"] <= r["ts"] + r["dur"]]
            if matching:
                parent = min(matching, key=lambda r: r["dur"])["name"]
        key = parent or "unattributed"
        work[key] += event["dur"]
        count[key] += 1
    return {"component_gpu_work_us": dict(work), "activity_counts": dict(count),
            "gpu_work_total_us": sum(work.values()),
            "contract": "sum of actual GPU kernel/copy/memset durations; not critical-path elapsed"}
