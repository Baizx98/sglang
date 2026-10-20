"""Execute the v05 representative M1/M3/M5 loop with real layer fixtures."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from horizonkv_single_runtime import GPU, REPO, notify, save, update
from horizonkv_layer_replay import (
    ReplayData, ReplayLayer, graph_measure, initialize, load_layer, pair_call, read_frames)

import torch


def choose_frames(frames, extended):
    active = [f for f in frames if len(f["request_ids"]) == 4]
    if not active:
        return []
    selected = [("first-c4", active[0])]
    if extended:
        later = next((f for f in active if min(f["steps"]) >= 32), None)
        if later is not None:
            selected.append(("after32", later))
    return selected


def io_counts(runner):
    indices = runner.last_topk
    valid = indices >= 0
    if runner.cache is None:
        misses = int(valid.sum()) if runner.mode == "d-staged" else 0
        prefetched = useful = 0
    else:
        initial = runner.initial_cache["map"].gather(1, indices.long().clamp_min(0))
        missing = valid & (initial < 0)
        misses = int(missing.sum())
        prefetched = int((runner.prefmap >= 0).sum()) if runner.mode != "l-cached" else 0
        useful = int((missing & (runner.prefmap.gather(1, indices.long().clamp_min(0)) >= 0)).sum()) if prefetched else 0
    transferred = (prefetched + misses - useful) * 2048
    if runner.mode == "no-io":
        transferred = (misses - useful) * 2048
    return {"selected_tokens": int(valid.sum()), "logical_miss_tokens": misses,
            "prefetched_tokens": prefetched, "useful_prefetched_tokens": useful,
            "logical_miss_bytes": misses * 2048, "actual_host_read_payload_bytes": transferred,
            "actual_host_write_payload_bytes": 0 if runner.k is not None else runner.batch * 2048}


def check_output(expected, reference):
    return all(torch.equal(a, b) for a, b in zip(expected, reference))


def resource_counts(runner):
    def nbytes(*values):
        return sum(v.numel() * v.element_size() for v in values)
    data, cache = runner.data, runner.cache
    persistent = nbytes(runner.k, runner.v) if runner.k is not None else 0
    metadata = nbytes(runner.prefmap, runner.prefcounts)
    reset_snapshot = 0
    if cache is not None:
        persistent = nbytes(cache.k, cache.v)
        metadata += nbytes(cache.map, cache.tag, cache.age, cache.slots, cache.rank, cache.misses)
        reset_snapshot = nbytes(*runner.initial_cache.values())
    return {"layer_parameter_bytes": nbytes(*runner.layer.parameters()),
            "host_history_allocated_bytes": nbytes(data.hostk, data.hostv),
            "host_history_logical_bytes": int(data.lengths.sum()) * 2048,
            "gpu_index_k_bytes": nbytes(data.index),
            "gpu_persistent_main_kv_bytes": persistent,
            "benchmark_stage_and_prefetch_scratch_bytes": nbytes(runner.stagek, runner.stagev, runner.prefk, runner.prefv),
            "gpu_cache_metadata_bytes": metadata,
            "measurement_reset_snapshot_bytes": reset_snapshot,
            "cache_tokens_per_request": cache.capacity if cache is not None else None,
            "cache_per_request_topk_multiple": cache.capacity / runner.topk if cache is not None else None,
            "cache_layer_total_topk_multiple": cache.batch * cache.capacity / runner.topk if cache is not None else None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--waves", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else root
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "timelines").mkdir(exist_ok=True)
    initialize_config = initialize()
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid != GPU.removeprefix("GPU-"):
        raise RuntimeError(f"unexpected measurement GPU {actual_uuid}")
    others = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                      "--format=csv,noheader,nounits"], text=True)
    if any(GPU in line and int(line.split(",")[0]) != os.getpid() for line in others.splitlines()):
        raise RuntimeError("A6000 is shared with another compute process; do not time it")
    update(root, phase="exclusive_a6000_layer_replay", child_pid=os.getpid(),
           detail="健康 A6000 独占，真实层/相邻层 CUDA Graph 计时；原始采集已暂停。")
    manifest = [json.loads(line) for line in (root / "fixtures/manifest_rank00.jsonl").read_text().splitlines()]
    total_cases = 0
    for wave in args.waves:
        for target_id in (12, 24):
            active = [r for r in manifest if r["layer"] == target_id and len(r["request_ids"]) == 4
                      and r["request_ids"][0].startswith(f"replay__wave{wave:03d}__")]
            snapshots = int(bool(active)) + int(args.extended and any(min(r["steps"]) >= 32 for r in active))
            total_cases += snapshots * (6 + (18 if args.extended else 6))
    update(root, completed_cases=0, total_cases=total_cases)
    notify("HorizonKV：输入与四路径/ECHO-style数值核对及CUDA Graph回放已通过，开始健康A6000独占M1/M3/M5代表点计时。")
    metadata = {"measurement_gpu_uuid": GPU, "gpu_name": torch.cuda.get_device_name(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "tp": 1, "dtype": "bfloat16", "seed": 20261014,
                "timing": "3 eager compile warmups, 25 graph warmups, 5 timed graph repeats; event-node graph and Kineto separate",
                "source": "healthy A6000 + L40S TP2 model-derived inputs",
                "cache": "per-request HiSparse recency refinement; GPU touch/sort/gather port",
                "echo": "EMA alpha=0.5; Ampere fused tile score/recall; bounded k-slot scratch; no TMA warp specialization",
                "moe": "SGLang Triton default config, not A6000 tuned",
                "started_unix_s": time.time()}
    save(out / "measurement-environment.json", metadata)
    layers = {}
    completed = 0
    for wave in args.waves:
        for target_id in (12, 24):
            for layer_id in (target_id - 1, target_id):
                if layer_id not in layers:
                    layers[layer_id] = load_layer(initialize_config, layer_id)
            frames = read_frames(root, wave, target_id)
            previous_frames = read_frames(root, wave, target_id - 1)
            for snapshot, frame in choose_frames(frames, args.extended):
                data = ReplayData(root, frames, frame)
                previous_frame = next(f for f in previous_frames if f["request_ids"] == frame["request_ids"] and f["steps"] == frame["steps"])
                previous_data = ReplayData(root, previous_frames, previous_frame)
                baseline = truth = None
                for mode in ("h-native", "h-staged", "d-staged", "l-cached", "no-io", "echo"):
                    runner = ReplayLayer(layers[target_id], data, mode)
                    if mode == "no-io":
                        runner.prepare_no_io(truth)
                    if mode == "echo":
                        runner.prepare_echo()
                    label = f"wave{wave:03d}-l{target_id:02d}-{snapshot}-{mode}"
                    result, output = graph_measure(runner.forward, runner.reset, [runner],
                                                   profile_path=out / "timelines" / f"{label}.json")
                    if baseline is None:
                        baseline, truth = output, runner.last_topk.clone()
                    if not check_output(output, baseline) or not torch.equal(runner.last_topk, truth):
                        raise RuntimeError(f"paired output changed: {label}")
                    result.update(window="single_layer", wave=wave, layer=target_id,
                                  snapshot=snapshot, mode=mode, request_ids=frame["request_ids"],
                                  decode_steps=frame["steps"], lengths=frame["seq_lens"].tolist(),
                                  capacity=4096, topk=2048, output_matches=True,
                                  **io_counts(runner), **resource_counts(runner))
                    save(out / "results" / f"{label}.json", result)
                    completed += 1
                    print(label, round(statistics.median(result["elapsed_us"]), 3), flush=True)
                    update(root, completed_cases=completed, detail=label)
                    del runner
                    gc.collect()
                # The actual previous layer produces the target input for every mode.
                for capacity in ([2048, 4096, 8192] if args.extended else [4096]):
                    reference = pair_truth = None
                    for mode in ("l-cached", "echo", "no-io", "query-ready", "ahead", "topk-ready"):
                        previous = ReplayLayer(layers[target_id - 1], previous_data, "l-cached", capacity)
                        target = ReplayLayer(layers[target_id], data, mode, capacity)
                        if mode == "echo":
                            target.prepare_echo()
                        if mode == "no-io":
                            target.prepare_no_io(pair_truth)
                        if mode == "topk-ready":
                            target.fixed_topk = pair_truth
                        call, reset, transfer_events = pair_call(previous, target, mode, pair_truth)
                        label = f"wave{wave:03d}-pair{target_id-1:02d}-{target_id:02d}-{snapshot}-b{capacity}-{mode}"
                        result, output = graph_measure(call, reset, [previous, target],
                                                       profile_path=out / "timelines" / f"{label}.json",
                                                       transfer_events=transfer_events if mode in {"ahead", "query-ready"} else None)
                        if reference is None:
                            reference, pair_truth = output, target.last_topk.clone()
                        if not check_output(output, reference) or not torch.equal(target.last_topk, pair_truth):
                            raise RuntimeError(f"two-layer output changed: {label}")
                        result.update(window="two_layers", wave=wave, layer=target_id,
                                      snapshot=snapshot, mode=mode, capacity=capacity, topk=2048,
                                      request_ids=frame["request_ids"], decode_steps=frame["steps"],
                                      lengths=frame["seq_lens"].tolist(), output_matches=True,
                                      **io_counts(target), **resource_counts(target))
                        save(out / "results" / f"{label}.json", result)
                        completed += 1
                        print(label, round(statistics.median(result["elapsed_us"]), 3), flush=True)
                        update(root, completed_cases=completed, detail=label)
                        del previous, target
                        gc.collect()
                notify(f"HorizonKV：wave{wave} 层{target_id} 的M1/M3/M5闭环完成，已核对 {completed} 组；原始时间线已保存。")
    update(root, phase="representative_loop_completed", completed_cases=completed, child_pid=None,
           detail="代表点闭环完成；等待汇总及扩展采集/容量扫描。")
    notify(f"HorizonKV：首轮单A6000代表点完成，共 {completed} 组，正在审计与整理。")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        notify(f"HorizonKV 单A6000回放异常：{type(exc).__name__}: {exc}；检查日志后继续。")
        raise
