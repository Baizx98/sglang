"""Audit real MoE inputs/weights and sparse-attention kernels on one A6000."""

import argparse
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import statistics

import torch

from horizonkv_layer_replay import initialize, load_layer, read_frames, ReplayData, ReplayLayer
from horizonkv_single_runtime import GPU, notify, save, update


@torch.inference_mode()
def benchmark(call, reset, repeats=10):
    for _ in range(3):
        reset(); call()
    torch.cuda.synchronize()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for _ in range(15):
        reset(); graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        reset()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    reset(); graph.replay(); torch.cuda.synchronize()
    return dict(median_us=statistics.median(times), repeats_us=times), output.clone()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    config = initialize()
    assert str(torch.cuda.get_device_properties(0).uuid) == GPU.removeprefix("GPU-")
    torch.manual_seed(20261014)
    frames = read_frames(args.source, 0, 12)
    frame = next(f for f in frames if len(f["request_ids"]) == 4 and min(f["steps"]) >= 32)
    data = ReplayData(args.source, frames, frame)
    layer = load_layer(config, 12)
    runner = ReplayLayer(layer, data, "h-native")
    captured = []
    hook = layer.mlp.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach().clone()))
    runner.reset(); runner.forward(); torch.cuda.synchronize(); hook.remove()
    actual_input = captured[-1]
    # Real selected-token attention must show sparse split/reduce kernels.
    runner.record_stages = True
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
        torch.ones(16, device="cuda").add_(1); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        runner.reset(); runner.forward(); torch.cuda.synchronize()
    profile = root / "sparse-kernel-audit.json"
    prof.export_chrome_trace(str(profile))
    trace = json.loads(profile.read_text())["traceEvents"]
    kernels = sorted({e["name"] for e in trace if e.get("cat") == "kernel"})
    sparse_names = [k for k in kernels if "_dsa_split_online" in k or "_dsa_reduce" in k]
    assert len(sparse_names) >= 2, "sparse decode split/reduce kernels not found"
    runner.record_stages = False
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    from sglang.srt.layers.moe.moe_runner import triton_utils

    candidates = [None] + [dict(BLOCK_SIZE_M=m, BLOCK_SIZE_N=n, BLOCK_SIZE_K=k,
                               GROUP_SIZE_M=1, num_warps=4, num_stages=2)
                           for m, n, k in itertools.product((8, 16), (32, 64, 128), (32, 64))]
    records = []
    best_by_batch = {}
    mlp = layer.mlp
    weights = {name: dict(shape=list(value.shape), bytes=value.numel()*value.element_size())
               for name, value in mlp.experts.named_parameters()}
    for batch in (1, 4):
        update(root, phase="moe_tuning", detail=f"真实层12、C{batch}；默认＋12个tile配置，保留专家路由与精度检查。")
        source = actual_input[:batch].clone()
        x = source.clone()
        reset = lambda: x.copy_(source)
        reset(); logits, _ = mlp.gate(x); routes = mlp.topk(x, logits)
        unique = torch.unique(routes.topk_ids).numel()
        bytes_per_expert = sum(v["bytes"] // v["shape"][0] for v in weights.values())
        default_output = None
        valid = []
        for index, candidate in enumerate(candidates):
            try:
                with override_config(candidate) if candidate else nullcontext():
                    timing, result = benchmark(lambda: mlp(x), reset)
                if default_output is None:
                    default_output = result
                max_abs = (result.float()-default_output.float()).abs().max().item()
                relative_l2 = ((result.float()-default_output.float()).norm() /
                               default_output.float().norm().clamp_min(1e-12)).item()
                torch.testing.assert_close(result, default_output, rtol=.01, atol=.001)
                record = dict(batch=batch, candidate=index, config=candidate, **timing,
                              max_abs_error=max_abs, relative_l2_error=relative_l2,
                              selected_expert_ids=routes.topk_ids.cpu().tolist(), unique_experts=unique,
                              bytes_per_expert=bytes_per_expert,
                              unique_expert_weight_bytes=unique*bytes_per_expert, accepted=True)
                valid.append(record)
            except (RuntimeError, AssertionError) as exc:
                record = dict(batch=batch, candidate=index, config=candidate, accepted=False,
                              error=f"{type(exc).__name__}: {exc}")
            finally:
                triton_utils._config = None
            records.append(record)
            save(root / "moe-audit-candidates.json", records)
            print(batch, index, record.get("median_us", record.get("error")), flush=True)
        best = min(valid, key=lambda r: r["median_us"])
        best_by_batch[batch] = best["config"]
        reset()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("router_gate"):
                logits, _ = mlp.gate(x)
            with torch.profiler.record_function("expert_selection"):
                routes = mlp.topk(x, logits)
            with torch.profiler.record_function("expert_ffn"):
                mlp.experts(x, routes)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(root / f"moe-components-c{batch}.json"))
    summary = dict(status="completed", gpu_uuid=GPU, source=str(args.source.resolve()), layer=12,
                   sparse_decode_kernel_names=sparse_names, expert_weights=weights,
                   best_by_batch=best_by_batch, candidates=records,
                   scope="Real 32K snapshot MoE input; TP1 BF16; no communication/dequantization. Tile tuning changes kernel configuration only.")
    save(root / "moe-audit.json", summary)
    notify("HorizonKV：真实稀疏Attention split/reduce内核已确认；MoE默认与12个tile配置完成数值/计时核对，路由、专家权重与组件profile已保存。")
    update(root, phase="moe_audit_completed", detail="MoE与稀疏内核核查完成；准备长上下文真实输入采集。")


if __name__ == "__main__":
    main()
