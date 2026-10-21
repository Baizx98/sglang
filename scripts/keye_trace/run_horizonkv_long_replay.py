"""Measure long-context native selection, sparse attention and KV demand paths."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from horizonkv_host_kernels import resident_gather
from horizonkv_layer_replay import initialize, load_layer, read_frames, ReplayData, ReplayLayer, graph_measure
from horizonkv_single_runtime import GPU, REPO, notify, save, update
from run_horizonkv_layer_replay import io_counts, resource_counts, check_output
from merge_horizonkv_long_frames import merge


def unsorted_topk(scores, lengths, k):
    cols = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    values, indices = torch.topk(scores.masked_fill(cols >= lengths.unsqueeze(1), -float("inf")),
                                k, dim=-1, sorted=False)
    return indices.to(torch.int32).masked_fill(~values.isfinite(), -1)


@contextmanager
def indexer_override(indexer, fixed_query=None, unsorted=False):
    original_query, original_topk = indexer._get_q_k_w_bf16, indexer._topk_sm80
    if fixed_query is not None:
        indexer._get_q_k_w_bf16 = lambda *args, **kwargs: fixed_query
    if unsorted:
        indexer._topk_sm80 = lambda scores, lengths: unsorted_topk(scores, lengths, indexer.topk)
    try:
        yield
    finally:
        indexer._get_q_k_w_bf16, indexer._topk_sm80 = original_query, original_topk


def freeze_residency(runner):
    def resolve(hostk, hostv, indices, outk, outv, lengths, step, prefetch=None):
        resident_gather(runner.cache, hostk, hostv, indices, outk, outv, lengths,
                        runner.data.maxlen, prefetch)
    runner.cache.resolve = resolve


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 24, 1, 36, 47])
    parser.add_argument("--groups", nargs="+")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    out = root / "long-replay-v01"
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "timelines").mkdir(exist_ok=True)
    merge(root)
    config = initialize()
    assert str(torch.cuda.get_device_properties(0).uuid) == GPU.removeprefix("GPU-")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                   "--format=csv,noheader,nounits"], text=True, timeout=15)
    import os
    assert not any(GPU in row and int(row.split(",")[0]) != os.getpid() for row in apps.splitlines()), apps
    torch.manual_seed(20261014)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
        torch.ones(16, device="cuda").add_(1); torch.cuda.synchronize()
    manifest = json.loads((root / "long-input-manifest.json").read_text())
    groups = [g for g in manifest["groups"] if not args.groups or g["label"] in args.groups]
    responses = {r["rid"]: r for f in root.glob("input-wave*.json") for r in json.loads(f.read_text())}
    params = [(g, wave, 1) for g in groups for wave in g["waves"]]
    params += [(g, g["first_wave"], 4) for g in groups]
    random.Random(20261014).shuffle(params)
    # Every selected cohort is retained; short natural outputs use a separate view.
    modes = [("native", "h-native", 4096, False, False),
             ("native-unsorted", "h-native", 4096, False, False),
             ("native-selection-ready", "h-native", 4096, True, False),
             ("native-indexer-ready", "h-native", 4096, True, True),
             ("h-staged", "h-staged", 4096, False, False),
             ("dram", "d-staged", 4096, False, False)]
    for capacity in (2048, 4096, 8192):
        modes += [(f"lru-b{capacity}", "l-cached", capacity, False, False),
                  (f"data-ready-b{capacity}", "no-io", capacity, False, False),
                  (f"indexer-data-ready-b{capacity}", "no-io", capacity, True, True)]
    modes += [("frozen-b4096", "l-cached", 4096, False, False),
              ("frozen-data-ready-b4096", "no-io", 4096, False, False)]
    env = dict(gpu_uuid=GPU, torch=torch.__version__, cuda=torch.version.cuda, tp=1,
               source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               seed=20261014, k=2048, modes=modes, input_source="A6000+L40S real C1 model inputs",
               replay="C1 and actual C4 layer compute from four independent C1 histories; not full-model serving",
               timing="3 eager +25 graph warmups; 5 graph timing repetitions; independent event/Kernel profile",
               ordering="Fixed-seed shuffled cohort order; native first for truth, remaining modes shuffled per point",
               comparison="Known-query/selection/KV oracles omit work; frozen-residency omits online replacement; not deployable baselines",
               model_config_sha256=hashlib.sha256(Path('/Tan/model/Keye-VL-2.0-30B-A3B/config.json').read_bytes()).hexdigest())
    save(out / "environment.json", env)
    total, completed = len(params)*len(args.layers)*len(modes), 0
    update(root, phase="long_a6000_layer_replay", child_pid=os.getpid(), total_cases=total, completed_cases=0,
           detail="真实长输入采集已结束；仅健康A6000计时，保留全部长度/缓存/层点。")
    layers = {}
    for group, wave, batch in params:
        for layer_id in args.layers:
            if layer_id not in layers:
                layers[layer_id] = load_layer(config, layer_id)
            frames = read_frames(root, wave, layer_id, batch=batch)
            valid = [f for f in frames if all(step < len(responses[rid]["response"]["output_ids"])-1
                                            for rid, step in zip(f["request_ids"], f["steps"]))]
            assert valid, (group["label"], wave, batch, layer_id)
            frame = next((f for f in valid if min(f["steps"]) >= 32), valid[-1])
            snapshot = "after32" if min(frame["steps"]) >= 32 else "last-valid-under32"
            data = ReplayData(root, frames, frame)
            layer, indexer = layers[layer_id], layers[layer_id].self_attn.sa_indexer
            reference = truth = fixed_query = None
            routing = None
            tail = list(modes[1:])
            random.Random(20261014 + wave*100 + batch*10 + layer_id).shuffle(tail)
            for label, mode, capacity, selected_ready, query_ready in [modes[0], *tail]:
                runner = ReplayLayer(layer, data, mode, capacity)
                if mode == "no-io":
                    runner.prepare_no_io(truth)
                if selected_ready:
                    runner.fixed_topk = truth
                if label.startswith("frozen"):
                    freeze_residency(runner)
                name = f"{group['label']}-w{wave:03d}-c{batch}-l{layer_id:02d}-{snapshot}-{label}"
                with indexer_override(indexer, fixed_query if query_ready else None, label == "native-unsorted"):
                    result, output = graph_measure(runner.forward, runner.reset, [runner],
                                                   profile_path=out / "timelines" / f"{name}.json")
                if label == "native":
                    reference, truth = output, runner.last_topk.clone()
                    fixed_query = tuple(t.clone() for t in indexer._get_q_k_w_bf16(
                        runner.last_attention_input, data.positions, False))
                    captured = []
                    hook = layer.mlp.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach().clone()))
                    runner.reset(); runner.forward(); torch.cuda.synchronize(); hook.remove()
                    logits, _ = layer.mlp.gate(captured[-1])
                    routes = layer.mlp.topk(captured[-1], logits)
                    ids = routes.topk_ids.cpu()
                    byte_count = sum(v.numel()*v.element_size()//v.shape[0] for _,v in layer.mlp.experts.named_parameters())
                    routing = dict(unique_experts=int(ids.unique().numel()), expert_ids=ids.tolist(),
                                   bytes_per_expert=byte_count, unique_expert_weight_bytes=int(ids.unique().numel())*byte_count)
                set_equal = torch.equal(runner.last_topk.sort(dim=-1).values, truth.sort(dim=-1).values)
                exact_equal = check_output(output, reference)
                errors = [float((a.float()-b.float()).abs().max()) for a,b in zip(output, reference)]
                if label != "native-unsorted":
                    assert exact_equal and torch.equal(runner.last_topk, truth), name
                    numerical_close = True
                else:
                    # This exact-set backend may reorder BF16 attention reductions.
                    numerical_close = all(torch.allclose(a, b, rtol=.02, atol=.005) for a,b in zip(output, reference))
                result.update(case=name, group=group["label"], window="single_layer", wave=wave,
                              layer=layer_id, concurrency=batch, snapshot=snapshot, mode=label,
                              base_storage_mode=mode, capacity=capacity, topk=2048,
                              request_ids=frame["request_ids"], decode_steps=frame["steps"],
                              lengths=frame["seq_lens"].tolist(), output_matches=exact_equal,
                              selected_set_matches=set_equal, output_max_abs_errors=errors,
                              numerical_close=numerical_close, usable_comparison=set_equal and numerical_close,
                              routing=routing, **io_counts(runner), **resource_counts(runner))
                save(out / "results" / f"{name}.json", result)
                completed += 1
                update(root, completed_cases=completed, detail=name)
                print(name, round(min(result["elapsed_us"]), 2), flush=True)
                del runner
                gc.collect()
            del data, frames, valid
            gc.collect()
        notify(f"HorizonKV长上下文层测量：{group['label']}、C{batch}、wave{wave}完成5个层的全部预定配置；结果包含短/长与改善/无改善点。")
    save(out / "complete.json", dict(cases=completed, expected_cases=total, completed_unix_s=time.time()))
    update(root, phase="long_replay_completed", completed_cases=completed, detail="预定长上下文层矩阵完成，待审计汇总。")


if __name__ == "__main__":
    main()
