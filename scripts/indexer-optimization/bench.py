"""Benchmark the Keye Decode Indexer from prepared BF16 Q, K cache, and gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("KEYE_SM80_DSA", "1")
os.environ.setdefault("KEYE_SM80_EXACT_TOPK", "1")

import torch

ROOT = Path(__file__).resolve().parents[2]
K = 2048
H = 16
D = 64
TRACE = ROOT / "data/horizonkv-motivation/2026-10-21_bs16-v01"
SEED = 20261014

sys.path.insert(0, str(ROOT / "scripts/keye_trace"))


@dataclass
class Inputs:
    q: torch.Tensor
    key_cache: torch.Tensor
    page_table: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    scale: float
    metadata: dict
    captured: torch.Tensor | None = None
    project: object = None

    def args(self):
        return self.q, self.key_cache, self.page_table, self.weights, self.lengths, self.scale


def page_layout(lengths, width):
    """Consecutive physical slots for each sequence, with unmapped padding."""
    index = torch.arange(width, dtype=torch.int32)[None, :]
    offsets = torch.tensor([0, *lengths[:-1]], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32)
    return (index + offsets[:, None]).masked_fill(index >= lens[:, None], -1).cuda(), lens.cuda()


def shuffle_pages(inputs, block, seed):
    """Repack into physically shuffled blocks while preserving the token histories."""
    slots = inputs.key_cache.shape[0]
    padded = math.ceil(slots / block) * block
    generator = torch.Generator(device="cuda").manual_seed(seed)
    block_count = padded // block
    shuffled = torch.randperm(block_count, generator=generator, device="cuda")
    permutation = (shuffled[:, None] * block + torch.arange(block, device="cuda")).view(-1)
    cache = torch.empty((padded, D), device="cuda", dtype=torch.bfloat16)
    cache.index_copy_(0, permutation[:slots], inputs.key_cache)
    table = inputs.page_table
    table = permutation[table.clamp_min(0).long()].int().masked_fill(table < 0, -1)
    return replace(inputs, key_cache=cache, page_table=table, metadata={
        **inputs.metadata, "physical_layout": f"shuffle_{block}_token_blocks",
    })


def page_layout_stats(inputs):
    table = inputs.page_table
    first, later = table[:, :-1], table[:, 1:]
    edges = (first >= 0) & (later >= 0)
    adjacent = (later == first + 1) & edges
    cols = torch.arange(1, table.shape[1], device=table.device)
    cut_at_tile_edges = (cols % 128) == 0
    violations = edges & ~adjacent & ~cut_at_tile_edges
    pad_left = torch.zeros((table.shape[0], 1), dtype=torch.bool, device=table.device)
    bad = torch.cat([pad_left, violations], dim=1)
    bad = torch.nn.functional.pad(bad, (0, -bad.shape[1] % 128))
    live_tiles = table[:, ::128] >= 0
    runs = live_tiles & ~bad.view(table.shape[0], -1, 128).any(dim=-1)
    return {
        "valid_neighbor_pairs": int(edges.sum()),
        "consecutive_neighbor_pairs": int(adjacent.sum()),
        "live_tiles_128": int(live_tiles.sum()),
        "consecutive_tiles_128": int(runs.sum()),
    }


def synthetic_inputs(batch, width, variant, seed):
    lengths = [width] * batch
    if variant == "ragged":
        fractions = [1.0, 0.82, 0.49, 0.17, 0.95, 0.67, 0.33, 0.08, 0.61, 0.39, 0.21, 0.12]
        lengths = [max(1, int(width * fractions[i % len(fractions)])) for i in range(batch)]
        if batch >= 4:
            lengths[-3:] = [min(width, K + 1), min(width, K), min(width, K - 1)]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).bfloat16()

    q, cache, weights = normal((batch, H, D)), normal((sum(lengths), D)), normal((batch, H))
    if variant == "all_ties":
        weights.zero_()
    elif variant == "boundary_ties":
        q.zero_()
        q[:, 0, 0] = 1
        weights.zero_()
        weights[:, 0] = 1
        for offset, length in zip(itertools.accumulate(lengths, initial=0), lengths):
            cache[offset:offset + length, 0] = 0
            cache[offset:offset + min(length, K // 2), 0] = 2
            cache[offset + K // 2:offset + min(length, K * 2), 0] = 1
    table, lens = page_layout(lengths, width)
    return Inputs(q, cache, table, weights, lens, D ** -0.5, {
        "source": "synthetic", "variant": variant, "batch": batch, "max_len": width,
        "sequence_lengths": lengths, "seed": seed, "physical_layout": "contiguous",
    })


class Trace:
    def __init__(self, root):
        self.root = root
        manifest = json.loads((root / "long-input-manifest.json").read_text())
        self.groups = {int(group["label"][5:-1]) * 1024: group for group in manifest["groups"]}
        self.frames = defaultdict(dict)
        for line in (root / "fixtures/manifest_rank00.jsonl").read_text().splitlines():
            record = json.loads(line)
            if len(record["request_ids"]) != 1:
                continue
            rid, = record["request_ids"]
            step, = record["steps"]
            self.frames[(rid, record["layer"])][step] = record["file"]

    def read(self, filename):
        return torch.load(self.root / "fixtures" / filename, mmap=True, weights_only=True)

    def load(self, ids, layer, step, indexer, projection_mode):
        # Retain only the tensors consumed by the Indexer. mmap avoids touching Main KV.
        keys, hidden, positions, captured, key_deltas = [], [], [], [], []
        for rid in ids:
            history = self.read(f"{rid}_layer{layer:02d}_rank00_history.pt")
            names = self.frames[(rid, layer)]
            frame = self.read(names[step])
            length = int(frame["seq_lens"][0])
            init = history["length"]
            cache = torch.empty((length, D), dtype=torch.bfloat16)
            cache[:init].copy_(history["index_k"][:length])
            write_start = init
            for delta_step in sorted(s for s in names if s <= step):
                update = frame if delta_step == step else self.read(names[delta_step])
                offset = int(update["seq_lens"][0]) - 1
                if offset >= init:
                    if offset != write_start:
                        raise ValueError(f"nonconsecutive index K deltas for {rid}: {offset=} {write_start=}")
                    write_start += 1
                cache[offset].copy_(update["deltas"][0]["index_k"])
            if write_start != length:
                raise ValueError(f"incomplete Indexer history for {rid}: {write_start=} {length=}")
            keys.append(cache)
            hidden.append(frame["attention_input"])
            positions.append(frame["positions"])
            captured.append(frame["topk"])
            key_deltas.append(cache[-1])

        hidden = torch.cat(hidden).cuda()
        positions = torch.cat(positions, dim=-1).cuda()
        def project():
            return indexer._get_q_k_w_bf16(hidden, positions, False)

        if projection_mode == "captured-singletons" and len(ids) > 1:
            projected = [
                indexer._get_q_k_w_bf16(hidden[i:i + 1], positions[..., i:i + 1].contiguous(), False)
                for i in range(len(ids))
            ]
            q, key, weights = (torch.cat(parts) for parts in zip(*projected))
        else:
            q, key, weights = project()
        lengths = [entry.shape[0] for entry in keys]
        table, lens = page_layout(lengths, max(lengths))
        return Inputs(q.contiguous(), torch.cat(keys).cuda(), table, weights.contiguous(),
                      lens, float(indexer.softmax_scale), {
                          "source": "real", "trace_root": str(self.root.relative_to(ROOT))
                          if self.root.is_relative_to(ROOT) else str(self.root),
                          "batch": len(ids), "max_len": max(lengths), "sequence_lengths": lengths,
                          "layer": layer, "step": step, "request_ids": ids,
                          "physical_layout": "contiguous",
                          "q_gate_projection_mode": projection_mode,
                          "projected_key_max_abs_diff": float(
                              (key.float() - torch.stack(key_deltas).cuda().float()).abs().max()),
                      }, torch.cat(captured).cuda(), project)

    def ids(self, width, batch, offset):
        group = self.groups[width]
        requests = list(zip(group["waves"], group["case_ids"]))
        requests = requests[offset:] + requests[:offset]
        return [f"replay__wave{wave:03d}__{case}" for wave, case in requests[:batch]]

    def append_frames(self, ids, layer, step):
        frames = [self.read(self.frames[(rid, layer)][step]) for rid in ids]
        return (torch.cat([frame["attention_input"] for frame in frames]).cuda(),
                torch.cat([frame["positions"] for frame in frames], dim=-1).cuda(),
                torch.cat([frame["seq_lens"] for frame in frames]).int().cuda(),
                torch.stack([frame["deltas"][0]["index_k"] for frame in frames]).cuda())


def torch_reference(inputs, host_lengths=None, block_size=None):
    """Native BF16-rounded dot; independent PyTorch gather, matmul, and reduction."""
    q, key, table, weights, lengths, scale = inputs.args()
    scores = torch.full(table.shape, -float("inf"), dtype=torch.float32, device=q.device)
    if host_lengths is None:
        host_lengths = lengths.tolist()
    for row, length in enumerate(host_lengths):
        width = max(length, 1) if block_size is None else block_size
        for start in range(0, length, width):
            stop = min(start + width, length)
            slots = table[row, start:stop]
            keys_t = key.index_select(0, slots.clamp_min(0).long()).T.contiguous()
            head_scores = (q[row] @ keys_t).float()
            score = ((head_scores * scale).relu() * weights[row, :, None].float()).sum(0)
            scores[row, start:stop] = score.masked_fill_(slots < 0, -float("inf"))
    return scores


def torch_select(scores):
    values, indices = torch.topk(scores, min(K, scores.shape[1]), sorted=False)
    logical = indices.int().masked_fill(~values.isfinite(), -1)
    return torch.nn.functional.pad(logical, (0, K - logical.shape[1]), value=-1)


def load_modules(paths):
    modules = []
    for path in paths:
        name = Path(path).stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def load_indexer(layer):
    from horizonkv_layer_replay import MODEL
    from safetensors import safe_open
    from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    config = json.loads((MODEL / "config.json").read_text())
    sparse = config["sa_config"]
    with torch.device("cuda"):
        indexer = KeyeIndexer(
            hidden_size=config["hidden_size"], num_heads=sparse["indexer_num_heads"],
            head_dim=sparse["indexer_head_dim"], topk=sparse["topk"],
            mrope_section=config["rope_scaling"]["mrope_section"], rope_theta=config["rope_theta"],
            main_head_dim=config["head_dim"], max_position_embeddings=config["max_position_embeddings"],
            layer_id=layer,
        )
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for name, parameter in indexer.named_parameters():
        key = f"model.layers.{layer}.self_attn.sa_indexer.{name}"
        with safe_open(str(MODEL / weight_map[key]), framework="pt", device="cpu") as checkpoint:
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, checkpoint.get_tensor(key))
    return indexer.eval()


def assess(indices, scores, references, tolerance=0.0):
    """Check score-optimal membership independently from the implementation tie order."""
    n_rows, width = scores.shape
    if indices.shape != (n_rows, K) or indices.dtype != torch.int32:
        return {"valid": False, "bad_output_contract": [list(indices.shape), str(indices.dtype)]}
    present = indices >= 0
    selected = scores.gather(1, indices.long().clamp(0, width - 1))
    legal = (indices == -1) | ((indices < width) & present & selected.isfinite())
    counts = present.sum(1)
    required = scores.isfinite().sum(1).clamp_max(K)
    ordered = indices.sort(dim=1).values
    duplicates = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).sum(1)
    reference_scores = scores.gather(1, references.long().clamp_min(0))
    threshold = reference_scores.masked_fill(references < 0, float("inf")).amin(1)
    under = (present & (selected < threshold[:, None] - tolerance)).sum(1)
    about = (present & (selected < threshold[:, None])).sum(1)
    strict_cutoff = threshold[:, None] + tolerance
    strict_misses = (
        (scores > strict_cutoff).sum(1) - (present & (selected > strict_cutoff)).sum(1)
    ).clamp_min(0)
    strict_misses_exact = (
        (scores > threshold[:, None]).sum(1) -
        (present & (selected > threshold[:, None])).sum(1)
    ).clamp_min(0)
    in_ref = references.sort(dim=-1).values
    loc = torch.searchsorted(in_ref, indices.contiguous()).clamp_max(K - 1)
    same = ((in_ref.gather(1, loc) == indices) & present).sum(1)
    regret = torch.where(present, (threshold[:, None] - selected).clamp_min(0), 0).amax(1)
    valid = (legal.all(1) & (counts == required) & (duplicates == 0) &
             (under == 0) & (strict_misses == 0))
    max_regret = float(regret.max())
    return {
        "valid": bool(valid.all()), "rows_pass": int(valid.sum()), "rows_total": n_rows,
        "invalid_indices": int((~legal).sum()), "duplicates": int(duplicates.sum()),
        "missing_or_extra": int((counts - required).abs().sum()),
        "below_threshold": int(about.sum()), "below_threshold_with_tolerance": int(under.sum()),
        "missing_strictly_above_threshold": int(strict_misses_exact.sum()),
        "missing_strictly_above_threshold_with_tolerance": int(strict_misses.sum()),
        "total_selected": int(required.sum()), "set_overlap": int(same.sum()),
        "max_threshold_regret": max_regret if math.isfinite(max_regret) else None,
        "tolerance_abs": tolerance,
    }


def measure_graph(calls, trials, iterations, seed, warmups):
    graphs, outputs = {}, {}
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    for name, prepared in calls.items():
        rotating = itertools.cycle(prepared if isinstance(prepared, list) else [prepared])
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                next(rotating)()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            live = [next(rotating)() for _ in range(iterations)]
        graphs[name], outputs[name] = graph, live
    current.wait_stream(stream)

    rng = random.Random(seed)
    samples = {name: [] for name in graphs}
    for _ in range(trials):
        names = list(graphs)
        rng.shuffle(names)
        events = {}
        for name in names:
            start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            finish.record()
            events[name] = start, finish
        torch.cuda.synchronize()
        for name, (start, finish) in events.items():
            samples[name].append(start.elapsed_time(finish) * 1000 / iterations)
    return {
        name: {"samples_us": values, "median_us": statistics.median(values),
               "mad_us": statistics.median(abs(v - statistics.median(values)) for v in values),
               "min_us": min(values), "max_us": max(values)}
        for name, values in samples.items()
    }


def profile_device_launches(name, call, path):
    from torch.profiler import ProfilerActivity, profile

    def collect(executor):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as observed:
            executor()
            torch.cuda.synchronize()
        names = [event.name for event in observed.events() if event.device_type.name == "CUDA"]
        events = Counter(names)
        kernels = {label: count for label, count in events.items()
                   if not re.match(r"^\[?(cuda )?mem(cpy|set)", label, re.IGNORECASE)}
        return observed, events, kernels

    errors = {}
    executors = [("eager", call)]
    try:
        # Prime Kineto before creating the graph. Its first trace can miss graph nodes.
        collect(lambda: torch.ones(1, device="cuda"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        executors.insert(0, ("cuda_graph", graph.replay))
    except RuntimeError as exc:
        errors["graph_capture"] = str(exc)

    for mode, executor in executors:
        try:
            observed, events, kernels = collect(executor)
            if not kernels:
                errors[mode] = "No CUDA kernels reported by Kineto"
                continue
            trace = path.with_suffix(f".{name}.{mode}.kineto.json")
            observed.export_chrome_trace(str(trace))
            return {"mode": mode, "kernel_count": sum(kernels.values()), "kernels": kernels,
                    "cuda_events": dict(events), "trace": str(trace), "errors": errors}
        except RuntimeError as exc:
            errors[mode] = str(exc)
    return {"errors": errors}


def make_selectors(lengths, width, args, modules):
    from sgl_kernel import fast_topk_v2

    selectors = {
        "fast_topk_v2": lambda s: fast_topk_v2(s, lengths, K),
        "torch_topk": torch_select,
    }
    if not args.skip_flashinfer:
        from flashinfer import top_k_ragged_transform
        offsets = torch.zeros_like(lengths)
        selectors["flashinfer_topk"] = lambda s: top_k_ragged_transform(s, offsets, lengths, K)
        if args.flashinfer_deterministic:
            selectors["flashinfer_deterministic_topk"] = lambda s: top_k_ragged_transform(
                s, offsets, lengths, K, deterministic=True)
    for module in modules:
        if module.__name__ == "vllm_baseline":
            selectors["vllm_persistent"] = module.prepare_selector(lengths, width)
    if args.ampere_cute:
        sys.path.insert(0, str(ROOT / "data/indexer-optimization/vllm-selector"))
        from ampere_topk.topk import decode_topk

        output = torch.empty((lengths.numel(), K), dtype=torch.int32, device=lengths.device)

        def select_ampere(scores):
            decode_topk(scores, lengths, output)
            return output

        selectors["vllm_ampere_cute"] = select_ampere
    return selectors


def prepare_calls(inputs, args, modules):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    q, key, table, weights, lengths, scale = inputs.args()
    score_call = lambda: keye_indexer_score_paged(q, key, table, weights, lengths, scale)
    scores = score_call()
    selectors = make_selectors(lengths, table.shape[1], args, modules)
    paths = {"sgl_score": score_call}
    for name, selector in selectors.items():
        paths[f"{name}_only"] = lambda selector=selector: selector(scores)
        paths[f"sgl_score_{name}"] = lambda selector=selector: selector(score_call())
    if args.torch_score_baseline:
        host_lengths = lengths.tolist()
        paths["torch_score"] = lambda: torch_reference(inputs, host_lengths, block_size=8192)
        paths["torch_score_torch_topk"] = lambda: torch_select(paths["torch_score"]())

    stages = {}
    for module in modules:
        if module.__name__ == "vllm_baseline":
            continue
        for name, call in module.prepare(*inputs.args()).items():
            if re.search(args.select, name):
                if name in paths:
                    raise ValueError(f"duplicate implementation name {name}")
                paths[name] = call
        if args.stages and hasattr(module, "prepare_stages"):
            stages.update(module.prepare_stages(*inputs.args()))

    return scores, paths, stages, selectors


def prepare_full(inputs, selectors, fixed_calls, modules, args):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    q, key, table, weights, lengths, scale = inputs.args()
    current = table.gather(1, lengths.long()[:, None] - 1).view(-1).long()

    def project_and_write():
        fresh_q, fresh_key, fresh_w = inputs.project()
        key.index_copy_(0, current, fresh_key)
        return fresh_q, fresh_w

    def full_native(selector):
        fresh_q, fresh_w = project_and_write()
        return selector(keye_indexer_score_paged(fresh_q, key, table, fresh_w, lengths, scale))

    def full_fixed(call):
        fresh_q, fresh_w = project_and_write()
        q.copy_(fresh_q)
        weights.copy_(fresh_w)
        return call()

    paths = {f"full_sgl_score_{name}": lambda selector=selector: full_native(selector)
             for name, selector in selectors.items()}
    paths.update({f"full_{name}": lambda call=call: full_fixed(call)
                  for name, call in fixed_calls.items() if name.startswith("opt_")})
    stages = {"project_and_write": project_and_write}
    packed_arenas = {}
    if args.direct_packed:
        pack_module, = [module for module in modules if hasattr(module, "prepare_packed_full")]
        for name, selector in selectors.items():
            if name not in {"flashinfer_topk", "vllm_persistent", "vllm_ampere_cute"}:
                continue
            packed = pack_module.prepare_packed_full(*inputs.args(), inputs.project, selector)
            packed_name = f"full_direct_pack12_n128w4_{name}"
            paths[packed_name] = packed
            packed_arenas[packed_name] = packed
            if "direct_pack12_write" not in stages:
                _, projected_key, _ = inputs.project()
                stages["direct_pack12_write"] = lambda packed=packed: packed.append_projected(
                    projected_key)
            if hasattr(pack_module, "prepare_dense_full"):
                block = 64 if inputs.metadata.get("length_bucket") == 32768 else 128
                paths[f"full_direct_dense_n{block}w4_{name}"] = pack_module.prepare_dense_full(
                    *inputs.args(), inputs.project, selector, block=block)
    return paths, stages, packed_arenas


def run_append_check(inputs, trace, indexer, layer, args, modules):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    module, = [entry for entry in modules if hasattr(entry, "prepare_packed_full")]
    ids = inputs.metadata["request_ids"]
    steps = args.append_steps
    frames = [trace.append_frames(ids, layer, step) for step in steps]
    initial_lengths = inputs.lengths.tolist()
    final_lengths = frames[-1][2].tolist()
    width = math.ceil(max(final_lengths) / 4) * 4
    table, _ = page_layout(final_lengths, width)
    lengths = inputs.lengths.clone()
    position = torch.arange(width, device="cuda")
    page_table = table.masked_fill(position >= lengths[:, None], -1)
    cache = torch.zeros((sum(final_lengths), D), dtype=torch.bfloat16, device="cuda")
    for row, length in enumerate(initial_lengths):
        slots = table[row, :length].long()
        initial_slots = inputs.page_table[row, :length].long()
        cache.index_copy_(0, slots, inputs.key_cache.index_select(0, initial_slots))
    state = replace(inputs, key_cache=cache, page_table=page_table, lengths=lengths)
    if args.shuffle_block != [0]:
        raise ValueError("append correctness uses separate fixed contiguous-slot capacities")

    current_input = frames[0]

    def project():
        hidden, positions, _, _ = current_input
        return indexer._get_q_k_w_bf16(hidden, positions, False)

    selectors = make_selectors(lengths, width, args, modules)
    selector_name = "vllm_persistent" if len(ids) <= 4 else "flashinfer_topk"
    packed = module.prepare_packed_full(*state.args(), project, selectors[selector_name],
                                        reserve=len(ids) * len(steps))
    step_results = []
    for step, current_input in zip(steps, frames):
        _, _, live, trace_keys = current_input
        if not torch.equal(live, lengths + 1):
            raise ValueError(f"append step {step} does not advance each request by one position")
        tail = (live.long() - 1)[:, None]
        physical = table.gather(1, tail)
        page_table.scatter_(1, tail, physical)
        lengths.copy_(live)
        indices = packed().clone()
        q, current_k, gates = project()
        cache.index_copy_(0, physical.view(-1).long(), current_k)
        raw = keye_indexer_score_paged(q, cache, page_table, gates, lengths, inputs.scale)
        reference = torch_reference(replace(state, q=q, weights=gates))
        sgl_result = assess(indices, raw, torch_select(raw))
        torch_result = assess(indices, reference, torch_select(reference))
        step_results.append({
            "step": step, "lengths": lengths.tolist(), "sgl": sgl_result, "torch": torch_result,
            "append_overflow_rows": int(packed.last_overflow.sum()),
            "projected_key_vs_trace_max_delta": float((current_k.float() - trace_keys.float()).abs().max()),
            "exception_rows_used": int(packed.arena.next_exception.item()),
        })
    decoded = module.decode_keys(packed.arena.keys)
    return {
        "input": inputs.metadata, "selector": selector_name, "append": step_results,
        "final_cache_bitwise_equal": torch.equal(decoded.view(torch.int16), cache.view(torch.int16)),
        "raw_index_k_bytes": cache.nbytes, "packed_index_k_bytes": packed.arena.keys.storage_bytes,
        "exception_insertions_reserved": len(ids) * len(steps),
    }


def case_benchmark(inputs, args, modules, rotation=()):
    if args.full:
        # Define full-scope correctness using the Indexer-projected current key.
        new_q, fresh_key, new_w = inputs.project()
        current = inputs.page_table.gather(1, inputs.lengths.long()[:, None] - 1).view(-1).long()
        inputs.key_cache.index_copy_(0, current, fresh_key)
        inputs.q.copy_(new_q)
        inputs.weights.copy_(new_w)
        inputs = replace(inputs, captured=None, metadata={
            **inputs.metadata, "current_index_k": "projected_at_benchmark_batch_size",
            "full_opt_q_gate_workspace_copies": True,
        })

    scores, paths, stages, selectors = prepare_calls(inputs, args, modules)
    packed_arenas = {}
    if args.full:
        full_paths, full_stages, packed_arenas = prepare_full(inputs, selectors, paths, modules, args)
        paths.update(full_paths)
        stages.update(full_stages)

    torch_scores = torch_reference(inputs)
    torch_ref = torch_select(torch_scores)
    sgl_ref = torch_select(scores)
    valid = torch_scores.isfinite() & scores.isfinite()
    delta = (torch_scores[valid] - scores[valid]).abs()
    score_check = {
        "abs_max": float(delta.max()) if delta.numel() else 0.0,
        "abs_mean": float(delta.mean()) if delta.numel() else 0.0,
        "p99_abs_delta": float(torch.quantile(delta, .99)) if delta.numel() else 0.0,
        "bit_equal": int((torch_scores == scores).sum()), "total": scores.numel(),
        "finite_mask_agrees": bool(torch.equal(torch_scores.isfinite(), scores.isfinite())),
        "sgl_topk_vs_torch": assess(sgl_ref, torch_scores, torch_ref),
    }
    score_check["sgl_topk_vs_torch_with_score_bound"] = assess(
        sgl_ref, torch_scores, torch_ref, tolerance=2 * score_check["abs_max"])
    if inputs.captured is not None:
        score_check["trace_topk_vs_torch"] = assess(inputs.captured, torch_scores, torch_ref)
        score_check["trace_topk_vs_sgl"] = assess(inputs.captured, scores, sgl_ref)

    storage = {"raw_index_k_bytes": inputs.key_cache.nbytes,
               "page_table_bytes": inputs.page_table.nbytes,
               "native_score_bytes": scores.nbytes,
               "page_layout_stats": page_layout_stats(inputs)}
    if args.packed_audit:
        for module in modules:
            if not hasattr(module, "encode_keys"):
                continue
            encoded = module.encode_keys(inputs.key_cache, tail_reserve=len(inputs.lengths))
            decoded = module.decode_keys(encoded)
            correct = torch.equal(decoded.view(torch.int16), inputs.key_cache.view(torch.int16))
            bytes_per_slot = inputs.key_cache.shape[1] * inputs.key_cache.element_size()
            exceptions = encoded.exceptions.shape[0] - len(inputs.lengths)
            timings = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                new = module.encode_keys(inputs.key_cache, tail_reserve=len(inputs.lengths))
                torch.cuda.synchronize()
                timings.append((time.perf_counter_ns() - start) / 1_000)
                del new
            storage.update({
                "codec_roundtrip_all_physical_slots_bit_exact": correct,
                "packed_index_k_bytes": encoded.storage_bytes,
                "packed_exception_slots": exceptions,
                "packed_reserved_tail_slots": len(inputs.lengths),
                "packed_exception_bytes": exceptions * bytes_per_slot,
                "packed_initial_build_wall_us": timings,
                "packed_initial_build_wall_median_us": statistics.median(timings),
            })

    evaluations = {}
    for name, call in paths.items():
        if name in {"sgl_score", "torch_score"}:
            continue
        repeats = [call().clone() for _ in range(args.correctness_repeats)]
        sgl_checks = [assess(x, scores, sgl_ref) for x in repeats]
        torch_checks = [assess(x, torch_scores, torch_ref) for x in repeats]
        torch_bounded = [
            assess(x, torch_scores, torch_ref, tolerance=2 * score_check["abs_max"])
            for x in repeats
        ]
        sorted_first = repeats[0].sort(dim=-1).values
        evaluations[name] = {
            "sgl": sgl_checks,
            "torch": torch_checks,
            "torch_with_native_score_bound": torch_bounded,
            "repeat_set_equal": all(torch.equal(x.sort(dim=-1).values, sorted_first)
                                    for x in repeats[1:]),
            "repeat_order_equal": all(torch.equal(x, repeats[0]) for x in repeats[1:]),
        }
        if name in packed_arenas:
            packed = packed_arenas[name]
            evaluations[name]["append_overflow_rows"] = int(packed.last_overflow.sum())
            evaluations[name]["packed_index_k_bytes"] = packed.arena.keys.storage_bytes
        if hasattr(call, "candidate_counts") and hasattr(call, "fallback_lengths"):
            evaluations[name]["candidate_counts"] = call.candidate_counts.tolist()
            evaluations[name]["fallback_lengths"] = call.fallback_lengths.tolist()
    if args.check_only:
        timings = {}
    else:
        timed_paths = {**paths, **stages}
        if rotation:
            independent = [prepare_calls(entry, args, modules) for entry in rotation]
            for name, main_path in paths.items():
                timed_paths[name] = [main_path, *(item[1][name] for item in independent)]
        timings = measure_graph(timed_paths, args.trials, args.graph_iterations, args.seed, args.warmups)
    launch_profile = {}
    if args.profile_kernels:
        baselines = [name for name in paths if name.startswith("full_sgl_score_") and
                     all(item["valid"] for item in evaluations[name]["sgl"])]
        expected_packed = ("full_direct_pack12_n128w4_vllm_persistent"
                           if len(inputs.lengths) <= 4 else "full_direct_pack12_n128w4_flashinfer_topk")
        baseline = min(baselines, key=lambda name: timings[name]["median_us"])
        launch_profile = {name: profile_device_launches(name, paths[name], args.output)
                          for name in (baseline, expected_packed)}
    return {"input": inputs.metadata, "score_check": score_check, "storage": storage,
            "evaluations": evaluations,
            "timings": timings, "launch_profile": launch_profile}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "real", "mixed"), default="synthetic")
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--length", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--layer", nargs="+", type=int, default=[12])
    parser.add_argument("--step", nargs="+", type=int, default=[0])
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--mixed-counts", nargs=3, type=int, default=[4, 4, 8])
    parser.add_argument("--variant", nargs="+", choices=("full", "ragged", "all_ties", "boundary_ties"),
                        default=["full"])
    parser.add_argument("--shuffle-block", type=int, nargs="+", default=[0],
                        help="0 leaves slots contiguous; positive values shuffle physical blocks")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--module", action="append", type=Path, default=[])
    parser.add_argument("--select", default=".")
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--graph-iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--correctness-repeats", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-flashinfer", action="store_true")
    parser.add_argument("--torch-score-baseline", action="store_true",
                        help="also time independent PyTorch gather, score, and Top-K")
    parser.add_argument("--flashinfer-deterministic", action="store_true")
    parser.add_argument("--full", action="store_true", help="also time projection and current K write")
    parser.add_argument("--direct-packed", action="store_true",
                        help="add projection and direct write to the persistent packed arena")
    parser.add_argument("--stages", action="store_true", help="also time module prepare_stages() entries")
    parser.add_argument("--rotate", type=int, default=1,
                        help="real single-request, same layer/step, disjoint index K buffers")
    parser.add_argument("--projection-mode", choices=("merged", "captured-singletons"),
                        default="merged")
    parser.add_argument("--ampere-cute", action="store_true",
                        help="include vLLM Ampere decode_topk via the isolated CuTe environment")
    parser.add_argument("--align-width", action="store_true",
                        help="round score width to four floats for cross-run comparisons")
    parser.add_argument("--packed-audit", action="store_true",
                        help="measure packed storage, initial build, and bitwise fixture round trip")
    parser.add_argument("--append-steps", type=int, nargs="+", default=[],
                        help="check sequential actual appends after the single initial --step")
    parser.add_argument("--profile-kernels", action="store_true",
                        help="record CUDA device kernels for one direct packed and baseline case")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def emit(case, args, modules, file, independent=()):
    if args.ampere_cute or args.align_width:
        def align(entry):
            width = entry.page_table.shape[1]
            aligned = math.ceil(width / 4) * 4
            return replace(entry,
                page_table=torch.nn.functional.pad(entry.page_table, (0, aligned - width), value=-1),
                metadata={**entry.metadata, "max_len": aligned, "unpadded_max_len": width})

        case = align(case)
        independent = [align(entry) for entry in independent]
    for block in args.shuffle_block:
        packed = shuffle_pages(case, block, args.seed) if block else case
        rotation = [shuffle_pages(entry, block, args.seed) if block else entry
                    for entry in independent]
        if rotation:
            packed.metadata["rotation_request_ids"] = [packed.metadata["request_ids"][0],
                *(entry.metadata["request_ids"][0] for entry in rotation)]
            packed.metadata["rotation_cache_bytes"] = sum(
                entry.key_cache.nbytes for entry in [packed, *rotation])
        row = case_benchmark(packed, args, modules, rotation)
        file.write(json.dumps(row, allow_nan=False) + "\n")
        file.flush()
        print(row["input"], {k: round(v["median_us"], 3) for k, v in row["timings"].items()},
              {k: all(x["valid"] for x in v["sgl"]) for k, v in row["evaluations"].items()},
              flush=True)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.full and (args.source == "synthetic" or args.rotate != 1):
        raise ValueError("--full requires real inputs without rotation")
    if args.full and args.projection_mode != "merged":
        raise ValueError("--full requires merged native projection")
    if args.direct_packed and not args.full:
        raise ValueError("--direct-packed requires --full")
    if args.profile_kernels and (not args.direct_packed or args.check_only or
            len(args.layer) * len(args.step) * len(args.batch) * len(args.length) != 1):
        raise ValueError("--profile-kernels requires one timed --direct-packed case")
    if args.append_steps and (args.source != "real" or len(args.step) != 1 or
            args.full or args.projection_mode != "merged" or args.rotate != 1):
        raise ValueError("--append-steps requires one initial step of merged real input")
    if args.rotate < 1 or (args.rotate > 1 and
            (args.source != "real" or args.batch != [1] or args.rotate > 16 or
             args.graph_iterations % args.rotate)):
        raise ValueError("--rotate requires --source real --batch 1 and divides the graph iteration count")
    modules = load_modules(args.module)
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    software = {p: version(p) for p in ("torch", "triton", "sglang-kernel", "flashinfer-python")}
    if not args.skip_flashinfer:
        import flashinfer
        software["flashinfer_imported"] = flashinfer.__version__
    provenance = {
        "args": vars(args), "gpu": str(torch.cuda.get_device_properties(0)),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), *args.module]},
        "software": software, "cuda_home": os.environ.get("CUDA_HOME"), "cuda": torch.version.cuda,
        "dimensions": {"heads": H, "head_dim": D, "topk": K},
        "protocol": "CUDA events around a graph containing repeated calls; one graph per implementation; "
                    "order shuffled between trials; GPU-resident input and workspaces; warmed with "
                    "same input; CPU preparation and any module prepare() time excluded",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    with args.output.open("w") as file:
        if args.source == "synthetic":
            for width in args.length:
                for batch in args.batch:
                    for variant in args.variant:
                        emit(synthetic_inputs(batch, width, variant, args.seed), args, modules, file)
            return
        from horizonkv_layer_replay import initialize
        initialize()
        trace = Trace(args.trace.resolve())
        for layer in args.layer:
            indexer = load_indexer(layer)
            for step in args.step:
                if args.source == "mixed":
                    indices = [rid for width, count in zip(sorted(trace.groups), args.mixed_counts)
                               for rid in trace.ids(width, count, args.request_offset)]
                    real = trace.load(indices, layer, step, indexer, args.projection_mode)
                    real.metadata["length_bucket"] = "mixed"
                    emit(real, args, modules, file)
                    continue
                for width in args.length:
                    for batch in args.batch:
                        if not 1 <= batch <= 16:
                            raise ValueError("one trace length bucket contains 16 distinct requests")
                        ids = trace.ids(width, max(batch, args.rotate), args.request_offset)
                        rotation = [trace.load([rid], layer, step, indexer, args.projection_mode)
                                    for rid in ids[batch:]]
                        ids = ids[:batch]
                        real = trace.load(ids, layer, step, indexer, args.projection_mode)
                        real.metadata["length_bucket"] = width
                        if args.append_steps:
                            check = run_append_check(real, trace, indexer, layer, args, modules)
                            print(json.dumps(check), file=file, flush=True)
                            print(real.metadata["batch"], width, layer,
                                  check["final_cache_bitwise_equal"],
                                  [result["sgl"]["valid"] for result in check["append"]], flush=True)
                        else:
                            emit(real, args, modules, file, rotation)


if __name__ == "__main__":
    main()
