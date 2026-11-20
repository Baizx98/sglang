"""Measure complete Keye Decode indexing on the frozen Full acceptance matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
ROUND1 = THIS.parent / "indexer-optimization"
sys.path[:0] = [str(THIS), str(ROUND1)]

import bench as previous
import optimize as lossless
import vllm_baseline
from projection import prepare_projection
from slab_index import IndexSlab, Plan
from int4_scan import Int4Scan


def steady_init(action):
    result = action()
    elapsed = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        rebuilt = action()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter_ns() - start) / 1_000)
        del rebuilt
    return result, elapsed


def compare_projection(reference, candidate):
    names = ("query", "current_key", "gates")
    return {
        name: {"different_bf16": int((left.view(torch.int16) != right.view(torch.int16)).sum()),
               "elements": left.numel(), "max_abs": float((left.float() - right.float()).abs().max())}
        for name, left, right in zip(names, reference, candidate)
    }


def benchmark_case(inputs, projections, args):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    q, raw_cache, table, weights, lengths, scale = inputs.args()
    tail_slots = table.gather(1, lengths.long()[:, None] - 1).view(-1).long()
    native = inputs.project()
    raw_cache.index_copy_(0, tail_slots, native[1])
    expected = keye_indexer_score_paged(native[0], raw_cache, table, native[2], lengths, scale)
    exact = previous.torch_reference(replace(inputs, q=native[0], weights=native[2]))
    native_ids = previous.torch_select(expected)
    exact_ids = previous.torch_select(exact)
    selectors = previous.make_selectors(lengths, table.shape[1], args, [vllm_baseline])

    def native_full(project, select, cache=raw_cache, pages=table, slots=tail_slots):
        def call():
            query, key, gates = project()
            cache.index_copy_(0, slots, key)
            values = keye_indexer_score_paged(query, cache, pages, gates, lengths, scale)
            return select(values)
        return call

    block = 64 if inputs.metadata["length_bucket"] == 32768 else 128
    fixed_selector_name = "vllm_persistent" if len(lengths) <= 4 else "vllm_ampere_cute"
    fixed_selector = selectors[fixed_selector_name]
    calls = {}
    stages = {"project_native": inputs.project,
              **{f"project_{name}": project for name, project in projections.items()}}
    for name, select in selectors.items():
        calls[f"original_sgl_{name}"] = native_full(inputs.project, select)
        if name in {"flashinfer_topk", "vllm_persistent", "vllm_ampere_cute"}:
            calls[f"original_tuned_dense_{name}"] = lossless.prepare_dense_full(
                *inputs.args(), inputs.project, select, block=block,
            )
            calls[f"old_lossless_{name}"] = lossless.prepare_packed_full(
                *inputs.args(), inputs.project, select,
            )
    for name, project in projections.items():
        calls[f"fixed_layout_{name}"] = native_full(project, fixed_selector)
        calls[f"fixed_layout_lossless_{name}"] = lossless.prepare_packed_full(
            *inputs.args(), project, fixed_selector,
        )

    plan = Plan.from_lengths(lengths, args.growth)
    slab_info = {"reserved_slots": plan.capacity, "live_slots": plan.live_slots,
                 "free_slot_bytes_raw_equivalent": (plan.capacity - plan.live_slots) * 128,
                 "offset_capacity_bytes": plan.bases.nbytes + plan.capacities.nbytes}
    slabs = {}
    for packed, name in ((False, "slab_bf16"), (True, "slab_lossless")):
        def init():
            return IndexSlab(plan, raw_cache, table, lengths, packed=packed,
                             exception_reserve=len(lengths) * args.growth)

        slab, init_wall = steady_init(init)
        slabs[packed] = slab
        calls[name] = slab.prepare(inputs.project, fixed_selector, scale, block)
        for projection, project in projections.items():
            calls[f"{name}_{projection}"] = slab.prepare(project, fixed_selector, scale, block)
        slab_info[name] = {
            "bytes": slab.storage_bytes,
            "initial_build_wall_us": init_wall,
            "initial_build_wall_median_us": statistics.median(init_wall),
        }
    row_positions = torch.arange(table.shape[1], device=table.device, dtype=torch.int32).view(1, -1)
    slab_pages = (plan.bases[:, None] + row_positions).masked_fill(row_positions >= lengths[:, None], -1)
    slab_tails = (plan.bases + lengths - 1).long()
    for name in ("flashinfer_topk", "vllm_persistent", "vllm_ampere_cute"):
        select = selectors[name]
        slab_raw = slabs[False].resident
        calls[f"same_placement_sgl_{name}"] = native_full(
            inputs.project, select, slab_raw, slab_pages, slab_tails,
        )
        calls[f"same_placement_old_lossless_{name}"] = lossless.prepare_packed_full(
            q, slab_raw, slab_pages, weights, lengths, scale, inputs.project, select,
        )
    if args.int4 or args.int4_ablate_2560:
        compact, wall = steady_init(lambda: Int4Scan(slabs[False], slabs[True]))
        slab_info["int4"] = {"bytes": compact.storage_bytes, "additional_encode_wall_us": wall}
        candidate_sizes = [4096, 8192] if args.int4 else []
        if args.int4_ablate_2560:
            candidate_sizes.append(2560)
        for candidate_k in candidate_sizes:
            call, coarse = compact.prepare(projections["fused_norm"], scale, candidate_k)
            calls[f"int4_scan_bf16_refine_{candidate_k}"] = call
            if candidate_k == 4096:
                stages["int4_quantize_and_scan"] = lambda coarse=coarse: coarse(*native)
    if args.int4_direct:
        direct, direct_wall = steady_init(lambda: Int4Scan(slabs[False]))
        calls["int4_direct_flashinfer"] = direct.prepare_direct(
            projections["fused_norm"], selectors["flashinfer_topk"], scale,
        )
        calls["int4_direct_vllm"] = direct.prepare_direct(
            projections["fused_norm"], fixed_selector, scale,
        )
        calls["int4_direct_flashinfer_native_projection"] = direct.prepare_direct(
            inputs.project, selectors["flashinfer_topk"], scale,
        )
        if args.direct_native_vllm:
            calls["int4_direct_vllm_native_projection"] = direct.prepare_direct(
                inputs.project, fixed_selector, scale,
            )
        slab_info["int4_direct"] = {
            "bytes": direct.storage_bytes,
            "initial_encode_from_bf16_slab_wall_us": direct_wall,
            "initial_build_wall_median_us": statistics.median(direct_wall)
            + slab_info["slab_bf16"]["initial_build_wall_median_us"],
            "workspace_bytes_before_selector": calls["int4_direct_flashinfer"].workspace_bytes,
        }

    def verify(call):
        result = [call().clone() for _ in range(3)]
        check = {
            "native": [previous.assess(value, expected, native_ids) for value in result],
            "torch": [previous.assess(value, exact, exact_ids) for value in result],
        }
        if hasattr(call, "last_overflow"):
            check["overflow_rows"] = int(call.last_overflow.sum())
        if hasattr(call, "approx_scores"):
            approximate = previous.torch_select(call.approx_scores)
            check["coarse_topk"] = [previous.assess(value, call.approx_scores, approximate)
                                    for value in result]
            sorted_reference = native_ids.sort(dim=-1).values
            width = sorted_reference.shape[1]
            overlaps = []
            for value in result:
                at = torch.searchsorted(sorted_reference, value.contiguous()).clamp_max(width - 1)
                match = (value >= 0) & (sorted_reference.gather(1, at) == value)
                overlaps.append(match.sum(1).tolist())
            check["native_overlap_by_request"] = overlaps
        return check, result[0] if hasattr(call, "approx_scores") else None

    checks = {}
    direct_outputs = {}
    projected_outputs = {name: compare_projection(native, project())
                         for name, project in projections.items()}
    for name, call in calls.items():
        checks[name], result = verify(call)
        if result is not None:
            direct_outputs[name] = result.sort(dim=1).values
    direct_pair_checks = {}
    for left, right in (
        ("int4_direct_flashinfer", "int4_direct_vllm"),
        ("int4_direct_flashinfer", "int4_direct_flashinfer_native_projection"),
        ("int4_direct_vllm", "int4_direct_vllm_native_projection"),
    ):
        if left in direct_outputs and right in direct_outputs:
            match = (direct_outputs[left] == direct_outputs[right]).all(dim=1)
            direct_pair_checks[f"{left}_vs_{right}"] = int(match.sum())
    timings = previous.measure_graph(
        {**calls, **stages}, args.trials, args.graph_iterations, args.seed, args.warmups,
    )
    cuda_alloc = {name: call.arena.keys.storage_bytes for name, call in calls.items()
                  if hasattr(call, "arena")}
    return {
        "input": inputs.metadata, "fixed_selector": fixed_selector_name,
        "projection_check": projected_outputs,
        "slots": previous.page_layout_stats(inputs),
        "raw_index_k_bytes": raw_cache.nbytes,
        "old_lossless_bytes": cuda_alloc, "slab": slab_info,
        "timings": timings, "checks": checks, "direct_pair_exact_sets": direct_pair_checks,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--layouts", type=int, nargs="+", default=[0, 128, 1])
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--request-offset", type=int, default=8)
    parser.add_argument("--growth", type=int, default=64)
    parser.add_argument("--int4", action="store_true", help="also measure the approximate INT4 paths")
    parser.add_argument("--int4-direct", action="store_true", help="select directly from the INT4 scan")
    parser.add_argument("--int4-direct-policy", type=Path,
                        default=THIS / "int4-direct-policy.json",
                        help="registered direct-selection policy and acceptance matrix")
    parser.add_argument("--int4-ablate-2560", action="store_true",
                        help="also measure BF16 refinement with a 2560-candidate shortlist")
    parser.add_argument("--seed", type=int, default=previous.SEED)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--graph-iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    args.skip_flashinfer = False
    args.flashinfer_deterministic = False
    args.ampere_cute = True
    args.direct_native_vllm = False
    if args.int4_direct:
        args.int4_direct_policy = args.int4_direct_policy.resolve()
        direct_policy = json.loads(args.int4_direct_policy.read_text())
        args.direct_native_vllm = (
            direct_policy["serving_projection_control"] == "int4_direct_vllm_native_projection"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from flashinfer import __version__ as flashinfer_version
    from horizonkv_layer_replay import initialize

    torch.set_num_threads(4)
    initialize()
    indexer = previous.load_indexer(args.layer)
    trace = previous.Trace(previous.TRACE)
    sources = [Path(__file__), THIS / "slab_index.py", THIS / "projection.py", THIS / "int4_scan.py",
               THIS / "full-policy.json",
               ROUND1 / "optimize.py", ROUND1 / "bench.py", ROUND1 / "vllm_baseline.py"]
    if args.int4_direct:
        sources.append(args.int4_direct_policy)
    metadata = {
        "args": vars(args), "gpu": str(torch.cuda.get_device_properties(0)),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": {str(p.relative_to(REPO) if p.is_relative_to(REPO) else p):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "torch": torch.__version__, "cuda": torch.version.cuda, "flashinfer": flashinfer_version,
        "timed_scope": "attention_input -> projections, normalization, MRoPE and Hadamard -> "
                       "write current key into the respective cache -> score all history -> select Top-2048",
        "timing": "CUDA Graph, events around repeated full calls, interleaved variants, warm input",
        "layout_scope": "identical Index K slot maps for original, old lossless and fixed_layout paths; "
                        "slab paths use a one-time rebuild into independent request-local Index Key "
                        "storage. Main KV retains its physical layout and output IDs remain logical.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    with args.output.open("x") as stream:
        for width in args.length:
            for batch in args.batch:
                ids = trace.ids(width, batch, args.request_offset)
                case = trace.load(ids, args.layer, args.step, indexer, "merged")
                hidden, positions, _, _ = trace.append_frames(ids, args.layer, args.step)
                projections = {"fused_norm": prepare_projection(indexer, hidden, positions),
                               "fused_epilogue": prepare_projection(indexer, hidden, positions, epilogue=True)}
                pad = -case.page_table.shape[1] % 4
                case.page_table = torch.nn.functional.pad(case.page_table, (0, pad), value=-1)
                case.metadata["length_bucket"] = width
                case.metadata["max_len"] = case.page_table.shape[1]
                for layout in args.layouts:
                    inputs = previous.shuffle_pages(case, layout, args.seed) if layout else case
                    result = benchmark_case(inputs, projections, args)
                    print(json.dumps(result, allow_nan=False), file=stream, flush=True)
                    print(width, batch, inputs.metadata["physical_layout"],
                          {key: round(value["median_us"], 2) for key, value in result["timings"].items()},
                          flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
