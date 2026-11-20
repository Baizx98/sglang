"""A6000 replay of exact candidate refinement on real adjacent-layer predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import geometric_mean
import subprocess
import sys
from types import SimpleNamespace
from typing import NamedTuple

import torch

from refine_kernels import K, prepare_logical_radix, prepare_mapping_selector, prepare_scorer


ROOT = Path(__file__).resolve().parents[2]
ROUND1 = ROOT / "scripts/indexer-optimization"
sys.path.insert(0, str(ROUND1))

from optimize import PackedKeyArena
from bench import TRACE, Trace, assess, load_indexer, make_selectors, measure_graph, shuffle_pages, torch_select
import vllm_baseline


METHOD_FIELDS = (
    "length_bucket", "batch", "candidate_width", "layout", "scope", "method",
    "baseline", "median_us", "mad_us", "valid_sgl", "valid_torch",
)
PAIR_FIELDS = (
    "length_bucket", "batch", "candidate_width", "layout", "scope",
    "baseline", "baseline_us", "primary", "primary_us", "primary_speedup",
    "optimized", "optimized_us", "speedup",
    "mean_candidate_coverage", "raw_line",
)
APPROX_FIELDS = (
    "length_bucket", "batch", "candidate_width", "layout", "scope", "method", "median_us",
    "speedup_vs_exact", "valid_candidates", "mean_overlap_exact_refine", "min_overlap_exact_refine",
    "mean_overlap_full_topk", "min_overlap_full_topk", "exact_refine_mean_overlap_full_topk",
    "mean_full_topk_recall_delta", "raw_line",
)


class RefineMethod(NamedTuple):
    call: object
    baseline: bool
    arena: PackedKeyArena | None = None


def reconstruct_layer0_input(ids, epsilon):
    from safetensors import safe_open
    from horizonkv_layer_replay import MODEL
    from sglang.srt.layers.layernorm import RMSNorm

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    embed_name = "model.embed_tokens.weight"
    rms_name = "model.layers.0.input_layernorm.weight"
    with safe_open(str(MODEL / index[embed_name]), framework="pt", device="cpu") as source:
        embedding = source.get_slice(embed_name)
        values = torch.cat([embedding[token:token+1] for token in ids]).cuda()
    with safe_open(str(MODEL / index[rms_name]), framework="pt", device="cpu") as source:
        params = source.get_tensor(rms_name).cuda()
    assert values.shape[1] == params.numel() and values.dtype == torch.bfloat16
    with torch.device("cuda"):
        norm = RMSNorm(params.numel(), eps=epsilon, weight_dtype=params.dtype)
    norm.weight.copy_(params)
    # Source frames are individual C1 requests from the original SGLang path.
    return torch.cat([norm.forward_native(row[None, :]) for row in values])


def predicted_candidates(trace, inputs, indexer, ids, step, widths, config):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    frames = [trace.read(trace.frames[rid, 1][step]) for rid in ids]
    tokens = [int(frame["input_ids"].item()) for frame in frames]
    wave_outputs = {}
    for rid, token in zip(ids, tokens, strict=True):
        wave = rid.split("__")[1]
        if wave not in wave_outputs:
            path = trace.root / f"input-{wave}.json"
            wave_outputs[wave] = {row["rid"]: row["response"]["output_ids"] for row in json.loads(path.read_text())}
        response = wave_outputs[wave][rid]
        assert step < len(response) - 1 and token == response[step], (rid, token, step)

    previous = reconstruct_layer0_input(tokens, config.rms_norm_eps)
    positions = torch.cat([frame["positions"] for frame in frames], dim=-1).cuda()
    predicted_q, _, predicted_w = indexer._get_q_k_w_bf16(previous, positions, False)
    historical = inputs.lengths - 1
    scores = keye_indexer_score_paged(
        predicted_q, inputs.key_cache, inputs.page_table, predicted_w, historical, inputs.scale,
    )
    # Use the serving convention: keep the current token separately, and
    # obtain the other M-1 predictions exclusively from target-layer history.
    ordered = torch.topk(scores, max(widths) - 1, dim=1, sorted=True).indices.int()
    candidates = {
        width: torch.cat((historical[:, None], ordered[:, :width-1]), dim=1)
        for width in widths
    }
    for width, candidate in candidates.items():
        assert candidate.shape == (len(ids), width)
        assert (candidate >= 0).all() and (candidate < inputs.lengths[:, None]).all()
        assert candidate.sort(dim=1).values.diff(dim=1).gt(0).all()
    return candidates, tokens


def torch_candidate_scores(inputs, candidates):
    rows = []
    for row, logical in enumerate(candidates):
        slots = inputs.page_table[row, logical.long()].long()
        keys = inputs.key_cache.index_select(0, slots).T.contiguous()
        per_head = (inputs.q[row] @ keys).float()
        score = ((per_head * inputs.scale).relu() * inputs.weights[row, :, None].float()).sum(0)
        rows.append(score)
    return torch.stack(rows)


def logical_scores(values, idx, lengths, context):
    result = torch.full((lengths.numel(), context), -float("inf"), dtype=torch.float32, device=idx.device)
    return result.scatter_(1, idx.long(), values)


def prepared_refiners(inputs, candidates, selector_args):
    from flashinfer import top_k_page_table_transform
    from sgl_kernel import fast_topk_transform_fused
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_candidates

    q, key, table, gates, lengths, scale = inputs.args()
    batch, width = candidates.shape
    candidate_lengths = torch.full((batch,), width, dtype=torch.int32, device=q.device)
    fi_map = lambda scores, logical: top_k_page_table_transform(
        scores, logical, candidate_lengths, K
    )
    selectors = make_selectors(candidate_lengths, width, selector_args, [vllm_baseline])
    cumulative_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    sgl_map = lambda scores, logical: fast_topk_transform_fused(
        scores, candidate_lengths, logical, cumulative_q, K
    )
    if selector_args.ampere_cute:
        from ampere_topk.topk import decode_topk

        cute_output = torch.empty((batch, K), dtype=torch.int32, device=q.device)

        def cute_map(scores, logical):
            decode_topk(
                scores, candidate_lengths, cute_output, candidates=logical,
                candidate_block_size=1, key_lens=lengths,
            )
            return cute_output

    def native_score(query=q, weights=gates):
        scores = keye_indexer_score_candidates(query, key, table, candidates, weights, lengths, scale)
        return scores, candidates

    def combine(scorer, selector):
        def call(query=q, weights=gates):
            return selector(*scorer(query, weights))
        return call

    methods = {}
    for name, select in selectors.items():
        methods[f"native_{name}_logical"] = RefineMethod(
            combine(native_score, prepare_mapping_selector(select, candidates)), True,
        )
    methods["native_flashinfer_page"] = RefineMethod(combine(native_score, fi_map), True)
    methods["native_sgl_fused_page"] = RefineMethod(combine(native_score, sgl_map), True)
    if selector_args.ampere_cute:
        methods["native_vllm_cute_fused_page"] = RefineMethod(combine(native_score, cute_map), True)

    # The strongest tuned BF16 baseline may change tile size without changing
    # its exact score or the FlashInfer selector's logical-ID contract.
    for tile, warps in ((64, 4), (64, 2), (128, 4)):
        scorer = prepare_scorer(q, key, table, gates, lengths, candidates, scale, tile, warps)
        methods[f"direct_n{tile}w{warps}_flashinfer_page"] = RefineMethod(combine(scorer, fi_map), True)
        if selector_args.ampere_cute:
            methods[f"direct_n{tile}w{warps}_vllm_cute_fused_page"] = RefineMethod(
                combine(scorer, cute_map), True,
            )
        for name in ("vllm_persistent", "vllm_ampere_cute"):
            if name in selectors:
                methods[f"direct_n{tile}w{warps}_{name}_logical"] = RefineMethod(
                    combine(scorer, prepare_mapping_selector(selectors[name], candidates)), True,
                )
        for threads, wide_digits in ((256, False), (512, False), (512, True), (1024, True)):
            radix = prepare_logical_radix(candidate_lengths, width, threads=threads, wide_digits=wide_digits)
            suffix = f"radix_wide{threads}" if wide_digits else f"radix{threads}"
            methods[f"direct_n{tile}w{warps}_{suffix}"] = RefineMethod(combine(scorer, radix), False)

    for tile, warps in ((64, 4), (128, 4)):
        scorer = prepare_scorer(
            q, key, table, gates, lengths, candidates, scale, tile, warps, sort_local=True,
        )
        methods[f"tile_sort_n{tile}w{warps}_flashinfer_page"] = RefineMethod(
            combine(scorer, fi_map), False,
        )
    packed = PackedKeyArena.from_cache(key, reserve=batch)
    packed.reserve_insertions(batch)
    score_packed = prepare_scorer(
        q, key, table, gates, lengths, candidates, scale, 128, 4, encoded=packed.keys,
    )
    methods["packed_n128w4_flashinfer_page"] = RefineMethod(
        combine(score_packed, fi_map), True, packed,
    )
    if selector_args.ampere_cute:
        methods["packed_n128w4_vllm_cute_fused_page"] = RefineMethod(
            combine(score_packed, cute_map), True, packed,
        )
    return methods, native_score


def prepared_approximations(inputs, candidates):
    from flashinfer import top_k_page_table_transform
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_candidates

    q, key, table, weights, lengths, scale = inputs.args()
    batch, width = candidates.shape

    def prefix_only(query=q, gates=weights):
        return candidates[:, :K].contiguous()

    methods = {"predicted_prefix2048": RefineMethod(prefix_only, False)}
    for keep in (1024, 1536):
        prefix = candidates[:, :keep]
        tail = candidates[:, keep:]
        tail_lengths = torch.full((batch,), width - keep, dtype=torch.int32, device=q.device)

        def boundary(query=q, gates=weights, prefix=prefix, tail=tail, tail_lengths=tail_lengths, keep=keep):
            scores = keye_indexer_score_candidates(query, key, table, tail, gates, lengths, scale)
            rest = top_k_page_table_transform(scores, tail, tail_lengths, K - keep)
            return torch.cat((prefix, rest), dim=1)

        methods[f"protect{keep}_refine_rest"] = RefineMethod(boundary, False)
    return methods


def overlap_fraction(values, sorted_reference):
    values = values.to(sorted_reference.dtype).contiguous()
    offsets = torch.searchsorted(sorted_reference, values).clamp_max(sorted_reference.size(1) - 1)
    return (sorted_reference.gather(1, offsets) == values).sum(1).float() / K


def verify_approximation(output, logical_values, exact, full):
    ordered = output.sort(dim=1).values
    valid = (output >= 0) & (output < logical_values.size(1))
    values = logical_values.gather(1, output.long().clamp(0, logical_values.size(1) - 1))
    assert output.shape == exact.shape and output.dtype == torch.int32
    return {
        "valid_candidates": bool(valid.all() & values.isfinite().all() & ordered.diff(dim=1).gt(0).all()),
        "overlap_exact_refine": overlap_fraction(output, exact).tolist(),
        "overlap_full_topk": overlap_fraction(output, full).tolist(),
        "exact_refine_overlap_full_topk": overlap_fraction(exact, full).tolist(),
    }


def scopes(inputs, methods, full):
    result = {"refine": methods}
    if not full:
        return result
    tail = inputs.page_table.gather(1, inputs.lengths.long()[:, None] - 1).view(-1).long()

    def refresh(method):
        overflow = torch.empty_like(inputs.lengths) if method.arena is not None else None

        def invoke():
            query, current_k, gates = inputs.project()
            if method.arena is None:
                inputs.key_cache.index_copy_(0, tail, current_k)
            else:
                method.arena.write_reserved(current_k, tail, overflow)
            return method.call(query, gates)

        return invoke

    result["project_write_refine"] = {
        name: RefineMethod(refresh(method), method.baseline, method.arena)
        for name, method in methods.items()
    }
    return result


def print_row(row):
    if row["speedup"] is not None:
        print(
            {k: row[k] for k in ("length_bucket", "batch", "candidate_width", "layout", "scope")},
            f"{row['baseline']}={row['baseline_us']:.3f}us",
            f"{row['optimized']}={row['optimized_us']:.3f}us",
            f"{row['speedup']:.3f}x",
            flush=True,
        )


def summarize(records, methods):
    overall = {}
    for scope in sorted({row["scope"] for row in records}):
        pooled = [row for row in records if row["scope"] == scope]
        per_method = {}
        for name in sorted({row["method"] for row in methods if not row["baseline"]}):
            actual = [row for row in methods if row["scope"] == scope and row["method"] == name]
            by_point = {
                tuple(row[k] for k in PAIR_FIELDS[:4]): row["median_us"]
                for row in actual if row["valid_sgl"] and row["valid_torch"]
            }
            ratios = [
                row["baseline_us"] / by_point[tuple(row[k] for k in PAIR_FIELDS[:4])]
                for row in pooled if row["baseline_us"] is not None
                and tuple(row[k] for k in PAIR_FIELDS[:4]) in by_point
            ]
            per_method[name] = {
                "points": len(ratios),
                "geomean": geometric_mean(ratios) if ratios else None,
                "wins": sum(ratio > 1 for ratio in ratios),
            }
        ratios = [row["speedup"] for row in pooled if row["speedup"] is not None]
        primary = [row["primary_speedup"] for row in pooled if row["primary_speedup"] is not None]
        overall[scope] = {
            "points": len(pooled), "points_with_correct_optimized_method": len(ratios),
            "primary_method": pooled[0]["primary"],
            "primary_points": len(primary),
            "primary_geomean": geometric_mean(primary) if primary else None,
            "primary_wins": sum(ratio > 1 for ratio in primary),
            "pointwise_best_geomean": geometric_mean(ratios) if ratios else None,
            "pointwise_best_wins": sum(ratio > 1 for ratio in ratios),
            "fixed_methods": per_method,
        }
    return overall


def write_csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args):
    from horizonkv_layer_replay import MODEL, initialize

    config = initialize()
    trace = Trace(args.trace)
    indexer = load_indexer(1)
    selector_args = SimpleNamespace(
        skip_flashinfer=False, flashinfer_deterministic=False, ampere_cute=not args.skip_cute,
    )
    rounds = args.output
    rounds.mkdir(parents=True, exist_ok=False)
    point_csv, method_csv, approx_csv = [], [], []
    raw_path = rounds / "raw.jsonl"
    meta = {
        "seed": args.seed, "target_layer": 1, "source_layer": 0,
        "step": args.step, "request_offset": args.offset, "k": K,
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "revision": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("refine*")) if path.is_file()
        },
        "fixture_manifest_sha256": hashlib.sha256(
            (trace.root / "fixtures/manifest_rank00.jsonl").read_bytes()
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest(),
        "device": str(torch.cuda.get_device_properties()),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "flashinfer": __import__("flashinfer").__version__,
        "prediction": "observed decode input IDs -> model embed_tokens -> native layer 0 input RMSNorm -> layer 1 indexer, history excludes current token; reserve current logical ID",
        "approximation": "same predicted-score-ordered candidates; prefix extraction/copy, optional true scoring and selection in remaining candidates, and final logical-ID output timed; shared prediction ordering outside all timed scopes",
        "timing": "CUDA events on randomized interleaved fixed-input graph replays; layout rearrangement and prediction outside timing; logical candidate transforms, scoring, selection and ID output inside",
    }
    (rounds / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    line = 0
    with raw_path.open("x") as raw:
        for context in args.length:
            for batch in args.batch:
                ids = trace.ids(context, batch, args.offset)
                assert len(ids) == batch and len(set(ids)) == batch
                original = trace.load(ids, 1, args.step, indexer, "merged")
                candidates, tokens = predicted_candidates(trace, original, indexer, ids, args.step, args.width, config)
                candidate_path = rounds / f"candidates-{context}-b{batch}.pt"
                torch.save({
                    "requests": ids, "source_input_ids": tokens, "lengths": original.lengths.cpu(),
                    "candidates": {width: rows.cpu() for width, rows in candidates.items()},
                }, candidate_path)
                _, key, _ = original.project()
                current = original.page_table.gather(1, original.lengths.long()[:, None] - 1).view(-1).long()
                original.key_cache.index_copy_(0, current, key)
                exact_all = keye_full_scores(original)
                global_topk = exact_all.topk(K, dim=1).indices.sort(dim=1).values
                for block in args.layout:
                    inputs = shuffle_pages(original, block, args.seed) if block else original
                    for width in args.width:
                        idx = candidates[width]
                        position = torch.searchsorted(global_topk, idx.long()).clamp_max(K - 1)
                        coverage = (global_topk.gather(1, position) == idx).sum(1) / K
                        methods, canonical = prepared_refiners(inputs, idx, selector_args)
                        if args.primary is not None and (
                            args.primary not in methods or methods[args.primary].baseline
                        ):
                            raise ValueError(f"unknown optimized primary method: {args.primary}")
                        scoped = scopes(inputs, methods, args.full_prepare)
                        approximate_scoped = scopes(
                            inputs, prepared_approximations(inputs, idx), args.full_prepare,
                        ) if args.approximate else {}
                        native_values, _ = canonical()
                        torch_values = torch_candidate_scores(inputs, idx)
                        context_width = inputs.page_table.size(1)
                        # Paged layout permutation must preserve the logical score.
                        gathered = exact_all.gather(1, idx.long())
                        assert torch.equal(native_values, gathered)
                        sgl_logical = logical_scores(native_values, idx, inputs.lengths, context_width)
                        independent = logical_scores(torch_values, idx, inputs.lengths, context_width)
                        sgl_reference, torch_reference = torch_select(sgl_logical), torch_select(independent)
                        gap = float((native_values - torch_values).abs().max())
                        for scope, paths in scoped.items():
                            calls = {name: method.call for name, method in paths.items()}
                            approximate_calls = {
                                name: method.call for name, method in approximate_scoped.get(scope, {}).items()
                            }
                            checks = {}
                            for name, fn in calls.items():
                                output = fn()
                                checks[name] = {
                                    "sgl": assess(output, sgl_logical, sgl_reference),
                                    "torch": assess(output, independent, torch_reference),
                                }
                            exact_sorted = sgl_reference.sort(dim=1).values
                            approximation = {
                                name: verify_approximation(fn(), sgl_logical, exact_sorted, global_topk)
                                for name, fn in approximate_calls.items()
                            }
                            timings = measure_graph(
                                {**calls, **approximate_calls}, args.trials, args.iterations, args.seed, args.warmups,
                            )
                            fields = {
                                "length_bucket": context, "batch": batch, "candidate_width": width,
                                "layout": inputs.metadata["physical_layout"], "scope": scope,
                            }
                            raw.write(json.dumps({
                                **fields, "requests": ids, "source_input_ids": tokens,
                                "candidate_file": candidate_path.name,
                                "lengths": inputs.metadata["sequence_lengths"], "candidate_coverage": coverage.tolist(),
                                "native_vs_torch_max_score_abs": gap, "checks": checks, "timing": timings,
                                "approximation": approximation,
                            }) + "\n")
                            raw.flush()
                            line += 1
                            for name in paths:
                                timing = timings[name]
                                method_csv.append({
                                    **fields, "method": name, "baseline": methods[name].baseline,
                                    "median_us": timing["median_us"], "mad_us": timing["mad_us"],
                                    "valid_sgl": checks[name]["sgl"]["valid"], "valid_torch": checks[name]["torch"]["valid"],
                                })
                            legal = {
                                name: timings[name]["median_us"] for name in paths
                                if checks[name]["sgl"]["valid"] and checks[name]["torch"]["valid"]
                            }
                            base = min(
                                ((time, name) for name, time in legal.items() if paths[name].baseline),
                                default=(None, None),
                            )
                            best = min(
                                ((time, name) for name, time in legal.items() if not paths[name].baseline),
                                default=(None, None),
                            )
                            primary = legal.get(args.primary)
                            row = {
                                **fields, "baseline": base[1], "baseline_us": base[0], "optimized": best[1],
                                "primary": args.primary, "primary_us": primary,
                                "primary_speedup": base[0] / primary if base[0] and primary else None,
                                "optimized_us": best[0],
                                "speedup": base[0] / best[0] if base[0] and best[0] else None,
                                "mean_candidate_coverage": float(coverage.mean()), "raw_line": line,
                            }
                            point_csv.append(row)
                            for name, validity in approximation.items():
                                near = validity["overlap_exact_refine"]
                                full = validity["overlap_full_topk"]
                                exact_full = validity["exact_refine_overlap_full_topk"]
                                median = timings[name]["median_us"]
                                approx_csv.append({
                                    **fields, "method": name, "median_us": median,
                                    "speedup_vs_exact": base[0] / median if base[0] else None,
                                    "valid_candidates": validity["valid_candidates"],
                                    "mean_overlap_exact_refine": sum(near) / len(near),
                                    "min_overlap_exact_refine": min(near),
                                    "mean_overlap_full_topk": sum(full) / len(full),
                                    "min_overlap_full_topk": min(full),
                                    "exact_refine_mean_overlap_full_topk": sum(exact_full) / len(exact_full),
                                    "mean_full_topk_recall_delta": (sum(full) - sum(exact_full)) / len(full),
                                    "raw_line": line,
                                })
                            print_row(row)
    write_csv(rounds / "methods.csv", method_csv, METHOD_FIELDS)
    write_csv(rounds / "paired.csv", point_csv, PAIR_FIELDS)
    if args.approximate:
        write_csv(rounds / "approximate.csv", approx_csv, APPROX_FIELDS)
    summary = summarize(point_csv, method_csv)
    summary["failed_method_points"] = [
        row for row in method_csv if not row["valid_sgl"] or not row["valid_torch"]
    ]
    (rounds / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def keye_full_scores(inputs):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    return keye_indexer_score_paged(*inputs.args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--width", type=int, nargs="+", choices=(3072, 8192), default=[3072, 8192])
    parser.add_argument("--layout", type=int, nargs="+", choices=(0, 128, 1), default=[0, 128, 1])
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--full-prepare", action="store_true")
    parser.add_argument("--primary", help="fixed optimized method selected before the timing matrix")
    parser.add_argument("--approximate", action="store_true", help="compare prefix-only and protected boundaries")
    parser.add_argument("--skip-cute", action="store_true")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(arguments.seed)
    with torch.inference_mode():
        benchmark(arguments)
