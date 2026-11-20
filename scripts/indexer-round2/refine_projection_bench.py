"""Test a frozen fused-projection and exact refinement pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import geometric_mean
import subprocess
from types import SimpleNamespace

import torch

from projection import prepare_projection
from refine_bench import (
    ROOT, TRACE, Trace, assess, load_indexer, logical_scores, measure_graph, predicted_candidates,
    prepared_refiners, scopes, shuffle_pages, torch_candidate_scores, torch_select, write_csv,
)


# Frozen from the fastest correct per-group geomean on the separate L40S pilot.
ROUTES = {
    (4, 3072): "direct_n64w4_vllm_cute_fused_page",
    (16, 3072): "direct_n64w4_vllm_persistent_logical",
    (4, 8192): "native_sgl_fused_page",
    (16, 8192): "native_sgl_fused_page",
}
RADIX = "direct_n64w4_radix_wide1024"
PAIR_FIELDS = (
    "length_bucket", "batch", "candidate_width", "layout", "frozen_route",
    "fastest_old", "fastest_old_us", "old_matched_route_us", "fused_norm_route_us",
    "fused_norm_radix_us", "native_projection_write_us", "fused_projection_write_us",
    "speedup_vs_fastest_old", "speedup_same_refine", "fused_norm_route_valid", "projection_bitwise",
    "raw_line",
)
METHOD_FIELDS = (
    "length_bucket", "batch", "candidate_width", "layout", "method",
    "median_us", "mad_us", "valid_sgl", "valid_torch",
)


def bitwise_projection(native, fused):
    return {
        name: int((expected.view(torch.int16) != observed.view(torch.int16)).sum())
        for name, expected, observed in zip(("q", "k", "gate"), native, fused, strict=True)
    }


def projected_write(inputs, project):
    slot = inputs.page_table.gather(1, inputs.lengths.long()[:, None] - 1).view(-1).long()

    def call():
        q, key, weights = project()
        inputs.key_cache.index_copy_(0, slot, key)
        return q, weights

    return call


def run(args):
    from horizonkv_layer_replay import initialize

    config = initialize()
    trace = Trace(args.trace)
    indexer = load_indexer(1)
    selector_args = SimpleNamespace(skip_flashinfer=False, flashinfer_deterministic=False, ampere_cute=True)
    args.output.mkdir(parents=True, exist_ok=False)
    pairs, all_methods = [], []
    meta = {
        "source_layer": 0, "target_layer": 1, "step": args.step, "request_offset": args.offset,
        "candidate_snapshot": "recomputed before timing" if args.recompute_candidates else str(args.candidates),
        "frozen_selector_calibration": "refine-l40s-freeze/methods.csv, fastest correct geometric mean within batch and candidate width",
        "frozen_routes": {f"{batch}-{width}": method for (batch, width), method in ROUTES.items()},
        "device": str(torch.cuda.get_device_properties()), "torch": torch.__version__,
        "flashinfer": __import__("flashinfer").__version__, "triton": __import__("triton").__version__,
        "methods": "legacy includes all old native-projection methods; fused norm uses native GEMMs, MRoPE and Hadamard, but fuses both norms",
        "repo_revision": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__), Path(__file__).with_name("projection.py"),
                Path(__file__).with_name("refine_bench.py"), Path(__file__).with_name("refine_kernels.py"),
            )
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (args.output / "raw.jsonl").open("x") as log:
        for length in args.length:
            for batch in args.batch:
                ids = trace.ids(length, batch, args.offset)
                inputs = trace.load(ids, 1, args.step, indexer, "merged")
                snapshot_name = f"candidates-{length}-b{batch}.pt"
                if args.recompute_candidates:
                    candidates, tokens = predicted_candidates(
                        trace, inputs, indexer, ids, args.step, args.width, config,
                    )
                    torch.save({
                        "requests": ids, "source_input_ids": tokens, "lengths": inputs.lengths.cpu(),
                        "candidates": {width: rows.cpu() for width, rows in candidates.items()},
                    }, args.output / snapshot_name)
                else:
                    snapshot = torch.load(args.candidates / snapshot_name, weights_only=True)
                    assert snapshot["requests"] == ids and snapshot["lengths"].equal(inputs.lengths.cpu())
                    candidates = {width: snapshot["candidates"][width].cuda() for width in args.width}
                hidden, positions, _, _ = trace.append_frames(ids, 1, args.step)
                fused_project = prepare_projection(indexer, hidden, positions)
                projection_check = bitwise_projection(inputs.project(), fused_project())
                projection_exact = all(err == 0 for err in projection_check.values())
                projected_write(inputs, inputs.project)()
                for layout in args.layout:
                    physical = shuffle_pages(inputs, layout, args.seed) if layout else inputs
                    for width in args.width:
                        projected_write(physical, physical.project)()
                        candidate = candidates[width]
                        methods, sgl_score = prepared_refiners(physical, candidate, selector_args)
                        old_full = scopes(physical, methods, True)["project_write_refine"]
                        route = ROUTES[batch, width]
                        fused_inputs = replace(physical, project=fused_project)
                        fused_full = scopes(
                            fused_inputs, {"fused_norm_route": methods[route], "fused_norm_radix": methods[RADIX]}, True,
                        )["project_write_refine"]
                        scores, _ = sgl_score()
                        torch_scores = torch_candidate_scores(physical, candidate)
                        sgl_scores = logical_scores(scores, candidate, inputs.lengths, physical.page_table.size(1))
                        torch_scores = logical_scores(torch_scores, candidate, inputs.lengths, physical.page_table.size(1))
                        sgl_gold, torch_gold = torch_select(sgl_scores), torch_select(torch_scores)
                        calls = {name: method.call for name, method in {**old_full, **fused_full}.items()}
                        checks = {}
                        for name, fn in calls.items():
                            selected = fn()
                            before = assess(selected, sgl_scores, sgl_gold)
                            audit = assess(selected, torch_scores, torch_gold)
                            checks[name] = {"sgl": before, "torch": audit}
                        stages = {
                            "native_project_write": projected_write(physical, physical.project),
                            "fused_project_write": projected_write(physical, fused_project),
                        }
                        times = measure_graph({**calls, **stages}, args.trials, args.iterations, args.seed, args.warmups)
                        values = {name: value["median_us"] for name, value in times.items()}
                        accepted = {
                            name: checks[name]["sgl"]["valid"] and checks[name]["torch"]["valid"]
                            for name in calls
                        }
                        fastest_old = min((values[name], name) for name in old_full if accepted[name])
                        fields = {
                            "length_bucket": length, "batch": batch, "candidate_width": width,
                            "layout": physical.metadata["physical_layout"],
                        }
                        pair = {
                            **fields, "frozen_route": route, "fastest_old": fastest_old[1],
                            "fastest_old_us": fastest_old[0], "old_matched_route_us": values[route],
                            "fused_norm_route_us": values["fused_norm_route"],
                            "fused_norm_radix_us": values["fused_norm_radix"],
                            "native_projection_write_us": values["native_project_write"],
                            "fused_projection_write_us": values["fused_project_write"],
                            "speedup_vs_fastest_old": fastest_old[0] / values["fused_norm_route"],
                            "speedup_same_refine": values[route] / values["fused_norm_route"],
                            "fused_norm_route_valid": accepted["fused_norm_route"],
                            "projection_bitwise": projection_exact, "raw_line": len(pairs) + 1,
                        }
                        pairs.append(pair)
                        all_methods.extend({
                            **fields, "method": name, "median_us": values[name], "mad_us": times[name]["mad_us"],
                            "valid_sgl": checks[name]["sgl"]["valid"], "valid_torch": checks[name]["torch"]["valid"],
                        } for name in calls)
                        log.write(json.dumps({
                            **pair, "projection_bit_differences": projection_check, "requests": ids,
                            "checks": checks, "timings": times,
                        }) + "\n")
                        log.flush()
                        print(pair, flush=True)
    write_csv(args.output / "paired.csv", pairs, PAIR_FIELDS)
    write_csv(args.output / "methods.csv", all_methods, METHOD_FIELDS)
    valid = [row for row in pairs if row["fused_norm_route_valid"]]
    summary = {
        "points": len(pairs), "bitwise_points": sum(row["projection_bitwise"] for row in pairs),
        "strict_logical_id_points": len(valid),
        "geomean_vs_fastest_old": geometric_mean(row["speedup_vs_fastest_old"] for row in valid) if valid else None,
        "geomean_same_refine": geometric_mean(row["speedup_same_refine"] for row in valid) if valid else None,
        "wins_vs_fastest_old": sum(row["speedup_vs_fastest_old"] > 1 for row in valid),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--width", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--layout", type=int, nargs="+", default=[0, 128, 1])
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/indexer-round2/results/refine-a6000-frozen")
    parser.add_argument("--recompute-candidates", action="store_true", help="generate ordered candidates for this step and cohort")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    with torch.inference_mode():
        run(options)
