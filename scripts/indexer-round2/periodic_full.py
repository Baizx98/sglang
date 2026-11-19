"""Replay complete and partial periodic Top-K refresh cycles on real Decode steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from full_index import REPO, THIS, previous, lossless, vllm_baseline, prepare_projection, IndexSlab, Plan

VALID_SELECTORS = {"flashinfer_topk", "vllm_persistent", "vllm_ampere_cute"}
LAYOUTS = (0, 128, 1)
STEPS = (0, 1, 2, 3, 4)


def series(calls):
    return lambda: tuple(call() for call in calls)


def periodic(full, append, period, start=0, end=len(STEPS)):
    def call():
        outputs = []
        previous_topk = None
        for step in range(start, end):
            if step % period == 0:
                previous_topk = full[step]()
            else:
                append[step]()
            outputs.append(previous_topk)
        return tuple(outputs)

    if start % period != 0:
        raise ValueError("a replay slice must begin at a known refresh step")
    return call


def prepare_old(pack, query, key, table, gates, lengths, scale, project, select):
    score = lossless._make_scoring_variant(
        query, None, table, gates, lengths, scale, 128, 4, encoded=pack.keys,
    )
    overflow = torch.empty_like(lengths)
    batch = lengths.numel()

    def run():
        q, current, weights = project()
        buffers = pack.keys
        lossless._pack_tail_keys[(batch,)](
            current, table, lengths, *buffers, pack.next_exception, overflow,
            *table.stride(), lengths.stride(0), buffers.exceptions.size(0),
            buffers.headers.numel(), table.size(1), buffers.headers.numel() * 64 >= 2**31,
            True, current.stride(0), num_warps=4,
        )
        return select(score(q, weights, buffers))

    return run, overflow


def benchmark_case(indexer, frames, inputs, args):
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

    positions = torch.arange(inputs.page_table.shape[1], device="cuda")
    table = inputs.page_table
    raw = inputs.key_cache
    initial = frames[0][2]
    final = frames[-1][2]
    future = (positions >= initial[:, None]) & (positions < final[:, None])
    raw[table[future].long()] = 0

    plan = Plan.from_lengths(final, growth=args.growth)
    batch = final.numel()
    reserve = batch * len(frames) + batch * args.growth
    slab = IndexSlab(plan, raw, table, initial, packed=True, exception_reserve=reserve)
    old_pack = lossless.PackedKeyArena.from_cache(raw, reserve=batch * len(frames))
    old_pack.reserve_insertions(batch * len(frames))

    selectors_args = SimpleNamespace(skip_flashinfer=False, flashinfer_deterministic=False, ampere_cute=True)
    baseline_steps = {}
    old_steps = {}
    old_overflows = []
    full_steps, append_steps = [], []
    truths, references, key_checks = [], [], []
    exact_reference = raw.clone()
    block = 64 if inputs.metadata["length_bucket"] == 32768 else 128
    fixed_selector = "vllm_persistent" if batch <= 4 else "vllm_ampere_cute"

    for hidden, positions, lengths, _ in frames:
        def native_project(hidden=hidden, positions=positions):
            return indexer._get_q_k_w_bf16(hidden, positions, False)

        def key_only(hidden=hidden, positions=positions):
            return indexer._get_k_bf16(hidden, positions)

        def append(key_only=key_only, lengths=lengths):
            return slab.append(key_only(), lengths)

        slots = table.gather(1, lengths.long()[:, None] - 1).view(-1).long()

        def native_score(project=native_project, slots=slots, lengths=lengths):
            query, key, gates = project()
            raw.index_copy_(0, slots, key)
            return keye_indexer_score_paged(query, raw, table, gates, lengths, inputs.scale)

        selectors = previous.make_selectors(lengths, table.shape[1], selectors_args, [vllm_baseline])
        fused = prepare_projection(indexer, hidden, positions)
        refresh_project = native_project if args.refresh_projection == "native" else fused
        for name, selector in selectors.items():
            baseline_steps.setdefault(name, []).append(
                lambda selector=selector, scorer=native_score: selector(scorer()))
            if name in VALID_SELECTORS:
                old_path, flag = prepare_old(
                    old_pack, inputs.q, raw, table, inputs.weights, lengths, inputs.scale,
                    native_project, selector,
                )
                old_steps.setdefault(name, []).append(old_path)
                old_overflows.append(flag)
        full_steps.append(slab.prepare(refresh_project, selectors[fixed_selector], inputs.scale, block, lengths))
        append_steps.append(append)

        q, current_k, gates = native_project()
        fused_k = fused()[1]
        skip_k = key_only()
        key_checks.append({
            "fused_diff_bf16": int((current_k.view(torch.int16) != fused_k.view(torch.int16)).sum()),
            "key_only_diff_bf16": int((current_k.view(torch.int16) != skip_k.view(torch.int16)).sum()),
            "refresh_diff_bf16": 0 if args.refresh_projection == "native"
            else int((current_k.view(torch.int16) != fused_k.view(torch.int16)).sum()),
        })
        exact_reference.index_copy_(0, slots, current_k)
        truth = keye_indexer_score_paged(q, exact_reference, table, gates, lengths, inputs.scale)
        truths.append(truth)
        references.append(previous.torch_select(truth))

    methods = {f"native_{name}": series(items) for name, items in baseline_steps.items()}
    methods.update({f"old_lossless_{name}": series(items) for name, items in old_steps.items()})
    methods["always_slab"] = series(full_steps)
    checks = {}
    for refresh in (3, 4):
        methods[f"period{refresh}"] = periodic(full_steps, append_steps, refresh)

    for name, call in methods.items():
        outputs = call()
        checks[name] = [previous.assess(out, scores, ref)
                        for out, scores, ref in zip(outputs, truths, references)]
        if name.startswith(("period", "always_slab")) and int(slab.overflow.sum()):
            raise RuntimeError(f"slab append overflow on {name}")
    if any(int(flag.sum()) for flag in old_overflows):
        raise RuntimeError("old lossless append overflow")

    timed = dict(methods)
    for refresh in (3, 4):
        timed[f"period{refresh}_cycle"] = periodic(full_steps, append_steps, refresh, end=refresh)
        timed[f"period{refresh}_tail"] = periodic(full_steps, append_steps, refresh, start=refresh)
    timing = previous.measure_graph(timed, args.trials, args.graph_iterations, args.seed, args.warmups)
    return {
        "input": inputs.metadata,
        "refresh_projection": args.refresh_projection,
        "steps": list(STEPS), "refresh_steps": {"period3": [0, 3], "period4": [0, 4]},
        "selector": fixed_selector, "key_checks": key_checks,
        "timings": timing, "checks": checks,
        "state_bytes": {"key_slab": slab.storage_bytes, "topk_cache": batch * 2048 * 4},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--layout", type=int, nargs="+", default=list(LAYOUTS))
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--request-offset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=previous.SEED)
    parser.add_argument("--growth", type=int, default=64)
    parser.add_argument("--refresh-projection", choices=("native", "fused_norm"), default="native")
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--graph-iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def main():
    from flashinfer import __version__ as flashinfer_version
    from horizonkv_layer_replay import initialize

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    initialize()
    indexer = previous.load_indexer(args.layer)
    trace = previous.Trace(previous.TRACE)
    source_paths = [Path(__file__), THIS / "slab_index.py", THIS / "projection.py", THIS / "full_index.py",
                    THIS / "periodic-policy.json"]
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "args": vars(args), "gpu": str(torch.cuda.get_device_properties(0)),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
        "flashinfer": flashinfer_version,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "bench": "five real consecutive forwards, same raw fixture and baseline contract, no scheduling or main attention",
        "length_baseline": "step-4 positions allocate stable physical slots; slots after step 0 start empty",
        "cycles": "period 3 refreshes steps 0 and 3. Period 4 refreshes 0 and 4. "
                  "Whole timing always includes all five steps. First cycle and final tail also measured.",
    }, indent=2, default=str) + "\n")
    with args.output.open("x") as stream:
        for width in args.length:
            for batch in args.batch:
                ids = trace.ids(width, batch, args.request_offset)
                frames = [trace.append_frames(ids, args.layer, step) for step in STEPS]
                case = trace.load(ids, args.layer, STEPS[-1], indexer, "merged")
                case.page_table = torch.nn.functional.pad(
                    case.page_table, (0, -case.page_table.shape[1] % 4), value=-1,
                )
                case.metadata["length_bucket"] = width
                case.metadata["max_len"] = case.page_table.shape[1]
                for layout in args.layout:
                    inputs = previous.shuffle_pages(case, layout, args.seed) if layout else case
                    report = benchmark_case(indexer, frames, inputs, args)
                    print(json.dumps(report, allow_nan=False), file=stream, flush=True)
                    print(width, batch, inputs.metadata["physical_layout"], {
                        name: round(values["median_us"] / len(STEPS), 2)
                        for name, values in report["timings"].items() if not name.endswith(("cycle", "tail"))
                    }, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
