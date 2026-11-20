"""Measure the read-only Keye cross-layer predictor, from hidden state to Top-M."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in ("scripts/indexer-optimization", "scripts/indexer-round2",
                  "scripts/keye_trace", "data/indexer-optimization/vllm-selector"):
    sys.path.insert(0, str(ROOT / directory))

from ampere_topk.topk import decode_topk, workspace_bytes as cute_workspace_bytes
from bench import TRACE, Trace, load_indexer, measure_graph, page_layout
from flashinfer import top_k_ragged_transform
from horizonkv_layer_replay import initialize
from lossless import StaticExponentKeys
from progressive import prepare_progressive, validate as validate_progressive
from quantized import QuantizedKeys
from query import predictor_projection
from refine_bench import reconstruct_layer0_input
from segment_oracle import audit as audit_segments
from slab_index import IndexSlab, Plan, _score_slab
from temporal import prepare_temporal, validate_temporal
from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged

K = 2048


@dataclass
class Replay:
    lengths: list[int]
    keys: torch.Tensor
    hidden: torch.Tensor
    positions: torch.Tensor
    target_ids: torch.Tensor
    current_keys: torch.Tensor
    request_ids: list[str]


class CapturedInputs:
    def __init__(self, directories):
        self.groups = defaultdict(list)
        for directory in directories:
            completed = directory / "completed.jsonl"
            if not completed.is_file():
                continue
            selected = {json.loads(s)["rid"] for s in completed.read_text().splitlines()}
            for s in (directory / "requests.jsonl").read_text().splitlines():
                row = json.loads(s)
                if row["rid"] in selected:
                    identity = (row["length_bucket_upper"], row["split"])
                    self.groups[identity].append((row["rid"], directory / "fixtures"))
        for group in self.groups.values():
            group.sort()
            if len({rid for rid, _ in group}) != len(group):
                raise ValueError("duplicate completed capture across input directories")

    def load(self, bucket, split, batch, layer, step, offset):
        group = self.groups[bucket, split][offset:offset + batch]
        if len(group) != batch:
            raise ValueError(f"need {batch} distinct captures for {bucket}/{split}, found {len(group)}")
        histories, frames, ids = [], [], []
        for rid, directory in group:
            prefix = directory / f"{rid}-l{layer:02}"
            keys = torch.load(f"{prefix}-keys.pt", weights_only=True, mmap=True)
            prior = [
                torch.load(f"{prefix}-s{s:02}.pt", weights_only=True, mmap=True)
                for s in range(1, step + 1)
            ]
            frame = prior[-1] if prior else torch.load(f"{prefix}-s00.pt", weights_only=True)
            if step:
                history = torch.cat([keys, *(data["target_key"] for data in prior[:-1])])
            else:
                history = keys[:-1]
            if history.size(0) + 1 != frame["length"]:
                raise ValueError(f"history is incomplete for {rid} layer {layer} step {step}")
            ids.append(rid)
            histories.append(history)
            frames.append(frame)
        return Replay(
            [len(h) for h in histories], torch.cat(histories).cuda(),
            torch.cat([f["source_hidden"] for f in frames]).cuda(),
            torch.cat([f["position"] for f in frames], dim=-1).cuda(),
            torch.cat([f["target_topk"] for f in frames]).cuda(),
            torch.cat([f["target_key"] for f in frames]).cuda(), ids,
        )


def existing_inputs(trace, bucket, batch, step, offset, indexer, config):
    ids = trace.ids(bucket, batch, offset)
    if len(ids) != batch or len(set(ids)) != batch:
        raise ValueError("pilot requires different real requests")
    captured = trace.load(ids, 1, step, indexer, "merged")
    _, positions, lengths, _ = trace.append_frames(ids, 1, step)
    tokens = [int(trace.read(trace.frames[rid, 1][step])["input_ids"].item()) for rid in ids]
    hidden = reconstruct_layer0_input(tokens, config.rms_norm_eps)
    all_lengths = lengths.tolist()
    histories, current = [], []
    for raw in captured.key_cache.split(all_lengths):
        histories.append(raw[:-1])
        current.append(raw[-1])
    return Replay([n - 1 for n in all_lengths], torch.cat(histories), hidden, positions,
                  captured.captured, torch.stack(current), ids)


def prepare_slab_score(slab, scale, block, warps):
    lengths = slab.lengths
    width = triton.cdiv(slab.plan.width, block) * block
    scores = torch.empty((lengths.numel(), width), device=lengths.device, dtype=torch.float32)
    buffers, _ = slab.buffers()

    def score(q, gates):
        _score_slab[(width // block, lengths.numel())](
            q, gates, slab.plan.bases, slab.plan.capacities, lengths, *buffers[:4], scores,
            q.stride(0), q.stride(1), gates.stride(0), width, scale, block, slab.packed,
            num_warps=warps,
        )
        return scores

    return score, scores.nbytes


def selectors(lengths, width, reference_width):
    offsets = torch.zeros_like(lengths)
    # CuTe requires k divisible by four. Treat the reserved, known current
    # logical ID as a synthetic winner; no target Key is scored or written.
    cute_out = torch.empty((lengths.numel(), width), dtype=torch.int32, device=lengths.device)
    current_ids = lengths[:, None].long()
    inclusive = lengths + 1

    def cute(scores):
        scores.scatter_(1, current_ids, float("inf"))
        decode_topk(scores, inclusive, cute_out)
        return cute_out[:, :-1]

    result = {
        "torch_sorted": lambda scores: scores[:, :reference_width].contiguous().topk(
            width - 1, sorted=True,
        ).indices.int(),
        "torch_unsorted": lambda scores: scores[:, :reference_width].contiguous().topk(
            width - 1, sorted=False,
        ).indices.int(),
        "flashinfer": lambda scores: top_k_ragged_transform(scores, offsets, lengths, width - 1),
    }
    # The upstream M=8192 specialization requests 139536 bytes of shared
    # memory, exceeding the A6000 limit. Avoid its fatal CUDA launch.
    if width <= 3072 and width % 4 == 0:
        result["vllm_cute"] = cute
    return result


def prepare_call(project, score, select, current):
    def call():
        values = score(*project())
        return torch.cat((current, select(values)), dim=1)

    return call


def reference_topm(scores, lengths, width):
    # The original predictor uses Torch sorted Top-M. Even same-score
    # substitutions at its cutoff are excluded from the exact-route results.
    gold_scores, gold = scores.topk(width - 1, sorted=True)
    cutoff = gold_scores[:, -1:]
    ties = ((scores == cutoff).sum(dim=1) != (gold_scores == cutoff).sum(dim=1))
    return torch.cat((lengths[:, None], gold.int()), dim=1), ties


def intersections(query, reference):
    ordered = reference.sort(dim=1).values.contiguous()
    rank = torch.searchsorted(ordered, query.contiguous()).clamp_max(reference.size(1) - 1)
    return (ordered.gather(1, rank) == query).sum(dim=1)


def timed_build(factory):
    # Exclude first-use compilation and collect a warm allocation/build time.
    warmed = factory()
    torch.cuda.synchronize()
    del warmed
    samples = []
    for i in range(3):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        storage = factory()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6)
        if i != 2:
            del storage
    return storage, statistics.median(samples)


def benchmark(replay, indexer, args, identity, previous_replay=None, codebook=None):
    scale = indexer.softmax_scale
    reference_width = max(replay.lengths) + 1
    width = triton.cdiv(reference_width, 256) * 256
    pages, lengths = page_layout(replay.lengths, width)
    current = lengths[:, None]
    slots = replay.keys
    plan = Plan.from_lengths(lengths, 64)
    raw, build_us = timed_build(lambda: IndexSlab(
        plan, slots, pages, lengths, packed=False, exception_reserve=0,
    ))
    packed, packed_us = timed_build(lambda: IndexSlab(
        plan, slots, pages, lengths, packed=True, exception_reserve=len(replay.lengths),
    ))
    bf16, _ = prepare_slab_score(raw, scale, 128, 4)
    q_only_native = predictor_projection(indexer, replay.hidden, replay.positions)
    q_only_fused = predictor_projection(indexer, replay.hidden, replay.positions, fused=True)

    def legacy_projection():
        q, _, gate = indexer._get_q_k_w_bf16(replay.hidden, replay.positions, False)
        return q, gate

    default_q, default_gate = legacy_projection()
    native_q, native_gate = q_only_native()
    if not torch.equal(default_q, native_q) or not torch.equal(default_gate, native_gate):
        raise RuntimeError("the Q-only projection changed Q or gate")
    fused_q, fused_gate = q_only_fused()
    fused_projection_exact = torch.equal(default_q, fused_q) and torch.equal(default_gate, fused_gate)
    use_fused_projection = args.fused_query and fused_projection_exact
    q_only = q_only_fused if use_fused_projection else q_only_native
    fast_q, fast_gate = (fused_q, fused_gate) if use_fused_projection else (native_q, native_gate)

    def native(q, gate):
        return keye_indexer_score_paged(q, slots, pages, gate, lengths, scale)

    native_scores = native(default_q, default_gate).clone()
    raw_scores = bf16(default_q, default_gate)[:, :width]
    if not torch.equal(raw_scores, native_scores):
        raise RuntimeError("contiguous BF16 scorer and original page-table scorer differ")
    tokens = sum(replay.lengths)
    raw_bytes = tokens * replay.keys.size(1) * replay.keys.element_size()
    logical_key_bytes = {"paged": raw_bytes}
    score_methods = {"paged": (native, width * len(replay.lengths) * 4, True, slots.nbytes, 0.0)}
    for tile, warps in ((64, 4), (128, 4), (256, 4)):
        scorer, workspace = prepare_slab_score(raw, scale, tile, warps)
        name = f"bf16_b{tile}"
        score_methods[name] = (scorer, workspace, True, raw.storage_bytes, build_us)
        logical_key_bytes[name] = raw_bytes
    packed_score, packed_workspace = prepare_slab_score(packed, scale, 128, 4)
    score_methods["lossless12"] = (packed_score, packed_workspace, True, packed.storage_bytes, packed_us)
    packed_exceptions = int((packed.arena.keys.headers < 0).sum())
    logical_key_bytes["lossless12"] = (100 * tokens) + 128 * packed_exceptions
    static, static_us = timed_build(lambda: StaticExponentKeys(raw))
    static_mask_groups = sum(triton.cdiv(length, 32) for length in replay.lengths)
    static_exception_groups = int(static.masks.view(torch.int32).count_nonzero())
    static_logical_bytes = 96 * tokens + 4 * (
        static_mask_groups + static_exception_groups
    ) + 128 * static.exception_count
    for tile in args.exact_tiles:
        variants = (("", True, ""), ("_m", False, ""), ("_cg", False, ".cg")) if args.exact_tuning else (("", True, ""),)
        for suffix, branched, modifier in variants:
            scorer, workspace = static.prepare(
                lengths, scale, tile, branched=branched, cache_modifier=modifier,
            )
            name = f"static12_b{tile}{suffix}"
            logical_key_bytes[name] = static_logical_bytes
            score_methods[name] = (
                scorer, workspace, True, static.storage_bytes, build_us + static_us,
            )

    formats = {}
    variants = [
        (int(bits), int(groups), 0, False, None)
        for bits, groups in (name.split("x") for name in args.quant_formats)
    ]
    variants.extend((4, groups, steps, False, None)
                    for steps in (args.fit_int4 or []) for groups in (1, 2))
    if args.fp4:
        variants.extend((4, groups, 0, True, None) for groups in (1, 2))
    if codebook is not None:
        variants.append((4, 1, 0, False, codebook))
    for bits, groups, fit_steps, fp4, table in variants:
        if args.direct_paged_build:
            state, code_us = timed_build(lambda: QuantizedKeys.from_paged(
                plan, slots, pages, lengths, bits, groups, fit_steps, fp4, table,
            ))
            initialization_us = code_us
        else:
            state, code_us = timed_build(lambda: QuantizedKeys(raw, bits, groups, fit_steps, fp4, table))
            initialization_us = build_us + code_us
        fit = f"fit{fit_steps}" if fit_steps else ""
        codec = "pq" if table is not None else "fp" if fp4 else "int"
        name = f"{codec}{bits}{fit}_g{groups}"
        formats[name] = state
        for tile in args.quant_tiles:
            dtypes = (torch.float32, torch.float16) if args.fp16_scores else (torch.float32,)
            for dtype in dtypes:
                scorer, workspace = state.prepare(lengths, scale, tile, score_dtype=dtype)
                output_name = f"{name}_b{tile}" + ("_f16" if dtype == torch.float16 else "")
                logical_key_bytes[output_name] = (
                    tokens * (8 * bits + 2 * groups) + (table[1].nbytes if table is not None else 0)
                )
                score_methods[output_name] = (
                    scorer, workspace, False, state.storage_bytes, initialization_us,
                )

    staged = {"projection_legacy": legacy_projection, "projection_q_only_native": q_only_native}
    if fused_projection_exact:
        staged["projection_q_only_fused"] = q_only_fused
    staged["score_exact_bf16"] = lambda: bf16(fast_q, fast_gate)
    staged["score_lossless12"] = lambda: packed_score(fast_q, fast_gate)
    append_slots = plan.bases + lengths
    static.append(replay.current_keys, append_slots)
    if not torch.equal(static.restore(append_slots).view(torch.int16), replay.current_keys.view(torch.int16)):
        raise RuntimeError("static exponent entries changed the real current K")
    staged["append_static12"] = lambda: static.append(replay.current_keys, append_slots)
    staged["append_lossless12"] = lambda: packed.append(replay.current_keys, lengths + 1)
    for name, state in formats.items():
        staged[f"append_{name}"] = lambda state=state: state.append(replay.current_keys, append_slots)

    for m in args.candidates:
        if m > min(replay.lengths) + 1:
            raise ValueError("candidate width exceeds the shortest request")
        selector_map = selectors(lengths, m, reference_width)
        # Common reference, including the known current logical ID. No method
        # projects it from the source hidden state or writes the canonical K.
        gold, ties = reference_topm(native_scores[:, :reference_width].contiguous(), lengths, m)
        target_ids = replay.target_ids.int()
        gold_recall = intersections(gold, target_ids).tolist()
        legacy = prepare_call(legacy_projection, native, selector_map["torch_sorted"], current)
        calls, rows = {"legacy_full_sorted": legacy}, {}
        for name, (score, workspace, exact, storage, init_us) in score_methods.items():
            actual_scores = score(fast_q, fast_gate).clone()
            baseline = exact
            # Ignore padding beyond the common score domain.
            score_equal = (
                torch.equal(actual_scores[:, :width], native_scores)
                if exact and actual_scores.size(1) >= width
                else (torch.equal(actual_scores, native_scores[:, :actual_scores.size(1)])
                      if exact else None)
            )
            for select_name, select in selector_map.items():
                if select_name == "vllm_cute" and actual_scores.dtype != torch.float32:
                    continue
                call_name = f"{name}:{select_name}"
                if exact and not score_equal:
                    rows[call_name] = {"failed": "different BF16 scores"}
                    continue
                call = prepare_call(q_only, score, select, current)
                try:
                    selected = call().clone()
                    torch.cuda.synchronize()
                except (RuntimeError, ValueError, AssertionError) as exc:
                    if select_name != "vllm_cute":
                        raise
                    rows[call_name] = {"failed": str(exc)}
                    continue
                sorted_ids = selected.sort(dim=1).values
                valid = ((sorted_ids >= 0) & (sorted_ids <= current)).all()
                unique = (sorted_ids.diff(dim=1) > 0).all()
                if not valid or not unique or not (selected[:, 0:1] == current).all():
                    rows[call_name] = {"failed": "invalid or duplicate candidate IDs"}
                    continue
                if exact and not torch.equal(sorted_ids, gold.sort(dim=1).values):
                    rows[call_name] = {"failed": "exact score but different Top-M set"}
                    continue
                if not exact:
                    threshold = actual_scores.topk(m - 1).values[:, -1:]
                    picked = actual_scores.gather(1, selected[:, 1:].long())
                    if (picked < threshold).any() or not torch.equal(
                        (actual_scores > threshold).sum(dim=1), (picked > threshold).sum(dim=1),
                    ):
                        rows[call_name] = {"failed": "selector disagrees with approximate scores"}
                        continue
                calls[call_name] = call
                rows[call_name] = dict(
                    exact=exact, baseline=baseline, score_equal=score_equal, index_key_bytes=storage,
                    score_dtype=str(actual_scores.dtype),
                    logical_key_bytes=logical_key_bytes[name],
                    logical_score_write_bytes=actual_scores.nbytes,
                    workspace_bytes_before_selector=workspace, initialize_wall_us=init_us,
                    known_cute_workspace_bytes=(
                        len(replay.lengths) * cute_workspace_bytes(actual_scores.size(1), m)
                        if select_name == "vllm_cute" else None
                    ),
                    predicted_set_intersection=intersections(selected, gold).tolist(),
                    true_top2048_intersection=intersections(selected, target_ids).tolist(),
                )
        if args.progressive:
            state = formats.get("int6_g1")
            if state is None:
                raise ValueError("--progressive needs --quant-formats to include 6x1")
            reference, _ = state.prepare(lengths, scale, block=128)
            for multiplier in args.progressive:
                mode = f"int6_progressive_{multiplier:g}:flashinfer"
                impl = prepare_progressive(state, q_only, lengths, scale, m, multiplier, torch.float16)
                selected = validate_progressive(impl, reference, fast_q, fast_gate, lengths, m)
                key_record = score_methods["int6_g1_b" + str(args.quant_tiles[0])]
                calls[mode] = impl.call
                logical_cols = triton.cdiv(plan.width, 128) * 128 + impl.shortlist
                rows[mode] = dict(
                    exact=False, baseline=False, score_equal=None,
                    index_key_bytes=state.storage_bytes, score_dtype="torch.float16",
                    logical_key_bytes=impl.logical_key_bytes,
                    logical_score_write_bytes=2 * logical_cols * len(replay.lengths),
                    workspace_bytes_before_selector=impl.graph_workspace_bytes,
                    known_cute_workspace_bytes=None, initialize_wall_us=key_record[4],
                    predicted_set_intersection=intersections(selected, gold).tolist(),
                    true_top2048_intersection=intersections(selected, target_ids).tolist(),
                    shortlist_width=impl.shortlist,
                )
        if args.temporal:
            state = formats.get("int6_g1")
            if previous_replay is None or state is None:
                raise ValueError("--temporal needs previous capture and --quant-formats 6x1")
            if [n + 1 for n in previous_replay.lengths] != replay.lengths:
                raise ValueError("temporal scoring needs consecutive captured states")
            prev_q, prev_gate = predictor_projection(
                indexer, previous_replay.hidden, previous_replay.positions,
            )()
            key_record = score_methods["int6_g1_b" + str(args.quant_tiles[0])]
            quantized_bytes = 8 * state.bits + 2 * state.groups
            for segment in args.temporal_segments:
                dense, _ = state.prepare(lengths, scale, block=segment)
                for fraction in args.temporal:
                    label = f"int6_temporal_{segment}_{fraction:g}:flashinfer"
                    impl = prepare_temporal(state, q_only, prev_q, prev_gate, lengths, scale,
                                            fraction, segment, m, torch.float16)
                    selected, kept = validate_temporal(impl, dense, fast_q, fast_gate, lengths, segment, m)
                    calls[label] = impl.partial
                    calls[f"temporal_refresh_{segment}"] = impl.refresh
                    rows[label] = dict(
                        exact=False, baseline=False, score_equal=None,
                        index_key_bytes=state.storage_bytes, score_dtype="torch.float16",
                        logical_key_bytes=quantized_bytes * sum(kept),
                        logical_score_write_bytes=2 * len(replay.lengths) * impl.shortlist_tokens,
                        workspace_bytes_before_selector=impl.workspace_bytes,
                        known_cute_workspace_bytes=None, initialize_wall_us=key_record[4],
                        predicted_set_intersection=intersections(selected, gold).tolist(),
                        true_top2048_intersection=intersections(selected, target_ids).tolist(),
                        kept_tokens_by_request=kept, shortlist_width=impl.shortlist_tokens,
                    )
        # Ensure an identical strongest denominator is available when comparing
        # methods based on a request-local contiguous slab and direct addressing.
        exact_names = [name for name, rec in rows.items() if rec.get("baseline")]
        same_layout = [name for name in exact_names if not name.startswith("paged:")]
        if not same_layout:
            raise RuntimeError("no validated BF16 implementation in the equal-layout denominator")
        timings = measure_graph(calls | staged, args.trials, args.iterations, args.seed, args.warmups)
        best = min(exact_names, key=lambda name: timings[name]["median_us"])
        matched = min(same_layout, key=lambda name: timings[name]["median_us"])
        result = dict(
            **identity, width=m, request_ids=replay.request_ids, history_lengths=replay.lengths,
            exact_topm_ties=ties.tolist(), exact_candidate_intersection=gold_recall,
            fastest_bf16=best, fastest_equal_layout_bf16=matched,
            static12_active_exception_rows=static.exception_count,
            fused_projection_exact=fused_projection_exact, fused_projection_selected=use_fused_projection,
            methods=rows, timings=timings,
        )
        if args.segment_audit:
            result["segment_audit"] = audit_segments(
                slots, fast_q, fast_gate, replay.lengths, native_scores, gold,
                target_ids, m, scale, args.segment_audit,
            )
        yield result


def enforce_exclusive_gpu():
    prop = torch.cuda.get_device_properties(0)
    raw_uuid = str(prop.uuid)
    gpu = raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"
    result = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip()
    others = sorted({int(line.strip()) for line in result.splitlines() if line.strip()} - {os.getpid()})
    if others:
        raise RuntimeError(f"A6000 timing requires exclusive compute use; other processes: {others}")


def metadata(args):
    sources = [*HERE.glob("*.py"),
               ROOT / "scripts/indexer-round2/slab_index.py", ROOT / "scripts/indexer-optimization/optimize.py"]
    books = [args.pair_pq / f"l{layer:02}.pt" for layer in args.layers] if args.pair_pq else []
    return dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
        flashinfer=__import__("flashinfer").__version__,
        gpu=str(torch.cuda.get_device_properties(0)),
        driver=subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
        ], text=True).splitlines()[0].strip(),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        codebook_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in books},
        scope="real previous-layer hidden state to M logical candidates; historical keys only; no prediction K or append",
        initialization="synchronized wall time, one untimed build and three timed repeats",
    )


def run(args):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("measure on a single visible SM86 GPU")
    enforce_exclusive_gpu()
    config = initialize()
    old = Trace(TRACE) if args.existing else None
    captured = CapturedInputs(args.capture or [])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata(args), default=str, indent=2) + "\n")
    with (args.output / "results.jsonl").open("x") as stream:
        for layer in args.layers:
            indexer = load_indexer(layer)
            codebook = None
            if args.pair_pq is not None:
                trained = torch.load(args.pair_pq / f"l{layer:02}.pt", weights_only=True)
                codebook = trained["mapping"].cuda(), trained["packed"].cuda()
            for bucket, batch, step in itertools.product(args.length, args.batch, args.steps):
                identity = dict(bucket=bucket, batch=batch, layer=layer, step=step, split=args.split)
                if old is not None:
                    if layer != 1:
                        raise ValueError("legacy real replay is only adjacent at layer 0 to 1")
                    replay = existing_inputs(old, bucket, batch, step, args.offset, indexer, config)
                else:
                    replay = captured.load(bucket, args.split, batch, layer, step, args.offset)
                previous = None
                if args.temporal:
                    if old is not None or step == 0:
                        raise ValueError("temporal benchmark needs a capture and a positive step")
                    previous = captured.load(bucket, args.split, batch, layer, step - 1, args.offset)
                enforce_exclusive_gpu()
                for row in benchmark(replay, indexer, args, identity, previous, codebook):
                    enforce_exclusive_gpu()
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    best = row["fastest_equal_layout_bf16"]
                    exact_us = row["timings"][best]["median_us"]
                    print(identity, "M", row["width"], "equal-layout", best, round(exact_us, 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    traced = parser.add_mutually_exclusive_group(required=True)
    traced.add_argument("--existing", action="store_true")
    traced.add_argument("--capture", type=Path, nargs="+")
    parser.add_argument("--length", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--layers", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, nargs="+", default=[2])
    parser.add_argument("--split", choices=["calibration", "test"], default="calibration")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--quant-formats", nargs="+", default=["4x1", "4x2", "5x1", "6x1", "8x1", "8x2"])
    parser.add_argument("--quant-tiles", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--exact-tuning", action="store_true")
    parser.add_argument("--exact-tiles", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--fused-query", action="store_true")
    parser.add_argument("--fp16-scores", action="store_true")
    parser.add_argument("--fp4", action="store_true")
    parser.add_argument("--pair-pq", type=Path)
    parser.add_argument("--direct-paged-build", action="store_true")
    parser.add_argument("--fit-int4", type=int, choices=(1, 2, 4), nargs="+")
    parser.add_argument("--temporal", type=float, nargs="+")
    parser.add_argument("--temporal-segments", type=int, choices=(32,64,128), nargs="+", default=[32,64])
    parser.add_argument("--progressive", type=float, nargs="+")
    parser.add_argument("--segment-audit", type=int, nargs="+")
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        run(args)
