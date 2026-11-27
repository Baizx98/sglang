"""Analyze real Top-k and predicted Top-M locality and fixed-capacity residency."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from horizonkv_belady import compile_belady, run
from horizonkv_capacity_replay import compile_replay

CAPACITIES = (2048, 4096, 8192)
LAGS = (1, 4, 16)
EMPTY = np.empty(0, dtype=np.int32)
POLICIES = {}


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prefix_length(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def read_agent(agent):
    root = agent / "agent-capture"
    groups = defaultdict(list)
    for name in ("swe", "bfcl"):
        for row in jsonl(agent / "cohort-v2" / f"{name}-requests.jsonl"):
            audit = json.loads((root / "turns" / f"{row['rid']}.json").read_text())
            assert (row["rid"], row["turn"], row["dataset"], row["session_id"]) == (
                audit["rid"], audit["turn"], audit["dataset"], audit["session_id"])
            chunks = defaultdict(list)
            for entry in audit["chunks"]:
                chunks[entry["layer"]].append(root / "events" / entry["file"])
            row["chunks"] = chunks
            row["steps"] = audit["decode_steps"]
            groups[(row["dataset"], row["session_id"])].append(row)
    jobs = []
    for (dataset, session), rows in groups.items():
        rows.sort(key=lambda r: r["turn"])
        assert [r["turn"] for r in rows] == list(range(len(rows)))
        prior = []
        for row in rows:
            ids = row.pop("input_ids")
            assert row["shared_prefix_tokens"] == prefix_length(prior, ids)
            assert row["prompt_tokens"] == len(ids)
            prior = ids
        jobs.append((dataset, session, 0, rows))
    return jobs


def read_ruler(root):
    manifest = {}
    for chunk in jsonl(root / "events" / "manifest.jsonl"):
        key = chunk["request_id"], chunk["target_layer_id"]
        if key in manifest:
            raise ValueError(f"Duplicate RULER trace {key}")
        manifest[key] = root / "events" / chunk["file"]
    jobs, seen = [], set()
    for row in jsonl(root / "requests.jsonl"):
        if row["dataset"] != "RULER":
            continue
        assert row["rid"] not in seen and len(row["input_ids"]) == row["prompt_tokens"]
        del row["input_ids"]
        seen.add(row["rid"])
        row["chunks"] = {layer: manifest[row["rid"], layer] for layer in range(1, 48)}
        row.update(steps=64, turn=0, shared_prefix_tokens=0, segments=[])
        jobs.append(("RULER", row["rid"], row["length_bucket_upper"], [row]))
    assert len(jobs) == 156
    return jobs


def arrays(row, layer, ruler):
    source = row["chunks"][layer]
    current = row["prompt_tokens"] + np.arange(row["steps"], dtype=np.int32)
    if ruler:
        from analyze_horizonkv_predictive import load_chunk

        chunk = load_chunk(source)
        assert chunk["exact_main"] and not chunk["prediction_uses_current_key"]
        assert chunk["request_id"] == row["rid"] and chunk["target_layer_id"] == layer
        steps = chunk["rows"]
        assert [r["step"] for r in steps] == list(range(len(current)))
        assert np.array_equal([r["context_len"] - 1 for r in steps], current)
        true = np.stack([np.asarray(r["exact_indices"], dtype=np.int32) for r in steps])
        pred = np.stack([np.asarray(r["candidates"][:3072], dtype=np.int32) for r in steps])
    else:
        truth, predictions, contexts, steps = [], [], [], []
        for path in source:
            with np.load(path, allow_pickle=False) as chunk:
                truth.append(chunk["true_topk"])
                if layer:
                    predictions.append(chunk["predicted_topm"])
                contexts.extend(chunk["context_len"])
                steps.extend(chunk["step"])
        assert steps == list(range(len(current))) and np.array_equal(contexts, current + 1)
        true = np.concatenate(truth)
        pred = np.concatenate(predictions) if layer else None
    assert true.shape == (len(current), 2048)
    if pred is not None:
        assert pred.shape == (len(current), 3072) and np.array_equal(pred[:, 0], current)
        pred = pred[:, 1:].copy()
    assert np.all(true <= current[:, None])
    true[true == current[:, None]] = -1
    if pred is not None:
        assert np.all(pred < current[:, None])
    return true, pred


def labels(row):
    n = row["prompt_tokens"]
    regions = row["segments"] or [dict(role="prompt", start=0, end=n, source_turn=None)]
    groups = defaultdict(list)
    for s in regions:
        age = "global" if s["source_turn"] is None else str(min(4, row["turn"] - s["source_turn"]))
        groups[(s["role"], age)].append(slice(s["start"], s["end"]))
    groups["generated_history", "0"].append(slice(n, n + row["steps"] - 1))
    return groups


def indexed_mask(values, universe):
    bits = np.zeros((len(values), universe + 1), dtype=bool)
    bits[np.arange(len(values))[:, None], values] = True
    bits[:, -1] = False
    return bits


def overlap(bits, target):
    return int(bits[np.arange(len(target))[:, None], target].sum())


def add(counter, tag, numer, denom):
    counter[tag, "numerator"] += int(numer)
    counter[tag, "denominator"] += int(denom)


def activation(row, true, pred, previous_layer, past_counts, counters, regions):
    n, steps = row["prompt_tokens"], row["steps"]
    universe = n + steps - 1
    available = np.concatenate((np.full(n, steps), np.arange(steps - 1, 0, -1)))
    demand_ok = true >= 0
    counts = np.bincount(true[demand_ok], minlength=universe)
    truth = indexed_mask(true, universe)
    for lag in LAGS:
        if lag < steps:
            add(counters, f"topk_step_lag{lag}", overlap(truth[:-lag], true[lag:]),
                demand_ok[lag:].sum())
    previous_true, previous_pred = previous_layer
    if previous_true is not None:
        add(counters, "topk_adjacent_layer", overlap(indexed_mask(previous_true, universe), true),
            demand_ok.sum())
    frequencies = {"top_k": counts}
    if pred is not None:
        pred_ok = pred >= 0
        predicted = indexed_mask(pred, universe)
        frequencies["top_m"] = np.bincount(pred[pred_ok], minlength=universe)
        both = truth[np.arange(steps)[:, None], pred]
        add(counters, "topm_coverage", both.sum(), demand_ok.sum())
        add(counters, "topm_precision", both.sum(), pred_ok.sum())
        if steps > 8:
            # A token is outside the last eight native selections iff none of
            # the aligned predicted positions were accessed in those steps.
            aged = pred[8:]
            index = np.arange(steps - 8)[:, None]
            recent = np.zeros(aged.shape, dtype=bool)
            for lag in range(1, 9):
                recent |= truth[index + 8 - lag, aged]
            new = (~recent) & pred_ok[8:]
            add(counters, "topm_new_to_history8", (both[8:] & new).sum(), new.sum())
        for lo, hi in ((0, 512), (512, 1024), (1024, 2048), (2048, 3071)):
            add(counters, f"topm_precision_rank{lo + 1}_{hi}", both[:, lo:hi].sum(),
                pred_ok[:, lo:hi].sum())
        for lag in LAGS:
            if lag < steps:
                add(counters, f"topm_step_lag{lag}", overlap(predicted[:-lag], pred[lag:]),
                    pred_ok[lag:].sum())
                add(counters, f"topm_to_future_topk_lag{lag}", overlap(truth[lag:], pred[:-lag]),
                    pred_ok[:-lag].sum())
        if previous_pred is not None:
            add(counters, "topm_adjacent_layer",
                overlap(indexed_mask(previous_pred, universe), pred), pred_ok.sum())
    for stream, freq in frequencies.items():
        add(counters, f"{stream}_never_prompt", (freq[:n] == 0).sum(), n)
        add(counters, f"{stream}_hot_half_prompt", (2 * freq[:n] >= steps).sum(), n)
        add(counters, f"{stream}_eligible_rate", freq.sum(), available.sum())
        for (region, age), spans in labels(row).items():
            key = stream, region, age
            for span in spans:
                hit, exposure = freq[span], available[span]
                add(regions[key], "frequency", hit.sum(), exposure.sum())
                add(regions[key], "ever", (hit > 0).sum(), hit.size)
    prefix = row["shared_prefix_tokens"]
    if past_counts is not None and prefix:
        previous = past_counts["top_k"][:prefix] > 0
        future = counts[:prefix] > 0
        add(counters, "turn_shared_topk_repeat", (previous & future).sum(), previous.sum())
        add(counters, "turn_shared_topk_previously_inactive", ((~previous) & future).sum(),
            (~previous).sum())
        for (region, age), spans in labels(row).items():
            for span in spans:
                shared = slice(min(span.start, prefix), min(span.stop, prefix))
                a, b = previous[shared], future[shared]
                key = "top_k", region, age
                add(regions[key], "turn_reactivate_previously_active", (a & b).sum(), a.sum())
                add(regions[key], "turn_reactivate_previously_inactive", ((~a) & b).sum(), (~a).sum())
        if "top_m" in past_counts:
            candidate = past_counts["top_m"][:prefix] > 0
            add(counters, "turn_shared_topm_only_to_topk",
                (candidate & ~previous & future).sum(), (candidate & ~previous).sum())
        first = true[:4]
        add(counters, "turn_early_topk_from_shared_prefix", ((first >= 0) & (first < prefix)).sum(),
            (first >= 0).sum())
    return frequencies


def prepare_block(values):
    valid = values >= 0
    sorted_ids = np.sort(values, axis=1)
    if np.any((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)):
        raise ValueError("Duplicate token in a simultaneous activation set")
    return valid.sum(axis=1, dtype=np.int64), sorted_ids[sorted_ids >= 0]


def replay_inputs(rows, blocks, mode):
    """Keep logical prefix identity and invalidate the expired suffix at each boundary."""
    previous = EMPTY
    ids, expired = [], []
    next_id = 0
    for row, (counts, ordered) in zip(rows, blocks, strict=True):
        prefix = row["shared_prefix_tokens"] if mode == "carry_prefix" else 0
        length = row["prompt_tokens"] + row["steps"] - 1
        mapping = np.concatenate((previous[:prefix], np.arange(
            next_id, next_id + length - prefix, dtype=np.int32)))
        ids.append(mapping[ordered])
        expired.append(previous[prefix:])
        next_id += length - prefix
        previous = mapping
    offsets = np.concatenate(([0], np.cumsum(np.concatenate([b[0] for b in blocks]))))
    flat = np.concatenate(ids)
    writes = np.concatenate(expired)
    write_counts = np.zeros(len(offsets) - 1, dtype=np.int64)
    boundaries = np.cumsum([0] + [r["steps"] for r in rows[:-1]])
    write_counts[boundaries] = [len(part) for part in expired]
    write_offsets = np.concatenate(([0], np.cumsum(write_counts)))
    return max(next_id, 1), [offsets, flat, offsets, flat, write_offsets, writes]


def evaluate(rows, values):
    blocks = [prepare_block(value) for value in values]
    demand = np.concatenate([row[0] for row in blocks])
    assert np.all(demand > 0)
    later_first4 = np.concatenate([np.arange(r["steps"]) < (4 if r["turn"] else 0) for r in rows])
    after16 = np.concatenate([np.arange(r["steps"]) >= 16 for r in rows])
    views = {"all": slice(None), "after16": after16, "later_turn_first4": later_first4}
    modes = ("reset_turn", "carry_prefix") if len(rows) > 1 else ("reset_request",)
    output = []
    for mode in modes:
        packed = replay_inputs(rows, blocks, mode)
        offsets = packed[1][0]
        previous_oracle = 0
        for capacity in CAPACITIES:
            totals = {}
            for policy, fn in POLICIES.items():
                hits = run(fn, capacity, packed)
                per_step = np.add.reduceat(hits, offsets[:-1], dtype=np.int64)
                assert (per_step <= np.minimum(capacity, demand)).all()
                totals[policy] = int(per_step.sum())
                for view, selection in views.items():
                    total = int(demand[selection].sum())
                    if total:
                        output.append(dict(mode=mode, policy=policy, capacity=capacity, view=view,
                                           hits=int(per_step[selection].sum()), demand=total))
            assert totals["oracle"] >= max(totals["lru"], previous_oracle)
            previous_oracle = totals["oracle"]
    return output


def sequence(job):
    dataset, session, length, rows = job
    ruler = dataset == "RULER"
    previous_layer = [(None, None)] * len(rows)
    result = dict(dataset=dataset, session=session, length=length, turns=len(rows),
                  steps=sum(r["steps"] for r in rows), activation=[], cache=[], regions=[])
    for layer in range(int(ruler), 48):
        metrics, regions = Counter(), defaultdict(Counter)
        prior_turn = None
        demand_turns, predicted_turns = [], []
        for i, row in enumerate(rows):
            true, pred = arrays(row, layer, ruler)
            prior_turn = activation(row, true, pred, previous_layer[i], prior_turn, metrics, regions)
            previous_layer[i] = true, pred
            demand_turns.append(true)
            if pred is not None:
                predicted_turns.append(pred)
        for stream, values in (("top_k", demand_turns), ("top_m", predicted_turns)):
            if values:
                result["cache"].extend(dict(layer=layer, stream=stream, **r)
                                       for r in evaluate(rows, values))
        result["activation"].extend(dict(layer=layer, **r) for r in metric_rows(metrics))
        for (stream, role, age), metrics in regions.items():
            result["regions"].extend(dict(layer=layer, stream=stream, role=role, age=age, **r)
                                     for r in metric_rows(metrics))
    return result


def metric_rows(counter):
    for metric, key in counter:
        if key == "denominator" and counter[metric, key]:
            yield dict(metric=metric, numerator=counter[metric, "numerator"],
                       denominator=counter[metric, "denominator"])


def export(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, results):
    for kind, fields, counts in (
        ("activation", ("metric",), ("numerator", "denominator")),
        ("regions", ("stream", "role", "age", "metric"), ("numerator", "denominator")),
        ("cache", ("stream", "mode", "policy", "capacity", "view"), ("hits", "demand")),
    ):
        aggregated = defaultdict(Counter)
        individual = []
        for item in results:
            base = {k: item[k] for k in ("dataset", "length")}
            for r in item[kind]:
                data = {**base, **r}
                if kind != "regions":
                    individual.append(dict(data, session=item["session"]))
                key = tuple(data[k] for k in (*base, "layer", *fields))
                for c in counts:
                    aggregated[key][c] += r[c]
        if individual:
            export(root / f"{kind}-by-session.csv", individual)
        export(root / f"{kind}-by-layer.csv", [
            dict(zip(("dataset", "length", "layer", *fields), key), **val)
            for key, val in sorted(aggregated.items())
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sessions = root / "sessions"
    sessions.mkdir(exist_ok=True)
    POLICIES.update(lru=compile_replay(root), oracle=compile_belady(root))
    jobs = read_agent(args.agent.resolve()) + read_ruler(args.ruler.resolve())
    if args.smoke:
        jobs = [next(j for j in jobs if j[0] == name)
                for name in ("BFCL", "RULER", "SWE-bench Verified")]
    save(root / "provenance.json", dict(agent=str(args.agent.resolve()), ruler=str(args.ruler.resolve()),
         code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         capacities=CAPACITIES, bytes_per_row=2048, deterministic=True, smoke=args.smoke,
         agent_status_sha256=digest(args.agent / "agent-capture/state.json"),
         ruler_manifest_sha256=digest(args.ruler / "events/manifest.jsonl")))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for job in jobs:
            path = sessions / f"{job[0]}__{job[1]}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
            else:
                pending[pool.submit(sequence, job)] = path
        for future in as_completed(pending):
            value = future.result()
            save(pending[future], value)
            results.append(value)
            print(f"{len(results)}/{len(jobs)} {value['dataset']} {value['session']}", flush=True)
    summarize(root, results)
    save(root / "state.json", dict(status="completed", sessions=len(results),
         turns=sum(x["turns"] for x in results), steps=sum(x["steps"] for x in results)))


if __name__ == "__main__":
    main()
