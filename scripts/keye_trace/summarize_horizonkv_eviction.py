"""Archive frozen eviction choices, traffic accounting, and matched turn ablations."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COUNTS = ("requests", "demand_loads", "prefetch_loads", "h2d_bytes", "repeated_h2d_bytes",
          "reload_after_eviction", "displaced_demand", "rescued_demand", "prefetched_consumed",
          "unused_evicted", "unused_at_end")
REAL_ONLY = ("lru", "lru-touch", "lru2", "lfu")
STEP_PRED = ("lru-pred", "lfu-pred")


def put(table, row, counts):
    policy = row["policy"]
    stat = table.get(policy)
    if stat is None:
        table[policy] = counts.copy()
    else:
        stat += counts


def best(policies, *families):
    return min((p for p in policies if p.split(":")[0] in families),
               key=lambda p: (policies[p][3], p))


def save_csv(path, data):
    with path.open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def inspect(run):
    counts, layers, tests = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    waves = defaultdict(dict)
    identities = defaultdict(set)
    with (run / "cells.csv").open() as stream:
        for row in csv.DictReader(stream):
            group = tuple(row[k] for k in ("batch", "capacity_over_batch_k", "prefetch_per_request"))
            split, layer, bucket = (row[k] for k in ("split", "layer", "bucket"))
            values = np.asarray([int(row[k]) for k in COUNTS], dtype=np.int64)
            put(counts[(split, *group)], row, values)
            put(layers[(split, *group, layer)], row, values)
            identities[(split, group[0])].update(row["requests_in_group"].split("|"))
            if split == "test":
                put(tests[(*group, layer, bucket)], row, values)
                put(waves[(*group, row["requests_in_group"])], row, values)
    assert all(not identities[("test", batch)].intersection(identities[("calibration", batch)])
               for _, batch in identities)
    return counts, layers, tests, waves, identities


def aggregate(run, rng):
    pools, layers, tests, waves, identities = inspect(run)
    out, detail = [], []
    for (split, *group), candidate in sorted(pools.items()):
        if split != "test" or ("calibration", *group) not in pools:
            continue
        group = tuple(group)
        calibration = pools[("calibration", *group)]
        baseline = best(calibration, *REAL_ONLY)
        combined = best(calibration, *STEP_PRED)
        header = dict(run=run.name, batch=group[0], capacity_over_batch_k=group[1],
                      prefetch_per_request=group[2], baseline=baseline,
                      heldout_requests=len(identities[("test", group[0])]))
        selected = [("baseline", baseline), ("lru", "lru:1:0"), ("lfu", best(calibration, "lfu")),
                    ("prediction", "pred:1:0"), ("step+prediction", combined)]
        if any("-issued:" in policy for policy in calibration):
            selected += [("issued-only", best(calibration, "lru-issued", "lfu-issued"))]
        if "turn:1:0" in calibration:
            selected += [(family, best(calibration, family)) for family in (
                "turn", "role", "lfu-turn", "lfu-role", "lfu-pred-turn", "lfu-pred-role",
            )]
        wave_values = [v for key, v in waves.items() if key[:3] == group]
        boot = rng.integers(0, len(wave_values), size=(2000, len(wave_values)))
        ref = np.asarray([v[baseline][3] for v in wave_values])[boot].sum(1)
        for role, policy in selected:
            selected_row = candidate[policy]
            matched = ""
            marginal = ""
            if role in ("lfu-pred-turn", "lfu-pred-role"):
                _, half, hint, _ = policy.split(":")
                matched = f"lfu-pred:{half}:{hint}"
                marginal = 1 - selected_row[3] / candidate[matched][3]
            measurements = np.asarray([v[policy][3] for v in wave_values])[boot].sum(1)
            lo, hi = np.quantile(1 - measurements / ref, [0.025, 0.975])
            out.append(dict(header, role=role, policy=policy, matched_no_prior=matched,
                            prior_h2d_saved_fraction=marginal, wave_groups=len(wave_values),
                            h2d_saved_fraction=1 - selected_row[3] / candidate[baseline][3],
                            bootstrap_low=lo, bootstrap_high=hi, **dict(zip(COUNTS, selected_row, strict=True))))
        for (*setting, layer, bucket), values in sorted(tests.items()):
            if tuple(setting) != group:
                continue
            prior_layer = layers.get(("calibration", *group, layer), calibration)
            chosen = best(prior_layer, *REAL_ONLY, *STEP_PRED)
            layer_baseline = best(prior_layer, *REAL_ONLY)
            for role, policy in (("baseline", baseline), ("step+prediction", combined),
                                 ("layer-calibrated-baseline", layer_baseline), ("layer-calibrated", chosen)):
                row = values[policy]
                detail.append(dict(header, layer=layer, bucket=bucket, role=role, policy=policy,
                                   h2d_saved_fraction=1 - row[3]/values[baseline][3],
                                   **dict(zip(COUNTS, row, strict=True))))
    return out, detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261126)
    overview, detail, source = [], [], {}
    for run in args.runs:
        rows, subgroups = aggregate(run, rng)
        overview.extend(rows)
        detail.extend(subgroups)
        for file in (run/"metadata.json", run/"cells.csv"):
            with file.open("rb") as stream:
                source[str(file)] = hashlib.file_digest(stream, "sha256").hexdigest()
    save_csv(args.output/"replay-overview.csv", overview)
    save_csv(args.output/"replay-by-layer-length.csv", detail)
    (args.output/"replay-input-sha256.json").write_text(json.dumps(source, indent=2) + "\n")
