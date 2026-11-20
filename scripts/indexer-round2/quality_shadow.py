"""Compare exact-trajectory overlap on the decode rows used for candidate reuse."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    ids = pa.array([row["rid"] for row in requests])
    data = pq.read_table(args.observations,
                         columns=["request_id", "layer", "step", "method", "recall"],
                         filters=[("request_id", "in", ids.to_pylist())])

    def odd(column):
        values = data[column].to_numpy()
        return pa.array(values % 2 == 1)

    stats = []
    for method, mask in [("previous_layer", odd("layer")), ("previous_step", odd("step"))]:
        rows = data.filter(pc.and_(mask, pc.equal(data["method"], method)))
        for request in requests:
            sample = rows.filter(pc.equal(rows["request_id"], request["rid"]))["recall"]
            if not len(sample):
                raise ValueError(f"missing rows for {method} {request['rid']}")
            stats.append({
                "request_id": request["rid"], "task": request["task"],
                "length_bucket": request["length_bucket_upper"], "method": method,
                "observations": len(sample), "recall_mean": pc.mean(sample).as_py(),
                "recall_min": pc.min(sample).as_py(),
                "recall_p05": pc.quantile(sample, q=0.05)[0].as_py(),
            })
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(stats[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stats)


if __name__ == "__main__":
    main()
