"""Independent tiny-history fixture catches padded-K denominators and audits."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from horizonkv_capacity_replay import access_groups


def main():
    with tempfile.TemporaryDirectory(prefix="hkv-analysis-") as tmp:
        root = Path(tmp)
        directory = root / "trace-p140000"
        trace = directory / "trace"
        trace.mkdir(parents=True)
        rids = [f"capacity__fixture__wave000__case{i}" for i in range(2)]
        requests = [{"rid": rid, "case_id": rid, "prompt_tokens": 3, "response": {"output_ids": [1,2,3,4][:4-i]},
                     "concurrency": 2, "cell": ["fixture", "tiny", 8192], "storage": "hbm"} for i,rid in enumerate(rids)]
        (directory / "requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in requests))
        (directory / "completed.json").write_text("{}\n")
        (root / "plan.json").write_text(json.dumps({"capacities": [0,2,8,16], "source_commit": "fixture", "gpu_uuids": []}))
        for rank in (0, 1):
            prefix = trace / f"rank_{rank:02d}"
            with prefix.with_suffix(".bin").open("wb") as data, prefix.with_suffix(".jsonl").open("w") as index:
                index.write(json.dumps({"kind": "reset", "rank": rank, "offset_bytes": 0,
                    "resident_after_reset": 0, "wave": "capacity__fixture__wave000"})+"\n")
                for step in range(3):
                    batch = 1 if step == 0 else 2
                    contexts = [4+step] + ([3+step] if step else [])
                    logical = np.full((batch,2048), -1, dtype="<i4")
                    physical = logical.copy()
                    for row,context in enumerate(contexts):
                        logical[row,:context] = np.arange(context)
                        physical[row,:context] = np.arange(context) + row*32
                    writes = np.array([context-1+32*i for i,context in enumerate(contexts)], dtype="<i4")
                    for layer in range(48):
                        record = {"kind": "access", "rank": rank, "offset_bytes": data.tell(),
                            "wave": "capacity__fixture__wave000", "layer_id": layer,
                            "request_ids": rids[:batch], "decode_steps": [step]+([step-1] if step else []), "context_tokens": contexts,
                            "shape": [batch,2048], "write_count": batch, "online_hits": None}
                        for value in (logical,physical,writes):
                            data.write(value.tobytes())
                        record["end_bytes"] = data.tell()
                        index.write(json.dumps(record)+"\n")
        assert len(list(access_groups(directory, requests, allow_padding=True))) == 96
        try:
            list(access_groups(directory, requests))
        except AssertionError:
            pass
        else:
            raise AssertionError("Legacy strict reader must reject padded fixtures")
        subprocess.run([sys.executable, str(Path(__file__).with_name("horizonkv_a6000_analyze.py")),
                        "--run-dir", str(root)], check=True, stdout=subprocess.DEVNULL)
        import csv
        with (root / "processed/capacity-layer-counts.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            expected_selected = 24 if row["view"] == "all_returned" else 20
            assert int(row["selected_tokens"]) == expected_selected, row
            if int(row["layer"]):
                assert float(row["topk_overlap"]) == 1
                assert float(row["miss_coverage"]) == 1
            if row["capacity_tokens"] == "0":
                assert float(row["hit_ratio"]) == 0
            elif row["capacity_tokens"] == "16":
                assert float(row["hit_ratio"]) == 13/expected_selected
        assert not json.loads((root / "processed/provenance.json").read_text())["paper_ready"]
        print("PASS: 2 ranks x 48 layers; padded Top-K counts exclude -1; initial partial batch warms cache; analytic LRU/Belady endpoints; Overlap/MissCoverage; legacy strict reader")


if __name__ == "__main__":
    main()
