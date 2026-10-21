"""Merge independent real C1 fixtures into actual C4 layer-replay inputs."""

import argparse
import json
from pathlib import Path

import torch

from horizonkv_single_runtime import save


def merge(root):
    root = Path(root)
    done = root / "merged-c4.json"
    if done.exists():
        return json.loads(done.read_text())
    groups = json.loads((root / "long-input-manifest.json").read_text())["groups"]
    fixtures = root / "fixtures"
    total = 0
    for rank in (0, 1):
        path = fixtures / f"manifest_rank{rank:02d}.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        additions = []
        for group in groups:
            if len(group["waves"]) != 4:
                continue
            for layer in (1, 12, 24, 36, 47):
                parts = []
                for wave in group["waves"]:
                    candidates = [r for r in records if r["layer"] == layer and len(r["request_ids"]) == 1
                                  and r["request_ids"][0].startswith(f"replay__wave{wave:03d}__")]
                    parts.append({r["steps"][0]: r for r in candidates})
                common = sorted(set.intersection(*(set(p) for p in parts)))
                assert 0 in common, (group["label"], layer, rank)
                for step in common:
                    frames = [torch.load(fixtures / p[step]["file"], weights_only=False) for p in parts]
                    result = {}
                    for key, value in frames[0].items():
                        vals = [f[key] for f in frames]
                        if isinstance(value, torch.Tensor):
                            result[key] = torch.cat(vals, dim=-1 if key == "positions" else 0)
                        elif isinstance(value, list):
                            result[key] = sum(vals, [])
                        else:
                            assert all(v == value for v in vals), key
                            result[key] = value
                    assert len(set(result["request_ids"])) == 4 and len(result["deltas"]) == 4
                    name = f"{result['request_ids'][0]}_layer{layer:02d}_rank{rank:02d}_mergedC4_step{step:03d}.pt"
                    torch.save(result, fixtures / name)
                    additions.append(dict(file=name, request_ids=result["request_ids"], steps=result["steps"], layer=layer))
                    total += 1
        with path.open("a") as stream:
            for record in additions:
                stream.write(json.dumps(record) + "\n")
    value = dict(frames=total, source="four distinct C1 requests per length/task group",
                 scope="C4 layer compute only; no full-model C4 serving/throughput claim")
    save(done, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(merge(parser.parse_args().run_dir))
