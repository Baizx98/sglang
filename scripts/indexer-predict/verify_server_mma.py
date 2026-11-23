"""Check the pluggable online predictor and append path with captured real states."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from benchmark import CapturedInputs, initialize, load_indexer, page_layout
from quality_cases import layer_policy


def audit(args, output):
    policy = json.loads(args.policy.read_text())
    cases = {}
    for route in policy["routes"]:
        plan = layer_policy(route, args.codebooks)
        if bits := plan.get("native_mma"):
            cases.setdefault(bits, (route, plan))
    if set(cases) != {4, 8}:
        raise ValueError("verification requires one native INT4 and one native INT8 route")
    initialize()
    inputs = CapturedInputs(args.capture)
    with tempfile.TemporaryDirectory(prefix="verify-warp-predictor-") as directory:
        config = Path(directory) / "policy.json"
        config.write_text(json.dumps(dict(append_reserve=64, candidate_width=3072)))
        os.environ["KEYE_PREDICT_QUALITY_POLICY"] = str(config)
        os.environ["KEYE_LOOKAHEAD_TRACE_DIR"] = directory
        from lookahead_quality_server import CompactState

        for bits, (route, mode) in sorted(cases.items()):
            layer, bucket, m = route["layer"], route["length_bucket"], route["M"]
            replay = inputs.load(bucket, "test", 1, layer, 2, 0)
            pages, lengths = page_layout(replay.lengths, max(replay.lengths) + 1)
            indexer = load_indexer(layer)
            base = dict(mode, native_mma=0, width=m)
            active = dict(mode, width=m)
            old = CompactState(indexer, replay.request_ids[0], layer, replay.keys, pages, lengths, base)
            new = CompactState(indexer, replay.request_ids[0], layer, replay.keys, pages, lengths, active)
            if new.native_mma != bits:
                raise RuntimeError("the test must exercise both native formats on SM86")
            for updated in (False, True):
                reference = old.scan(*old.project(replay.hidden, replay.positions))
                prediction = new.predict(replay.hidden, replay.positions)
                gates = new.project.encode_into(
                    replay.hidden, replay.positions, new.query_codes, new.key.query_scales,
                    1, packed_nibbles=bits == 4,
                )
                scores = new.scan(gates)
                as_int = torch.int16 if scores.dtype == torch.float16 else torch.int32
                if not torch.equal(scores.view(as_int), reference.view(as_int)):
                    raise RuntimeError("the serving scorer changed the original score bits")
                if prediction.shape != (1, m) or prediction[0, 0] != old.lengths[0]:
                    raise RuntimeError("the serving route lost the reserved current position")
                history = prediction[:, 1:].long()
                if (history < 0).any() or (history >= old.lengths[:, None]).any() or history.unique().numel() != m - 1:
                    raise RuntimeError("the serving route returned invalid historical candidates")
                selected = scores.gather(1, history).sort(1, descending=True).values
                if not torch.equal(selected, reference.topk(m - 1).values):
                    raise RuntimeError("the serving route lost a strictly higher historical score")
                record = dict(bits=bits, layer=layer, length_bucket=bucket, M=m, request=replay.request_ids[0],
                              history_tokens=int(old.lengths[0]), appended_real_key=updated,
                              bitwise_score_match=True, topm_score_match=True)
                output.write(json.dumps(record) + "\n")
                output.flush()
                if not updated:
                    for state in (old, new):
                        state.append(replay.current_keys, replay.lengths[0] + 1)
                        state.lengths.add_(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode(), args.output.open("x") as out:
        audit(args, out)
