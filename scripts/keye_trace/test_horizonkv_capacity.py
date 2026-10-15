"""Check compiled replay against a simple independent reference and real GPU KV."""

import argparse
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from horizonkv_capacity_replay import compile_replay, replay


def reference(capacity, slots, writes):
    resident, result = OrderedDict(), []
    for selected, written in zip(slots, writes):
        for token in written:
            resident.pop(int(token), None)
        hits = set(resident)
        result.append(
            np.array(
                [int(x) in hits if x >= 0 else False for x in selected.flat],
                dtype=np.uint8,
            ).reshape(selected.shape)
        )
        access = sorted(set(map(int, selected[selected >= 0])))
        for token in access:
            if token in resident:
                resident.move_to_end(token)
        if capacity:
            for token in access:
                if token in hits:
                    continue
                if len(resident) == capacity:
                    resident.popitem(last=False)
                resident[token] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(20261014)
    slots = [rng.integers(-1, 32, size=(4, 7), dtype=np.int32) for _ in range(64)]
    writes = [rng.integers(0, 32, size=4, dtype=np.int32) for _ in slots]
    with tempfile.TemporaryDirectory(prefix="horizonkv-capacity-test-") as tmp:
        function = compile_replay(tmp)
        for capacity in [0, 1, 2, 7, 16, 32, 64]:
            actual = replay(function, capacity, slots, writes)
            expected = reference(capacity, slots, writes)
            assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
        if args.gpu:
            os.environ["KEYE_SM80_DSA"] = "1"
            os.environ["KEYE_HORIZON_CAPACITY_DIR"] = tmp
            import torch
            from sglang.srt.layers.attention.keye_topk import capacity_trace
            from sglang.srt.layers.attention.keye_topk.horizonkv_probe import HostLayer

            k = torch.randn(32, 2, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            for capacity in [0, 2, 16]:
                host = HostLayer(32, 2, 128, torch.bfloat16, "cuda", capacity)
                host.write(torch.arange(32), k, v)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: True),
                    rids=[f"capacity__test__wave000__case{i}" for i in range(4)],
                    _capacity_online_hits={},
                )
                expected = replay(function, capacity, slots, writes)
                for selected, written, mask in zip(slots, writes, expected):
                    host.write(torch.from_numpy(written).long(), k[written], v[written])
                    sk, sv, _ = host.stage(
                        torch.from_numpy(selected), batch, 0, torch.from_numpy(selected)
                    )
                    assert batch._capacity_online_hits[0] == mask.sum(axis=1).tolist()
                    valid = selected.reshape(-1) >= 0
                    torch.testing.assert_close(
                        sk[torch.from_numpy(valid).cuda()],
                        k[selected.reshape(-1)[valid]],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        sv[torch.from_numpy(valid).cuda()],
                        v[selected.reshape(-1)[valid]],
                        rtol=0,
                        atol=0,
                    )
                # A new wave must explicitly discard the old resident mappings.
                batch.seq_lens_cpu = torch.tensor([32] * 4)
                batch.rids = [
                    f"capacity__test__wave{capacity:03d}__case{i}" for i in range(4)
                ]
                capacity_trace.begin_layer(
                    batch,
                    0,
                    SimpleNamespace(horizon_store=SimpleNamespace(layers=[host])),
                )
                assert not host.resident and len(host.free) == capacity
            capacity_trace.close()
            assert not list(Path(tmp).glob("*.error"))
    print(
        "PASS: 7 capacities x 64 batches; batch-snapshot eviction, duplicate/padded slots and write invalidation"
        + ("; 192 GPU KV/hit checks and cold reset" if args.gpu else "")
    )


if __name__ == "__main__":
    main()
