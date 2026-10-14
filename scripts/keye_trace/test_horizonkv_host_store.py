"""GPU integration checks for exact host KV, eviction and physical-slot reuse."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KEYE_SM80_DSA", "1")

import torch
from sglang.srt.layers.attention.keye_dsa_sm80 import keye_dsa_paged_attention
from sglang.srt.layers.attention.keye_topk.horizonkv_probe import (
    HostLayer,
    HostStore,
    _close_writer,
    span,
)


def main():
    torch.manual_seed(20261014)
    device = "cuda:0"
    k = torch.randn(32, 2, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    q = torch.randn(2, 16, 128, dtype=torch.bfloat16, device=device)
    batches = [
        torch.tensor(x, dtype=torch.int32)
        for x in (
            [[1, 2, 3, 4, -1], [7, 8, 9, 10, 11]],
            [[2, 3, 5, 6, 12], [7, 8, 10, 14, -1]],
            [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]],
        )
    ]
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))
    comparisons = 0
    for capacity in [0, 2, 16]:
        host = HostLayer(32, 2, 128, torch.bfloat16, device, capacity)
        host.write(torch.arange(32), k, v)
        for slots in batches:
            sk, sv, remapped = host.stage(slots, batch, 0, slots)
            mask = slots.flatten() >= 0
            torch.testing.assert_close(
                sk[mask.to(device)], k[slots.flatten()[mask].to(device)], rtol=0, atol=0
            )
            torch.testing.assert_close(
                sv[mask.to(device)], v[slots.flatten()[mask].to(device)], rtol=0, atol=0
            )
            reference = keye_dsa_paged_attention(
                q,
                k,
                v,
                slots.to(device),
                num_q_heads=16,
                head_dim=128,
                sm_scale=128**-0.5,
                is_decode=True,
            )
            actual = keye_dsa_paged_attention(
                q,
                sk,
                sv,
                remapped,
                num_q_heads=16,
                head_dim=128,
                sm_scale=128**-0.5,
                is_decode=True,
            )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert len(host.resident) <= capacity
            comparisons += 1
        # A finished request's physical pool slot may be reused by a new one.
        replacement_k = torch.zeros_like(k[2:3])
        replacement_v = torch.ones_like(v[2:3])
        host.write(torch.tensor([2]), replacement_k, replacement_v)
        assert 2 not in host.resident
        sk, sv, _ = host.stage(
            torch.tensor([[2]], dtype=torch.int32), batch, 0, torch.tensor([[2]])
        )
        torch.testing.assert_close(sk, replacement_k, rtol=0, atol=0)
        torch.testing.assert_close(sv, replacement_v, rtol=0, atol=0)
    # Non-monotonic physical page tables must preserve prefill attention too.
    os.environ["KEYE_HORIZON_STORAGE"] = "dram"
    pool = SimpleNamespace(
        size=32,
        page_size=0,
        head_num=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device=device,
        layer_num=1,
        start_layer=0,
    )
    store = HostStore(pool)
    store.write(0, torch.arange(32, device=device), k, v)
    table = torch.tensor([[4, 7, 1, 6]], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        page_table_1=table, cache_seqlens_int32=torch.tensor([4], device=device)
    )
    physical = torch.tensor(
        [[4, -1, -1, -1], [4, 7, -1, -1], [4, 7, 1, -1], [4, 7, 1, 6]],
        dtype=torch.int32,
        device=device,
    )
    pk, pv, remapped = store.prefill(0, metadata, physical)
    pq = torch.randn(4, 16, 128, dtype=torch.bfloat16, device=device)
    a = keye_dsa_paged_attention(
        pq,
        k,
        v,
        physical,
        num_q_heads=16,
        head_dim=128,
        sm_scale=128**-0.5,
        is_decode=False,
    )
    b = keye_dsa_paged_attention(
        pq,
        pk,
        pv,
        remapped,
        num_q_heads=16,
        head_dim=128,
        sm_scale=128**-0.5,
        is_decode=False,
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # M3 can be high while coverage of actual pre-update HBM misses is zero.
    host = HostLayer(32, 2, 128, torch.bfloat16, device, 4)
    host.write(torch.arange(32), k, v)
    host.stage(
        torch.tensor([[1, 2, 3]], dtype=torch.int32),
        batch,
        1,
        torch.tensor([[1, 2, 3]]),
    )
    with tempfile.TemporaryDirectory() as directory:
        os.environ["KEYE_HORIZON_PROBE_DIR"] = directory
        audited = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            rids=["selection__test"],
            seq_lens_cpu=torch.tensor([16]),
            _horizon_steps=[0],
            _horizon_previous_selected={"selection__test": {1, 2, 3}},
            _horizon_previous_physical={"selection__test": {1: 1, 2: 2, 3: 3}},
        )
        host.stage(
            torch.tensor([[1, 2, 4]], dtype=torch.int32),
            audited,
            1,
            torch.tensor([[1, 2, 4]]),
        )
        clock_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            rids=["horizon__clock"],
            seq_lens_cpu=torch.tensor([16]),
            _horizon_steps=[0],
        )
        with (
            span(clock_batch, -1, "model_total"),
            span(clock_batch, 0, "attention_compute"),
        ):
            _ = q + q
        _close_writer()
        records = [
            json.loads(line)
            for path in Path(directory).glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        v = next(v for v in records if v["kind"] == "selection")
        assert v["selected_count"] == 3 and v["miss"] == [4] and v["hit_count"] == 2
        assert (
            v["overlap_numerator"] == 2
            and v["miss_coverage_numerator"] == 0
            and v["hint_resident_count"] == 3
        )
        model = next(v for v in records if v.get("component") == "model_total")
        child = next(v for v in records if v.get("component") == "attention_compute")
        assert 0 <= child["start_from_model_ms"] <= child["end_from_model_ms"]
        assert child["end_from_model_ms"] <= model["end_from_model_ms"] + 0.01
        assert (
            abs(
                child["cuda_stream_span_ms"]
                - (child["end_from_model_ms"] - child["start_from_model_ms"])
            )
            < 0.01
        )
    print(
        f"PASS: {comparisons + 1} exact GPU attention comparisons; three slot-reuse checks; pre-update Miss coverage audit"
    )


if __name__ == "__main__":
    main()
