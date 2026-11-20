"""SM86 correctness checks for the new Index Key representations."""

from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/indexer-round2"))
sys.path.insert(0, str(ROOT / "scripts/indexer-optimization"))

from bench import page_layout
from lossless import StaticExponentKeys
from quantized import QuantizedKeys
from slab_index import IndexSlab, Plan, _score_slab


def slab(keys, lengths):
    table, lives = page_layout(lengths, max(lengths))
    plan = Plan.from_lengths(lives, growth=64)
    return IndexSlab(plan, keys, table, lives, packed=False, exception_reserve=0)


def locations(owner, lengths):
    return torch.cat([
        torch.arange(n, dtype=torch.int32, device="cuda") + base
        for base, n in zip(owner.plan.bases.unbind(), lengths, strict=True)
    ])


def exact_bitspace():
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    keys = bits.view(torch.bfloat16).reshape(-1, 64)
    lengths = [513, 511]
    owner = slab(keys, lengths)
    encoded = StaticExponentKeys(owner)
    live = locations(owner, lengths)
    assert torch.equal(encoded.restore(live).view(torch.int16), keys.view(torch.int16))
    for offset in range(4):
        append = ((torch.arange(128, device="cuda", dtype=torch.int32) + offset * 971) * 171)
        append = append.to(torch.int16).view(torch.bfloat16).reshape(2, 64)
        targets = owner.plan.bases + owner.lengths + offset
        for _ in range(2):
            encoded.append(append, targets)
        values = encoded.restore(targets)
        assert torch.equal(values.view(torch.int16), append.view(torch.int16))
    assert torch.equal(encoded.restore(live).view(torch.int16), keys.view(torch.int16))
    print("65536 BF16 bit patterns plus monotonic/replayed escapes verified")


def expected_format(keys, bits, groups):
    shape = keys.shape
    vectors = keys.float().view(shape[0], groups, 64 // groups)
    max_code = 2 ** (bits - 1) - 1
    scales = (vectors.abs().amax(dim=2).clamp_min(1e-20) / max_code).bfloat16()
    values = (vectors / scales.float().unsqueeze(2)).round().clamp(-max_code, max_code)
    return values.to(torch.int32).view(shape), scales


def read_format(state, slots):
    raw = state.codes[slots.long()].int()
    if state.bits == 8:
        return raw
    nibble = torch.stack((raw & 15, raw >> 4), dim=2).flatten(1)
    if state.bits > 4:
        width = state.bits - 4
        word = state.planes[slots.long()].int()
        fields = torch.arange(8 // width, device=word.device) * width
        high = ((word[:, :, None] >> fields) & ((1 << width) - 1)).flatten(1)
        nibble |= high << 4
    return nibble - (1 << (state.bits - 1))


def verify_formats():
    length = [515, 700, 803, 513]
    generator = torch.Generator(device="cuda").manual_seed(20261120)
    keys = torch.randn((sum(length), 64), device="cuda", generator=generator).bfloat16()
    q = torch.randn((4, 16, 64), device="cuda", generator=generator).bfloat16()
    gate = torch.randn((4, 16), device="cuda", generator=generator).bfloat16()
    new = torch.randn((4, 64), device="cuda", generator=generator).bfloat16()
    owner = slab(keys, length)
    live = locations(owner, length)
    next_slots = owner.plan.bases + owner.lengths
    for bits, groups in ((4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (8, 1), (8, 2)):
        state = QuantizedKeys(owner, bits, groups)
        for reference, slots in ((keys, live), (new, next_slots)):
            if reference is new:
                state.append(new, next_slots)
            codes, scales = expected_format(reference, bits, groups)
            assert torch.equal(read_format(state, slots), codes), (bits, groups)
            assert torch.equal(state.scales[slots.long()], scales), (bits, groups)
        call, _ = state.prepare(owner.lengths, .125, 128)
        measured = call(q, gate)
        expected_q = q.float().view(4, 16, groups, 64 // groups)
        q_scales = expected_q.abs().amax(-1).clamp_min(1e-20) / 127
        q_int = (expected_q / q_scales[..., None]).round().clamp(-127, 127)
        for row, n in enumerate(length):
            ids = owner.plan.bases[row] + torch.arange(n, device="cuda")
            native_codes, native_scales = expected_format(owner.resident[ids], bits, groups)
            est = torch.zeros((16, n), device="cuda")
            for g in range(groups):
                d = slice(g * 64 // groups, (g + 1) * 64 // groups)
                dots = q_int[row, :, g] @ native_codes[:, d].float().T
                est += dots * q_scales[row, :, g, None] * native_scales[:, g].float()
            score = (est.bfloat16().float().mul(.125).relu() * gate[row, :, None]).sum(0)
            torch.testing.assert_close(measured[row, :n], score, atol=1e-5, rtol=1e-5)
        print(f"INT{bits} x {groups}: packed codes, scales, append and integer-score semantics verified")

    static = StaticExponentKeys(owner)
    compressed, _ = static.prepare(owner.lengths, .125, 128)
    oracle = torch.empty_like(compressed(q, gate))
    cache, _ = owner.buffers()
    _score_slab[(oracle.size(1) // 128, 4)](
        q, gate, owner.plan.bases, owner.plan.capacities, owner.lengths, *cache[:4],
        oracle, q.stride(0), q.stride(1), gate.stride(0), oracle.size(1), .125, 128, False,
        num_warps=4,
    )
    assert torch.equal(oracle, compressed(q, gate))
    print("Static exponent scores match the same-layout raw BF16 scorer")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        exact_bitspace()
        verify_formats()
