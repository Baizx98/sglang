"""GPU numerical checks for mapped Host gather and batched device caching."""

from collections import OrderedDict

import torch

from horizonkv_host_kernels import DeviceLRU, HostPointer, gather, resident_gather, write_current


def main():
    torch.manual_seed(20261014)
    batch, length, topk = 2, 64, 8
    hostk = torch.randn(batch, length, 4, 128, dtype=torch.bfloat16).pin_memory()
    hostv = torch.randn_like(hostk).pin_memory()
    hk, hv = HostPointer(hostk), HostPointer(hostv)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    outk = torch.empty(batch, topk, 4, 128, device="cuda", dtype=torch.bfloat16)
    outv = torch.empty_like(outk)
    ids = (torch.arange(batch * topk, device="cuda").reshape(batch, topk) * 3 % length).int()
    gather(hk, hv, ids, outk, outv, lengths, length)
    torch.cuda.synchronize()
    expectedk = torch.stack([hostk[b, ids[b].cpu().long()] for b in range(batch)])
    expectedv = torch.stack([hostv[b, ids[b].cpu().long()] for b in range(batch)])
    assert torch.equal(outk.cpu(), expectedk)
    assert torch.equal(outv.cpu(), expectedv)
    cache = DeviceLRU(batch, length, 16, topk)
    reference = [OrderedDict() for _ in range(batch)]
    checks = 0
    for step in range(20):
        # Recurrent demand plus new positions exercises hits, eviction and return.
        selected = torch.stack([torch.randperm(length)[:topk] for _ in range(batch)])
        ids = selected.int().cuda()
        # Frozen-residency diagnostics must read correct hits/misses without eviction.
        before = cache.map.clone()
        resident_gather(cache, hk, hv, ids, outk, outv, lengths, length)
        assert torch.equal(cache.map, before)
        assert torch.equal(outk.cpu(), torch.stack([hostk[b, selected[b]] for b in range(batch)]))
        assert torch.equal(outv.cpu(), torch.stack([hostv[b, selected[b]] for b in range(batch)]))
        misses = cache.resolve(hk, hv, ids, outk, outv, lengths, step).cpu().tolist()
        for b in range(batch):
            old = reference[b]
            wanted = selected[b].tolist()
            hit = [i for i in wanted if i in old]
            miss = [i for i in wanted if i not in old]
            assert misses[b] == len(miss)
            for i in hit:
                old.move_to_end(i)
            for i in miss:
                while len(old) >= 16:
                    old.popitem(last=False)
                old[i] = None
            # This kernel's refinement promotes old hits above first-time misses.
            for i in hit:
                old.move_to_end(i)
            assert set(torch.where(cache.map[b] >= 0)[0].cpu().tolist()) == set(old)
            assert torch.equal(outk[b].cpu(), hostk[b, selected[b]])
            assert torch.equal(outv[b].cpu(), hostv[b, selected[b]])
            checks += 1
    newk = torch.ones(batch, 512, device="cuda", dtype=torch.bfloat16)
    newv = torch.full_like(newk, 2)
    write_current(hk, hv, newk, newv, lengths, length)
    torch.cuda.synchronize()
    assert bool((hostk[:, -1] == 1).all())
    assert bool((hostv[:, -1] == 2).all())
    print(f"mapped Host gather/write and {checks} cache transitions passed")


if __name__ == "__main__":
    main()
