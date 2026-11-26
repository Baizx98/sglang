"""Instrumented Keye demand-only and lookahead LRU with real GPU transfers."""

from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch


@dataclass
class Transfer:
    start: torch.cuda.Event
    end: torch.cuda.Event
    incoming: dict
    evicted: dict
    buffers: tuple


class Partition:
    def __init__(self, hot_k, hot_v):
        self.k, self.v = hot_k, hot_v
        self.rid = None
        self.reset()

    def reset(self):
        self.resident = OrderedDict()
        self.free = deque(range(self.k.shape[0]))
        self.unused = set()
        self.pending = None
        self.step = 0
        self.transferred = set()

    def install(self, physical):
        victim = None
        if self.free:
            hot = self.free.popleft()
        else:
            victim, hot = self.resident.popitem(last=False)
        self.resident[physical] = hot
        wasted = victim in self.unused
        self.unused.discard(victim)
        return hot, victim, wasted


class LayerStudy:
    def __init__(self, host, capacity, max_requests, stream):
        self.host = host
        self.capacity = capacity
        self.parts = [
            Partition(host.hot_k[i * capacity:(i + 1) * capacity],
                      host.hot_v[i * capacity:(i + 1) * capacity])
            for i in range(max_requests)
        ]
        self.stream = stream
        self.row_bytes = 2 * host.k[0].nbytes

    def partition(self, rids):
        current = set(rids)
        assigned = {}
        idle = []
        for part in self.parts:
            if part.rid in current:
                assigned[part.rid] = part
            else:
                idle.append(part)
        for rid in rids:
            if rid not in assigned:
                part = idle.pop()
                if part.pending is not None:
                    raise RuntimeError("Request left with pending KV transfers")
                part.reset()
                part.rid = rid
                assigned[rid] = part
        return [assigned[rid] for rid in rids]

    def gather(self, physical):
        index = torch.as_tensor(physical, dtype=torch.long)
        shape = (len(index), *self.host.k.shape[1:])
        k = torch.empty(shape, dtype=self.host.k.dtype, pin_memory=True)
        v = torch.empty_like(k, pin_memory=True)
        torch.index_select(self.host.k, 0, index, out=k)
        torch.index_select(self.host.v, 0, index, out=v)
        return k, v

    def prefetch(self, rids, logical, physical):
        main_stream = torch.cuda.current_stream()
        for part, predicted, mapped in zip(self.partition(rids), logical, physical, strict=True):
            if not part.rid.startswith("ab__"):
                continue
            if part.pending is not None:
                raise RuntimeError("Overlapping prefetches into the same LRU partition")
            incoming, evicted, hot_slots = {}, {}, []
            for pos, slot in zip(predicted, mapped, strict=True):
                if pos < 0 or slot < 0 or slot in part.resident:
                    continue
                if len(incoming) == self.capacity:
                    break
                hot, victim, unused = part.install(slot)
                incoming[slot] = (pos, hot)
                hot_slots.append(hot)
                if victim is not None:
                    evicted[victim] = (hot, unused)
            if not incoming:
                continue
            cpu_k, cpu_v = self.gather(list(incoming))
            destinations = torch.tensor(hot_slots, device=self.host.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.stream.wait_stream(main_stream)
            with torch.cuda.stream(self.stream):
                start.record()
                part.k.index_copy_(0, destinations, cpu_k.to(self.host.device, non_blocking=True))
                part.v.index_copy_(0, destinations, cpu_v.to(self.host.device, non_blocking=True))
                end.record()
            part.pending = Transfer(start, end, incoming, evicted, (cpu_k, cpu_v, destinations))
            part.unused.update(incoming)

    def consume(self, batch, layer, logical, physical, mark, k, v):
        shape = (physical.numel(), *self.host.k.shape[1:])
        out_k = torch.empty(shape, dtype=self.host.k.dtype, device=self.host.device)
        out_v = torch.empty_like(out_k)
        width = physical.shape[1]
        current_k, current_v = k.reshape(-1, *shape[1:]), v.reshape(-1, *shape[1:])
        records = []
        for i, part in enumerate(self.partition(batch.rids)):
            begin = i * width
            context = int(batch.seq_lens_cpu[i])
            demand = [(pos, slot, begin + j) for j, (pos, slot) in
                      enumerate(zip(logical[i].tolist(), physical[i].tolist(), strict=True))
                      if 0 <= pos < context - 1 and slot >= 0]
            current = (logical[i] == context - 1).nonzero(as_tuple=True)[0].tolist()
            if current:
                out_k[begin + current[0]] = current_k[i]
                out_v[begin + current[0]] = current_v[i]
            transfer = part.pending
            prefetch, evicted = {}, {}
            ready_ms = wait_ms = None
            if transfer is not None:
                # Main-stream demand is marked before the CPU reads the indices.
                # Waiting here orders reads even when a transfer is classified late.
                torch.cuda.current_stream().wait_event(transfer.end)
                # The timestamp comparison is audit instrumentation. It does not
                # control which values are consumed or issue duplicate copies.
                transfer.end.synchronize()
                ready_ms = mark.elapsed_time(transfer.end)
                wait_ms = max(0.0, ready_ms)
                prefetch, evicted = transfer.incoming, transfer.evicted
                part.pending = None
            hits = [(s, o, part.resident[s]) for _, s, o in demand if s in part.resident]
            misses = [(pos, s, o) for pos, s, o in demand if s not in part.resident]
            if hits:
                sources = torch.tensor([h for _, _, h in hits], device=self.host.device)
                destinations = torch.tensor([o for _, o, _ in hits], device=self.host.device)
                out_k[destinations] = part.k[sources]
                out_v[destinations] = part.v[sources]
            if misses:
                cpu_k, cpu_v = self.gather([s for _, s, _ in misses])
                destinations = torch.tensor([o for _, _, o in misses], device=self.host.device)
                out_k[destinations] = cpu_k.to(self.host.device, non_blocking=True)
                out_v[destinations] = cpu_v.to(self.host.device, non_blocking=True)
            for _, slot, _ in sorted(demand):
                if slot in part.resident:
                    part.resident.move_to_end(slot)
                    part.unused.discard(slot)
            writes, demand_evictions = {}, []
            for _, slot, output in sorted(misses):
                hot, victim, wasted = part.install(slot)
                if victim is not None:
                    demand_evictions.append((victim, hot, wasted))
                writes[hot] = output
            if writes:
                target = torch.tensor(list(writes), device=self.host.device)
                source = torch.tensor(list(writes.values()), device=self.host.device)
                part.k.index_copy_(0, target, out_k[source])
                part.v.index_copy_(0, target, out_v[source])
            if part.rid.startswith("ab__"):
                transferred = list(prefetch) + [s for _, s, _ in misses]
                absent = {s for _, s, _ in misses}
                demand_rows = []
                for pos, slot, _ in demand:
                    if slot in prefetch:
                        state = "prefetch_ready" if ready_ms <= 0 else "inflight"
                    else:
                        state = "absent" if slot in absent else "resident"
                    demand_rows.append([pos, slot, state])
                records.append(dict(rid=part.rid, layer=layer, step=part.step, context=context,
                    active_batch=len(batch.rids),
                    demand=demand_rows, prefetch=[[p, s, h] for s, (p, h) in prefetch.items()],
                    prefetch_evictions=[[s, h, u] for s, (h, u) in evicted.items()],
                    demand_evictions=demand_evictions, current_selected=bool(current),
                    current_physical=int(physical[i, current[0]]) if current else None,
                    unused_resident=len(part.unused),
                    prefetch_completion_minus_demand_ms=ready_ms, exposed_transfer_ms=wait_ms,
                    transfer_ms=transfer.start.elapsed_time(transfer.end) if transfer else None,
                    demand_bytes=len(misses) * self.row_bytes,
                    prefetch_bytes=len(prefetch) * self.row_bytes,
                    repeat_h2d=sum(s in part.transferred for s in transferred)))
                part.transferred.update(transferred)
                part.step += 1
        if records:
            directory = Path(os.environ["KEYE_EVICTION_AB_DIR"])
            with (directory / f"layer{layer:02d}.jsonl").open("a") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        remapped = torch.arange(physical.numel(), dtype=torch.int32).reshape_as(physical)
        remapped[physical < 0] = -1
        return out_k, out_v, remapped.to(self.host.device)


def issue(source_layer, hidden_states, positions, batch):
    if not batch.forward_mode.is_decode() or not any(r.startswith("ab__") for r in batch.rids or ()):
        return
    if os.environ["KEYE_EVICTION_AB_MODE"] != "prefetch":
        return
    from sglang.srt.layers.attention.keye_topk.agent_trace import predict
    from sglang.srt.layers.attention.keye_sa_backend import transform_index_page_table_decode

    store = batch.token_to_kv_pool.horizon_store
    for target, previous in ((source_layer, getattr(batch, "_keye_lookahead_previous_x", None)),
                             (source_layer + 1, hidden_states)):
        # The layer at a PP boundary is issued by its own rank once the
        # preceding hidden state arrives. Interior layers issue one layer ahead.
        indexer = batch._keye_ab_indexers.get(target)
        if indexer is None or previous is None:
            continue
        if target == source_layer and target != store.start_layer:
            continue
        ids = predict(indexer, previous, positions, batch, target)[:, 1:].contiguous()
        metadata = batch.attn_backend.get_indexer_metadata(target, batch)
        slots = transform_index_page_table_decode(
            page_table=metadata.get_page_table_1(), topk_indices=ids, page_size=1)
        pairs = torch.stack((ids, slots)).cpu().tolist()
        store.studies[target - store.start_layer].prefetch(batch.rids, *pairs)
