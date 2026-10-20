"""Opt-in real Decode fixtures; capture overhead is never a timing result.

The first step saves each request's logical history. Later steps save only the
new KV, preserving continuous cache accesses without rewriting entire histories.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

_contexts = {}
_steps = {}


def cpu(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def begin(layer, positions, hidden, residual, batch):
    root = os.getenv("KEYE_LAYER_REPLAY_DIR")
    if not root or not batch.forward_mode.is_decode():
        return
    wanted = {int(v) for v in os.getenv("KEYE_LAYER_REPLAY_LAYERS", "11,12,23,24").split(",")}
    if layer.layer_id not in wanted:
        return
    rids = list(batch.rids)
    if not rids or not all(r.startswith("replay__") for r in rids):
        return
    steps = [_steps.get((r, layer.layer_id), 0) for r in rids]
    limit = int(os.getenv("KEYE_LAYER_REPLAY_STEPS", "64"))
    if min(steps) >= limit:
        return
    for r, step in zip(rids, steps):
        _steps[r, layer.layer_id] = step + 1
    _contexts[layer.layer_id] = {
        "root": root, "request_ids": rids, "steps": steps,
        "layer": layer.layer_id, "positions": cpu(positions),
        "hidden": cpu(hidden), "residual": cpu(residual),
        "needs_allreduce": bool(getattr(hidden, "_sglang_needs_allreduce_fusion", False)),
        "seq_lens": cpu(batch.seq_lens), "input_ids": cpu(batch.input_ids),
    }


def attention_input(layer, hidden):
    context = _contexts.get(layer)
    if context is not None:
        context["attention_input"] = cpu(hidden)


def attention_output(layer, q, indices, slots, k_cache, v_cache, output, batch, page_table):
    context = _contexts.get(layer)
    if context is None:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    root = Path(context["root"])
    root.mkdir(parents=True, exist_ok=True)
    context.update(query=cpu(q), topk=cpu(indices), attention_output=cpu(output))
    index_cache = batch.token_to_kv_pool.get_index_k_bf16_buffer(layer)
    for row, (rid, step) in enumerate(zip(context["request_ids"], context["steps"])):
        if step >= int(os.getenv("KEYE_LAYER_REPLAY_STEPS", "64")):
            continue
        length = int(context["seq_lens"][row])
        prefix = f"{rid}_layer{layer:02d}_rank{rank:02d}"
        current_slot = batch.out_cache_loc[row].long()
        delta = {"main_k": cpu(k_cache[current_slot]),
                 "main_v": cpu(v_cache[current_slot]),
                 "index_k": cpu(index_cache[current_slot])}
        context.setdefault("deltas", []).append(delta)
        if step == 0:
            physical = page_table[row, :length].long()
            snapshot = {
                "schema": "keye-layer-history-v1", "request_id": rid,
                "layer": layer, "rank": rank, "length": length,
                "main_k": cpu(k_cache[physical]), "main_v": cpu(v_cache[physical]),
                "index_k": cpu(index_cache[physical]),
            }
            torch.save(snapshot, root / f"{prefix}_history.pt")


def end(layer, output):
    context = _contexts.pop(layer, None)
    if context is None:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    context.update(output_hidden=cpu(output[0]), output_residual=cpu(output[1]),
                   rank=rank, schema="keye-layer-step-v1")
    root = Path(context.pop("root"))
    name = f"{context['request_ids'][0]}_layer{layer:02d}_rank{rank:02d}_batchstep{context['steps'][0]:03d}.pt"
    torch.save(context, root / name)
    with (root / f"manifest_rank{rank:02d}.jsonl").open("a") as stream:
        stream.write(json.dumps({"file": name, "request_ids": context["request_ids"],
                                 "steps": context["steps"], "layer": layer}) + "\n")
