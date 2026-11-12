"""Opt-in, rank-zero shadow capture. Never changes native selection or KV keys.

The predictor scores only historical target-layer keys. The newly generated
token reserves one candidate slot because its target-layer key is unavailable
at the proposed early production point. The full exact trajectory is retained.
"""
import json
import os
from pathlib import Path

import torch

from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_candidates,
    keye_indexer_score_paged,
)

_buffers = {}


@torch.no_grad()
def record(indexer, previous_x, current_x, previous_indices, exact_indices,
           exact_scores, positions, batch, layer, directory, selected_rows,
           steps, rids, step_limit, max_k):
    if not selected_rows:
        return
    metadata = batch.attn_backend.get_indexer_metadata(layer, batch)
    keys = batch.token_to_kv_pool.get_index_k_bf16_buffer(layer)
    pages = metadata.get_page_table_1()
    lengths = metadata.get_seqlens_int32()
    predicted_q, _, predicted_w = indexer._get_q_k_w_bf16(previous_x, positions, enable_dual_stream=False)
    history_lengths = (lengths-1).clamp_min(0)
    predicted_scores = keye_indexer_score_paged(predicted_q, keys, pages, predicted_w,
        history_lengths, indexer.softmax_scale)
    history_ids = indexer._topk_with_lengths(predicted_scores, history_lengths, max_k-1)
    candidates = torch.cat(((lengths-1).to(torch.int32).unsqueeze(1), history_ids), dim=1)
    true_q, _, true_w = indexer._get_q_k_w_bf16(current_x, positions, enable_dual_stream=False)
    rescored = keye_indexer_score_candidates(true_q, keys, pages, candidates, true_w,
        lengths, indexer.softmax_scale)
    reference_scores = exact_scores.gather(1, candidates.clamp_min(0).long())
    score_delta = torch.where(candidates >= 0, (rescored-reference_scores).abs(), 0).max(dim=1).values
    # The first-step full-score self-check detects hidden-state/key/position errors.
    selfcheck_delta = selfcheck_equal = None
    if 0 in steps:
        check_scores = keye_indexer_score_paged(true_q, keys, pages, true_w, lengths, indexer.softmax_scale)
        valid = torch.arange(check_scores.shape[1],device=lengths.device)[None,:] < lengths[:,None]
        selfcheck_delta = torch.where(valid, (check_scores-exact_scores).abs(), 0).max(dim=1).values
        check_ids = indexer._topk_with_lengths(check_scores, lengths, indexer.topk)
        selfcheck_equal = check_ids.sort(dim=1).values.eq(exact_indices.sort(dim=1).values).all(dim=1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, row_index in enumerate(selected_rows):
        rid, step = rids[i], steps[i]
        key = (rid, layer)
        rows = _buffers.setdefault(key, [])
        if step != len(rows):
            raise RuntimeError(f'Noncontiguous study trace {key}: {step} != {len(rows)}')
        def cpu(t):
            return t[row_index].detach().cpu().clone()
        rows.append(dict(step=step, context_len=int(lengths[row_index]), exact_indices=cpu(exact_indices),
            previous_layer_indices=cpu(previous_indices), candidates=cpu(candidates),
            candidate_scores=cpu(rescored), score_max_abs=float(score_delta[row_index]),
            native_cutoff=float(exact_scores[row_index].gather(0, exact_indices[row_index].long()).min()),
            selfcheck_max_abs=float(selfcheck_delta[row_index]) if step == 0 else None,
            selfcheck_equal=bool(selfcheck_equal[row_index]) if step == 0 else None))
        if len(rows) != step_limit:
            continue
        file = f'{rid}__layer{layer:02d}.pt'
        path = directory / file
        data = dict(schema_version=2, request_id=rid, target_layer_id=layer,
            source_layer_id=layer-1, topk=indexer.topk, max_candidates=max_k,
            pipeline_start_layer=batch.token_to_kv_pool.start_layer,
            capture_device=torch.cuda.get_device_name(),
            prediction_uses_current_key=False, current_token_reserved=True,
            exact_main=True, rows=rows)
        tmp = path.with_suffix('.tmp')
        torch.save(data, tmp)
        os.replace(tmp,path)
        with (directory/'manifest.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(file=file, request_id=rid, target_layer_id=layer,
                num_steps=len(rows), bytes=path.stat().st_size))+'\n')
        del _buffers[key]
