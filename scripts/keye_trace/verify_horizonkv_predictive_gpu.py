"""Replay the native exact Top-K helper on recorded real candidate scores.

Run after the collection server releases the selected GPU. Candidate scoring
was already checked in the live model; this isolates the actual selection
helper on all widths, keeping numerical ties visible in the exported rows.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer

WIDTHS=[2048,2304,2560,3072,4096,6144,8192]


@torch.inference_mode()
def main(root):
    torch.set_num_threads(4)
    output=root/'analysis/gpu-rerank-observations.csv'
    manifest=[json.loads(s) for s in (root/'events/manifest.jsonl').read_text().splitlines()]
    count=ties=0
    with output.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['request_id','layer','step','candidate_k','candidate_recall',
            'gpu_rerank_recall','boundary_tied_tokens_lost','strict_tokens_lost'])
        writer.writeheader()
        for c in manifest:
            data=torch.load(root/'events'/c['file'],map_location='cpu',weights_only=False)
            ids=torch.stack([r['candidates'] for r in data['rows']]).cuda()
            scores=torch.stack([r['candidate_scores'] for r in data['rows']]).cuda()
            lengths=torch.tensor([r['context_len'] for r in data['rows']],device='cuda',dtype=torch.int32)
            for width in WIDTHS:
                offsets=KeyeIndexer._topk_with_lengths(None,scores[:,:width],lengths.clamp(max=width),2048)
                final=ids[:,:width].gather(1,offsets.long()).cpu().numpy()
                for i,r in enumerate(data['rows']):
                    candidates=r['candidates'].numpy()[:width]
                    sc=r['candidate_scores'].numpy()[:width]
                    candidate_valid=candidates>=0
                    native=r['exact_indices'].numpy()
                    covered=np.isin(candidates,native)&candidate_valid
                    kept=np.isin(candidates,final[i])
                    strict=int((covered&~kept&(sc>r['native_cutoff'])).sum())
                    assert strict==0, (c,r['step'],width)
                    candidate_hits=int(covered.sum())
                    final_hits=int(np.isin(final[i],native).sum())
                    tie_loss=candidate_hits-final_hits
                    assert tie_loss>=0
                    writer.writerow(dict(request_id=data['request_id'],layer=data['target_layer_id'],step=r['step'],
                        candidate_k=width,candidate_recall=candidate_hits/2048,gpu_rerank_recall=final_hits/2048,
                        boundary_tied_tokens_lost=tie_loss,strict_tokens_lost=strict))
                    count+=1;ties+=int(tie_loss>0)
    result=dict(observations=count,strict_losses=0,boundary_tie_cases=ties,
        gpu_name=torch.cuda.get_device_name(),scope='Native exact Top-K helper replay on live candidate-kernel scores; no timing or task-quality inference')
    (root/'analysis/gpu-rerank-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir.resolve())
