"""Audit same-state raw chunks and export observation/request/layer tables."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

K = 2048
WIDTHS = [2048, 2304, 2560, 3072, 4096, 6144, 8192]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, partial=False):
    torch.set_num_threads(4)
    requests = {r['rid']: r for r in [json.loads(s) for s in (root/'requests.jsonl').read_text().splitlines()]}
    manifest = [json.loads(s) for s in (root/'events/manifest.jsonl').read_text().splitlines()]
    identities = [(r['request_id'],r['target_layer_id']) for r in manifest]
    chunk_lookup={(r['request_id'],r['target_layer_id']):root/'events'/r['file'] for r in manifest}
    capture_stages={}
    pp_boundaries_verified=0
    assert len(identities) == len(set(identities))
    if not partial:
        assert set(identities) == {(rid,layer) for rid in requests for layer in range(1,48)}
    base_rows, expansion_rows, raw_sources = [], [], []
    for chunk in manifest:
        path = root/'events'/chunk['file']
        data = torch.load(path, map_location='cpu', weights_only=False)
        assert data['schema_version']==2 and data['exact_main'] and not data['prediction_uses_current_key']
        assert [r['step'] for r in data['rows']] == list(range(32))
        meta = requests[data['request_id']]
        if 'pipeline_start_layer' in data:
            start=int(data['pipeline_start_layer'])
            device=data['capture_device']
            if start in capture_stages:
                assert capture_stages[start]==device
            capture_stages[start]=device
            if start>0 and data['target_layer_id']==start:
                preceding=torch.load(chunk_lookup[(data['request_id'],start-1)],map_location='cpu',weights_only=False)
                for here,there in zip(data['rows'],preceding['rows'],strict=True):
                    assert torch.equal(here['previous_layer_indices'],there['exact_indices']), (path,here['step'])
                pp_boundaries_verified+=1
        previous = None
        for r in data['rows']:
            length = r['context_len']
            assert length == meta['prompt_tokens']+r['step']+1
            native = r['exact_indices'].numpy().astype(np.int64)
            candidates = r['candidates'].numpy().astype(np.int64)
            scores = r['candidate_scores'].numpy()
            valid = candidates>=0
            assert len(set(native))==K and native.min()>=0 and native.max()<length
            assert len(set(candidates[valid]))==min(8192,length)
            assert (candidates[valid]<length).all() and candidates[0]==length-1
            assert r['score_max_abs']==0, (path,r['step'],r['score_max_abs'])
            if r['step']==0:
                assert r['selfcheck_max_abs']==0 and r['selfcheck_equal']
            member = np.zeros(length,dtype=bool)
            member[native]=True
            common = dict(request_id=data['request_id'],case_id=meta['case_id'],group=meta['group'],
                task=meta['task'],length_bucket=meta['length_bucket_upper'],prompt_tokens=meta['prompt_tokens'],
                layer=data['target_layer_id'],source_layer=data['source_layer_id'],step=r['step'],
                context_len=length,matched_three_methods=r['step']>0)
            methods = dict(native=native, previous_layer=r['previous_layer_indices'].numpy(),
                horizonkv=candidates[:K])
            if previous is not None:
                methods['previous_step']=previous
            for method, ids in methods.items():
                ids = ids[ids>=0]
                assert len(np.unique(ids))==K
                intersection=int(member[ids].sum())
                base_rows.append(dict(common,method=method,recall=intersection/K,
                    jaccard=intersection/(K+len(ids)-intersection),selected=len(ids)))
            for width in WIDTHS:
                take=min(width,length)
                ids=candidates[:width]
                sc=scores[:width]
                keep=ids>=0
                ids,sc=ids[keep],sc[keep]
                assert len(ids)==take
                hits=int(member[ids].sum())
                # Stable descending ranking makes offline tie behavior explicit.
                chosen=np.argsort(-sc,kind='stable')[:K]
                final=ids[chosen]
                final_hits=int(member[final].sum())
                lost=member[ids].copy()
                lost[chosen]=False
                strict_lost=int((lost & (sc>r['native_cutoff'])).sum())
                assert strict_lost==0, (path,r['step'],width)
                tie_loss=hits-final_hits
                assert tie_loss>=0
                expansion_rows.append(dict(common,n=width-K,candidate_k=width,candidate_ratio=width/K,
                    effective_candidates=take,recall_at_M=hits/K,rerank_recall=final_hits/K,
                    rerank_jaccard=final_hits/(2*K-final_hits),coverage_minus_rerank= tie_loss/K,
                    tied_native_tokens_lost=tie_loss,strict_tokens_lost=strict_lost,
                    score_max_abs=r['score_max_abs'],native_cutoff=r['native_cutoff']))
            previous=native
        raw_sources.append(dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size))
    out=root/'analysis';out.mkdir(exist_ok=True)
    base=pd.DataFrame(base_rows);expansion=pd.DataFrame(expansion_rows)
    base.to_csv(out/'method-observations.csv',index=False)
    expansion.to_csv(out/'candidate-observations.csv',index=False)
    request_base=base[base.matched_three_methods].groupby(['request_id','group','length_bucket','method'],as_index=False).agg(
        recall=('recall','mean'),jaccard=('jaccard','mean'),observation_p10=('recall',lambda x:x.quantile(.1)),observations=('recall','size'))
    request_base.to_csv(out/'method-by-request.csv',index=False)
    request_exp=expansion.groupby(['request_id','group','length_bucket','candidate_k','candidate_ratio'],as_index=False).agg(
        recall=('recall_at_M','mean'),rerank_recall=('rerank_recall','mean'),observation_p10=('recall_at_M',lambda x:x.quantile(.1)),
        observation_min=('recall_at_M','min'),observations=('recall_at_M','size'))
    request_exp.to_csv(out/'candidate-by-request.csv',index=False)
    def summary(frame,keys):
        return frame.groupby(keys,as_index=False).agg(mean=('recall','mean'),request_min=('recall','min'),
            request_max=('recall','max'),request_p10=('recall',lambda x:x.quantile(.1)),requests=('request_id','nunique'))
    summary(request_base,['group','length_bucket','method']).to_csv(out/'method-summary.csv',index=False)
    summary(request_exp,['group','length_bucket','candidate_k','candidate_ratio']).to_csv(out/'candidate-summary.csv',index=False)
    expansion.groupby(['group','length_bucket','candidate_k'],as_index=False).agg(
        observation_p10=('recall_at_M',lambda x:x.quantile(.10)),
        observation_p01=('recall_at_M',lambda x:x.quantile(.01)),
        observation_min=('recall_at_M','min'),
        fully_covered_fraction=('recall_at_M',lambda x:(x==1).mean()),
        observations=('recall_at_M','size')).to_csv(out/'candidate-tail.csv',index=False)
    base[base.matched_three_methods].groupby(['group','layer','method'],as_index=False).agg(
        recall=('recall','mean'),jaccard=('jaccard','mean'),minimum=('recall','min')).to_csv(out/'method-by-layer.csv',index=False)
    expansion.groupby(['group','layer','candidate_k'],as_index=False).agg(recall=('recall_at_M','mean'),
        rerank_recall=('rerank_recall','mean'),minimum=('recall_at_M','min')).to_csv(out/'candidate-by-layer.csv',index=False)
    audit=dict(partial=partial,request_count=base.request_id.nunique(),chunks=len(manifest),
        capture_stages=capture_stages,pp_boundary_request_pairs_verified=pp_boundaries_verified,
        layer_step_observations=int(len(expansion)/len(WIDTHS)),method_rows=len(base),candidate_rows=len(expansion),
        all_candidate_scores_exact=True,all_first_step_selfchecks_exact=True,strict_rerank_losses=0,
        tied_loss_observations=int((expansion.tied_native_tokens_lost>0).sum()),
        raw_sources=raw_sources,selection_sha256=sha(root/'selection.json'),
        statistics='Equal request weighting after averaging all 47 layers and matched steps 1..31 for method comparison; steps0..31 for expansion. Request min/max are ranges, not confidence intervals.',
        first_decode_step='No previous-step baseline; missing intentionally, never replaced by current-step ground truth.',
        scope='Exact trajectory shadow fidelity; not free-running task quality, asynchronous overlap or performance')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='raw_sources'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();analyze(a.run_dir.resolve(),a.partial)
