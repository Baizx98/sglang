"""Merge audited completed observations and export multilevel distributions."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path

import numpy as np
import pandas as pd
from summarize_predictive_distributions import describe
from horizonkv_predictive_runtime import notify
from horizonkv_single_runtime import save

MASK=[3]+list(range(5,46))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]

def main(old,new,view,out):
    out.mkdir(exist_ok=False);source_meta=read(view/'requests.jsonl')
    meta=pd.DataFrame([dict(request_id=r['rid'],dataset=r['dataset'],split=r['split'],task_name=r['task'],
        length_bucket=r['length_bucket_upper'],prompt_tokens=r['prompt_tokens']) for r in source_meta])
    assert len(meta)==330 and meta.request_id.nunique()==330
    meta.to_csv(out/'request-metadata.csv',index=False);metadata=meta.set_index('request_id')
    outputs=[];source_hashes=[];facts={};start=time.time()
    def write(df,name):
        target=out/name
        if target.suffix=='.parquet':df.to_parquet(target,index=False,compression='zstd')
        else:df.to_csv(target,index=False)
        outputs.append(dict(file=name,rows=len(df),sha256=sha(target),fields={k:str(v) for k,v in df.dtypes.items()}))
    def progress(phase):
        save(out/'progress.json',dict(status='running',phase=phase,files=len(outputs),elapsed_s=time.time()-start))
        print(phase,flush=True)
    specs=[('method','method',['recall','jaccard'],330*47*255),
        ('candidate','candidate_k',['recall_at_M','rerank_recall','rerank_jaccard','coverage_minus_rerank'],330*47*64*8),
        ('attention','method',['attention_cosine','attention_relative_l2','attention_max_abs','exact_norm'],330*47*64*3),
        ('representation',None,['hidden_cosine','hidden_relative_l2','score_mae','score_nmae','score_cosine','score_spearman'],330*47*64)]
    for kind,variant,metrics,expected in specs:
        progress('loading_'+kind);frames=[]
        for source in [old,new]:
            f=source/'analysis'/(kind+'-observations.csv');frame=pd.read_csv(f)
            frames.append(frame);source_hashes.append(dict(path=str(f),sha256=sha(f),rows=len(frame)))
        df=pd.concat(frames,ignore_index=True);del frames
        assert len(df)==expected,(kind,len(df),expected)
        keys=['request_id','layer','step']+([variant] if variant else [])
        assert not df.duplicated(keys).any(),kind
        assert set(df.request_id)==set(metadata.index)
        group=df.groupby('request_id',observed=True)
        expected_per_request=expected//330
        assert group.size().eq(expected_per_request).all()
        assert group.layer.nunique().eq(47).all() and group.step.nunique().eq(64).all()
        for field in ['dataset','split','task_name']:
            df[field]=df.request_id.map(metadata[field])
        for field in ['length_bucket','prompt_tokens']:
            if field not in df:df[field]=df.request_id.map(metadata[field])
        for metric in metrics:
            values=df[metric].dropna()
            assert np.isfinite(values).all(),(kind,metric)
            if metric.endswith('cosine'):assert values.between(-1.00001,1.00001).all(),metric
            elif metric in ['recall','jaccard','recall_at_M','rerank_recall','rerank_jaccard']:
                assert values.between(0,1).all(),metric
            else:assert (values>=0).all() if metric not in ['score_spearman'] else values.between(-1,1).all()
        if kind=='candidate':
            assert (df.score_max_abs==0).all() and (df.strict_tokens_lost==0).all()
            facts['cpu_boundary_tie_cases']=int((df.tied_native_tokens_lost>0).sum())
            df['fully_covered']=(df.recall_at_M==1).astype(float)
            df['candidate_precision']=df.recall_at_M*2048/df.effective_candidates
            df['missing_native_tokens']=(1-df.recall_at_M)*2048
            metrics+=['fully_covered','candidate_precision','missing_native_tokens']
            # Required-M is defined only on the tested discrete grid. Missing
            # values mean right censoring; report its rate separately.
            pivot=df.pivot(index=['request_id','layer','step'],columns='candidate_k',values='recall_at_M')
            widths=pivot.columns.to_numpy();v=pivot.to_numpy();budget=pivot.index.to_frame(index=False)
            assert list(widths)==[2048,2304,2560,3000,3072,4096,6144,8192] and np.isfinite(v).all()
            for threshold,label in [(.9,'90'),(.95,'95'),(.99,'99'),(.999,'999')]:
                required=np.where(v>=threshold,widths[None,:],np.inf).min(axis=1)
                budget['required_M_'+label]=np.where(np.isfinite(required),required,np.nan)
                budget['censored_'+label]=(~np.isfinite(required)).astype(float)
            for field in ['dataset','split','task_name','length_bucket']:budget[field]=budget.request_id.map(metadata[field])
            write(budget,'required-budget-observations.parquet')
            write(describe(budget,['dataset','task_name','length_bucket','split'],[c for c in budget if c.startswith(('required_M','censored_'))]),'required-budget-by-task.csv')
            del pivot,v,budget
        if kind=='representation':
            sampled=df.step.isin([0,15,31,63]);assert df.loc[~sampled,'score_spearman'].isna().all()
            facts['spearman_sampled_rows']=int(df.score_spearman.notna().sum())
        if kind=='attention':facts['zero_attention_norms']=int((df.exact_norm==0).sum())
        write(df,kind+'-observations.parquet');facts[kind+'_rows']=len(df)
        raw=df if kind!='method' else df[df.matched_three_methods].copy()
        variants=[variant] if variant else []
        group_keys=['dataset','task_name','length_bucket','split']+variants
        progress('summarizing_'+kind)
        write(describe(raw,['request_id','layer']+variants,metrics),kind+'-request-layer.parquet')
        write(describe(raw,group_keys+['layer'],metrics),kind+'-task-layer.parquet')
        # Include every layer as the main result. The fixed historical mask is
        # an explicit diagnostic and is never selected on these test outcomes.
        for scope,part in [('all47',raw),('historical42',raw[raw.layer.isin(MASK)])]:
            request=part.groupby(['request_id']+group_keys,as_index=False,observed=True)[metrics].mean()
            write(request,kind+'-'+scope+'-by-request.csv')
            write(describe(request,group_keys,metrics),kind+'-'+scope+'-request-distribution.csv')
            write(describe(part,group_keys,metrics),kind+'-'+scope+'-pooled-distribution.csv')
            write(describe(request,['dataset','split']+variants,metrics),kind+'-'+scope+'-dataset-request-distribution.csv')
        if kind=='method':
            per=raw.groupby(['request_id','method'],observed=True).recall.mean().unstack()
            facts['h_beats_previous_layer_requests']=int((per.horizonkv>per.previous_layer).sum())
            facts['h_beats_previous_step_requests']=int((per.horizonkv>per.previous_step).sum())
        if kind=='attention':
            worst=raw[raw.method=='horizonkv_M3072'].nlargest(50,'attention_relative_l2')
            write(worst,'worst-attention-M3072.csv')
        del df,raw;gc.collect()
    progress('write_provenance')
    audits=[json.loads((r/'analysis/audit.json').read_text()) for r in [old,new]]
    assert sum(a['request_count'] for a in audits)==330
    assert sum(a['pp_boundary_request_pairs_verified'] for a in audits)==330
    provenance=dict(scope='Same-state exact-main shadow fidelity, not free-running task accuracy or performance',
        metadata_sha256=sha(out/'request-metadata.csv'),source_observations=source_hashes,outputs=outputs,facts=facts,
        source_audits=audits,view_audit=json.loads((view/'view-audit.json').read_text()),
        statistics='Method steps1..63; others0..63. Raw layer-step quantiles are correlated descriptive observations; request distributions average within each request first. Sample variance ddof1; linear quantiles; missing Spearman intentionally unsampled.',
        units='fractions and cosine unitless; fraction variance fraction^2; errors in attention/score native units; required M tokens; censoring rates separately reported; no CI from correlated observations',
        required_budget='Minimum TESTED M; NA if none reaches target; right-censored above8192, never treated as8192; conditional required_M summaries must be read with censored_* mean',
        script_sha256=sha(Path(__file__)))
    save(out/'provenance.json',provenance);save(out/'facts.json',facts)
    save(out/'progress.json',dict(status='completed',files=len(outputs),elapsed_s=time.time()-start))
    print(json.dumps(facts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--view',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:main(a.old.resolve(),a.new.resolve(),a.view.resolve(),a.output.resolve())
    except BaseException as e:
        save(a.output/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
