"""Describe correlated observations and independent request means separately."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

QUANTILES=[.01,.05,.10,.25,.50,.75,.90,.95,.99]


def describe(frame,keys,metrics):
    result=None
    for metric in metrics:
        groups=frame.groupby(keys,dropna=False)[metric]
        block=groups.agg(count='count',size='size',mean='mean',median='median',variance='var',std='std',minimum='min',maximum='max')
        block['missing_count']=block.pop('size')-block['count']
        q=groups.quantile(QUANTILES,interpolation='linear').unstack(-1)
        q.columns=[f'p{round(x*100):02d}' for x in q.columns]
        block=block.join(q).rename(columns=lambda x:metric+'_'+x)
        result=block if result is None else result.join(block)
    return result.reset_index()


def main(data,prefix,out):
    out.mkdir(parents=True,exist_ok=False)
    outputs=[];sources=[]
    for kind,value,metrics in [
        ('method','method',['recall','jaccard']),
        ('candidate','candidate_k',['recall_at_M','rerank_recall','rerank_jaccard','coverage_minus_rerank','tied_native_tokens_lost','strict_tokens_lost'])]:
        source=data/(prefix+kind+'-observations_v01_reviewed.parquet')
        df=pd.read_parquet(source)
        sources.append(dict(path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        if kind=='method':df=df[df.matched_three_methods]
        for scope,part in [('all47',df),('historical42',df[df.layer.isin([3]+list(range(5,46)))])]:
            for level,keys in [('request-layer',['request_id','group','length_bucket','layer',value]),('group-layer',['group','length_bucket','layer',value])]:
                result=describe(part,keys,metrics)
                name=f'{kind}-{scope}-{level}.parquet';result.to_parquet(out/name,index=False)
                outputs.append(dict(file=name,rows=len(result),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
            # Each request has equal weight in the main summary. Its dispersion
            # must not be confused with the pooled layer-step distribution.
            request=part.groupby(['request_id','group','length_bucket',value],as_index=False)[metrics].mean()
            for level,frame in [('request-means',request),('pooled-observations',part)]:
                result=describe(frame,['group','length_bucket',value],metrics)
                name=f'{kind}-{scope}-{level}.csv';result.to_csv(out/name,index=False)
                outputs.append(dict(file=name,rows=len(result),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
    (out/'provenance.json').write_text(json.dumps(dict(sources=sources,outputs=outputs,
        variance='Sample variance ddof=1; NA for fewer than two nonmissing observations',
        quantiles='linear; P01 P05 P10 P25 P50 P75 P90 P95 P99',
        masks='all47=1..47; historical42={3,5..45}; zero-based target layer IDs',
        scope='Descriptive statistics; pooled observations correlated; no confidence intervals',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(files=len(outputs),output=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data_dir,a.prefix,a.output)
