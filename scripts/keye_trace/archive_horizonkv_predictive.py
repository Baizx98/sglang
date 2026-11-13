"""Archive audited, processed 4.2 tables only; keep raw traces in the run dir."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root,destination,prefix,partial=False):
    source=root/'analysis'
    audit=json.loads((source/'audit.json').read_text())
    gpu_path=source/'gpu-rerank-audit.json'
    gpu=json.loads(gpu_path.read_text()) if gpu_path.exists() else None
    collection=json.loads((root/'server-command.json').read_text())
    if partial:
        assert audit['partial'] and audit['request_count']>0
        assert audit['chunks']==audit['request_count']*47
    else:
        assert not audit['partial'] and audit['request_count']==24 and audit['chunks']==24*47
        assert audit['layer_step_observations']==24*47*32 and gpu['strict_losses']==0
    outputs=[]
    names=['method-observations','candidate-observations','gpu-rerank-observations',
        'method-by-request','candidate-by-request','method-summary','candidate-summary',
        'method-by-layer','candidate-by-layer','candidate-tail']
    if gpu is None:
        names.remove('gpu-rerank-observations')
    for name in names:
        raw=source/(name+'.csv')
        frame=pd.read_csv(raw)
        extension='parquet' if name.endswith('observations') else 'csv'
        path=destination/f'{prefix}{name}_v01_reviewed.{extension}'
        if path.exists():
            raise FileExistsError(f'Never overwrite archived core data: {path}')
        if extension=='parquet':
            frame.to_parquet(path,index=False,compression='zstd')
            pd.testing.assert_frame_equal(pd.read_parquet(path),frame)
        else:
            shutil.copyfile(raw,path)
            assert digest(raw)==digest(path)
        outputs.append(dict(file=path.name,rows=len(frame),source_sha256=digest(raw),sha256=digest(path),
            fields={k:str(v) for k,v in frame.dtypes.items()}))
    provenance=dict(run_dir=str(root),completion='partial' if partial else 'complete',source_audit=audit,gpu_rerank_audit=gpu,
        environment=json.loads((root/'environment.json').read_text()),
        collection=collection,
        selection=json.loads((root/'selection.json').read_text()),outputs=outputs,
        archive_script_sha256=digest(Path(__file__)),
        units='Recall/Jaccard/coverage fractions 0..1; candidate_ratio unitless; lengths/n/counts tokens; layer/step zero-based. No latency/throughput measurements.',
        evidence=f"Fresh PP{collection.get('pp_size',1)}/TP{collection.get('tp_size',2)} exact-main shadow trace; all 47 adjacent layers included. Live candidate-kernel scores verified; CPU stable reranking audited. Native GPU Top-K replay "+('verified.' if gpu else 'not run due to driver failure.'),
        exclusions='First Decode step intentionally absent from previous-step method; matched method comparisons use steps1..31. No request or layer excluded by outcome.')
    path=destination/f'{prefix}provenance_v01_reviewed.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(outputs)+1,destination=str(destination)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--prefix',default='2026-11-12_1655_predictive-')
    p.add_argument('--partial',action='store_true',help='Explicitly archive the audited subset, never label the full study complete')
    a=p.parse_args();main(a.run_dir.resolve(),a.destination.resolve(),a.prefix,a.partial)
