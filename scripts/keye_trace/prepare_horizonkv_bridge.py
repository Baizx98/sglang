"""Freeze August token inputs and original split for a current-code PP bridge."""
import argparse
import hashlib
import json
from pathlib import Path


def freeze(source, output):
    selection=json.loads((source/'selection.json').read_text())
    split={r['rid']:r['split'] for r in selection['selected']}
    original=[json.loads(s) for s in (source/'prepared_requests.jsonl').read_text().splitlines()]
    assert len(original)==24 and {r['rid'] for r in original}==set(split)
    rows=[]
    for i,r in enumerate(original):
        ids=r['input_ids']
        assert len(ids)==r['prompt_len']
        digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        rows.append(dict(r,rid=f'predictive__bridge__{i:02d}',original_rid=r['rid'],
            case_id=r['rid'],group='BFCL-'+split[r['rid']],task='BFCL',
            prompt_tokens=len(ids),length_bucket_upper=8192,input_ids_sha256=digest,
            source_file=str(source/'prepared_requests.jsonl'),split=split[r['rid']]))
    assert len({r['input_ids_sha256'] for r in rows})==24
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); rows=freeze(a.source.resolve(),a.output.resolve())
    print(json.dumps(dict(requests=len(rows),calibration=sum(r['split']=='calibration' for r in rows),
        test=sum(r['split']=='test' for r in rows),min_tokens=min(r['prompt_tokens'] for r in rows),
        max_tokens=max(r['prompt_tokens'] for r in rows))))
