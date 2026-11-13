"""Freeze document-disjoint LongBench cohorts and the full LBv2 set."""
import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path
import pandas as pd
import requests
from transformers import AutoTokenizer

TASKS='narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news lcc repobench-p multifieldqa_zh'.split()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def textsha(t):return hashlib.sha256(t.encode()).hexdigest()

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    url='https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json'
    response=requests.get(url,timeout=30);response.raise_for_status();templates=response.json()
    (out/'dataset2prompt.json').write_bytes(response.content)
    tokenizer=AutoTokenizer.from_pretrained('/Tan/model/Keye-VL-2.0-30B-A3B',trust_remote_code=True)
    previous=Path('data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl')
    excluded={str(r.get('sample_id')) for r in [json.loads(s) for s in previous.read_text().splitlines()]}
    counts={};rows=[];rng=random.Random(20261113)
    archive=Path('/Tan/dataset/LongBench/data.zip')
    with zipfile.ZipFile(archive) as z:
        for task in TASKS:
            pool=[json.loads(s) for s in z.read('data/'+task+'.jsonl').decode().splitlines() if s]
            rng.shuffle(pool);selected=[];split_counts={'calibration':0,'test':0};seen_samples=set()
            for r in pool:
                sid=str(r.get('_id',textsha(r['context']+str(r.get('input')))))
                doc=textsha(r['context'])
                split='calibration' if int(textsha(doc+'20261113'),16)%3==0 else 'test'
                target=20 if split=='calibration' else 40
                if sid in excluded or sid in seen_samples or split_counts[split]>=target:continue
                seen_samples.add(sid);split_counts[split]+=1;selected.append((sid,doc,r,split))
                if len(selected)==60:break
            counts[task]=len(selected)
            for i,(sid,doc,r,split) in enumerate(selected):
                prompt=templates[task].format(**r)
                rendered=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                ids=tokenizer.encode(rendered,add_special_tokens=False)
                original=len(ids);truncated=original>120000
                if truncated:ids=ids[:60000]+ids[-60000:]
                rows.append(dict(rid='predictive__lb__'+task+'__'+sid,case_id=sid,dataset='LongBench',task=task,
                    group=task,split=split,document_sha256=doc,
                    input_ids=ids,prompt_tokens=len(ids),original_prompt_tokens=original,truncated=truncated,
                    length_bucket_upper=next(c for c in [4096,8192,16384,32768,65536,131072] if len(ids)<=c),
                    expected_answer=r['answers'],input_ids_sha256=textsha(json.dumps(ids,separators=(',',':')))))
    # LBv2 remains a test-only cohort. Its prompt is frozen separately from LBv1.
    v2url='https://raw.githubusercontent.com/THUDM/LongBench/main/prompts/0shot.txt'
    v2response=requests.get(v2url,timeout=30);v2response.raise_for_status()
    v2template=v2response.text
    (out/'lbv2-0shot.txt').write_bytes(v2response.content)
    v2=Path('/Tan/dataset/LongBench-v2/0000.parquet');frame=pd.read_parquet(v2)
    assert len(frame)==503 and frame._id.nunique()==503
    for r in frame.to_dict('records'):
        prompt=v2template.replace('$DOC$',r['context'].strip()).replace('$Q$',r['question'].strip())
        for c in 'ABCD':prompt=prompt.replace('$C_'+c+'$',r['choice_'+c].strip())
        rendered=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        ids=tokenizer.encode(rendered,add_special_tokens=False);original=len(ids)
        if original>120000:ids=ids[:60000]+ids[-60000:]
        rows.append(dict(rid='predictive__lbv2__'+r['_id'],case_id=r['_id'],dataset='LongBench-v2',task=r['domain'],
            group='LongBench-v2',split='test',document_sha256=textsha(r['context']),input_ids=ids,
            prompt_tokens=len(ids),original_prompt_tokens=original,truncated=original>120000,
            length_bucket_upper=next(c for c in [4096,8192,16384,32768,65536,131072] if len(ids)<=c),
            expected_answer=r['answer'],difficulty=r['difficulty'],input_ids_sha256=textsha(json.dumps(ids,separators=(',',':')))))
    # Prevent a v1/v2 document collision from silently crossing calibration/test.
    calibration_docs={r['document_sha256'] for r in rows if r['split']=='calibration'}
    assert not any(r['document_sha256'] in calibration_docs for r in rows if r['split']=='test')
    with (out/'requests.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    metadata=[{k:v for k,v in r.items() if k not in ['input_ids','expected_answer']} for r in rows]
    (out/'selection.json').write_text(json.dumps(dict(seed=20261113,counts=counts,requests=metadata,
        sources=[dict(path=str(p),sha256=sha(p)) for p in [archive,v2,previous]],template_source=url,
        template_sha256=sha(out/'dataset2prompt.json'),lbv2_template_source=v2url,
        lbv2_template_sha256=sha(out/'lbv2-0shot.txt'),requests_sha256=sha(out/'requests.jsonl'),
        scope='Frozen inputs; exact token lengths and thinking/official LBv2 prompt parity still require protocol pilot; not collected results'),indent=2)+'\n')
    print(json.dumps(dict(counts=counts,total=len(rows),output=str(out))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
