"""Freeze complete, untruncated MRCR/GraphWalks prompts within Keye's budget."""
import argparse,hashlib,json,random
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer


def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def main(out):
    out.mkdir(parents=True,exist_ok=False);rng=random.Random(20261113)
    tok=AutoTokenizer.from_pretrained('/Tan/model/Keye-VL-2.0-30B-A3B',trust_remote_code=True)
    cohorts=[]
    for needles in [2,4,8]:
        p=Path(f'/Tan/dataset/openai-mrcr/{needles}needle/{needles}needle_0.parquet')
        d=pd.read_parquet(p);d=d[d.n_chars.between(10000,500000)]
        records=d.to_dict('records');rng.shuffle(records)
        cohorts.append((f'MRCR-{needles}',records,7 if needles!=8 else 6))
    p=Path('/Tan/dataset/openai-graphwalks/graphwalks_128k_and_shorter.parquet');d=pd.read_parquet(p)
    for kind in ['parents','bfs']:
        records=d[(d.problem_type==kind)&d.prompt_chars.between(10000,500000)].to_dict('records');rng.shuffle(records)
        cohorts.append(('GraphWalks-'+kind,records,10))
    rows=[];seen=set();documents={};counts={}
    for name,records,cal_count in cohorts:
        wanted=20 if name.startswith('MRCR') else 30;count=0
        for r in records:
            if name.startswith('MRCR'):
                messages=json.loads(r['prompt']);doc=digest(json.dumps(messages[:-1],sort_keys=True))
                answer=r['answer'];extra=dict(random_string_to_prepend=r['random_string_to_prepend'],n_needles=r['n_needles'])
            else:
                messages=[dict(role='user',content=r['prompt'])];doc=digest(r['prompt'].rsplit('Operation:',1)[0])
                answer=list(r['answer_nodes']);extra=dict(problem_type=r['problem_type'])
            identity=digest(r['prompt'])
            if identity in seen:continue
            rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=tok.encode(rendered,add_special_tokens=False)
            if not 2048<len(ids)<=120000:continue
            split='calibration' if count<cal_count else 'test'
            if doc in documents and documents[doc]!=split:continue
            seen.add(identity);documents[doc]=split
            rows.append(dict(rid='predictive__external__'+identity[:20],case_id=identity,dataset=name.split('-')[0],
                task=name,group=name,split=split,document_sha256=doc,input_ids=ids,prompt_tokens=len(ids),
                length_bucket_upper=next(c for c in [4096,8192,16384,32768,65536,131072] if len(ids)<=c),
                input_ids_sha256=digest(json.dumps(ids,separators=(',',':'))),expected_answer=answer,truncated=False,**extra))
            count+=1
            print(json.dumps(dict(cohort=name,count=count,tokens=len(ids))),flush=True)
            if count==wanted:break
        counts[name]=count
    (out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'selection.json').write_text(json.dumps(dict(seed=20261113,counts=counts,
        protocol='Public MRCR (version not equated to IndexCache MRCR v2); only shard0 per needle; GraphWalks short shard; full prompts, no truncation',
        requests=[{k:v for k,v in r.items() if k not in ['input_ids','expected_answer']} for r in rows]),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
