"""Freeze new BFCL trajectories, excluding all old trajectory base suffixes."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from transformers import AutoTokenizer
from run_bfcl_segmented import build_case_requests,load_category_entries,literal_assignment,DEFAULT_CATEGORIES,git_revision


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    package=Path('data/external/gorilla/berkeley-function-call-leaderboard').resolve()
    old=Path('data/cross-layer-lookahead/2026-08-04_2005_phase-a-v01/formal/selection.json')
    previous=json.loads(old.read_text())
    used={r['trajectory_id'].rsplit('_',1)[-1] for r in previous['selected']}
    tokenizer=AutoTokenizer.from_pretrained('/Tan/model/Keye-VL-2.0-30B-A3B',trust_remote_code=True)
    extension=literal_assignment(package/'bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py','FILE_CONTENT_EXTENSION')
    rng=random.Random(20261113);selected=[];exclusions=[];snapshots=[]
    for category in DEFAULT_CATEGORIES:
        cases,answers=load_category_entries(package/'bfcl_eval/data',category);rng.shuffle(cases);count=0
        for case in cases:
            suffix=case['id'].rsplit('_',1)[-1]
            if suffix in used or len(case['question'])<4:continue
            requests,segments,snapshot=build_case_requests(tokenizer,package,category,case,answers[case['id']],extension,limit_rounds=None)
            eligible=[r for r in requests if r['ground_truth_calls'] and 2048<len(r['input_ids'])<=120000]
            if not eligible:
                exclusions.append(dict(id=case['id'],reason='no_sparse_tool_bearing_prompt_in_input_budget'));continue
            r=max(eligible,key=lambda r:(r['prompt_len'],r['round_id']))
            selected.append(dict(r,rid='predictive__new__'+r['rid'],case_id=case['id'],dataset='BFCL',task=category,
                group='BFCL',split='calibration' if count<5 else 'test',prompt_tokens=r['prompt_len'],
                length_bucket_upper=next(c for c in [4096,8192,16384,32768,65536,131072] if r['prompt_len']<=c),
                input_ids_sha256=hashlib.sha256(json.dumps(r['input_ids'],separators=(',',':')).encode()).hexdigest()))
            used.add(suffix);snapshots.append(snapshot);count+=1
            print(json.dumps(dict(category=category,count=count,prompt_tokens=r['prompt_len'])),flush=True)
            if count==20:break
        if count!=20:raise RuntimeError(f'{category}: only {count}/20 eligible unique trajectories')
    (out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    (out/'selection.json').write_text(json.dumps(dict(seed=20261113,source_commit=git_revision(package),
        historical_selection_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),exclusions=exclusions,
        requests=[{k:v for k,v in r.items() if k!='input_ids'} for r in selected]),ensure_ascii=False,indent=2)+'\n')
    (out/'dataset-snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
