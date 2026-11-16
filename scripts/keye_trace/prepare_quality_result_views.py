"""Build immutable analysis views of completed requests across a paused capture."""
import argparse,hashlib,json,shutil
from collections import Counter
from pathlib import Path


def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(base,resume,out):
    old=base/'deep64';check=json.loads((old/'pause-checkpoint.json').read_text())
    assert sha(old/'events/manifest.jsonl')==check['manifest_sha256']
    state=json.loads((resume/'state.json').read_text());assert state['status']=='completed' and state['completed_requests']==278
    inputs=read(base/'deep-inputs.jsonl');meta={r['rid']:r for r in inputs};done=set(check['completed_request_ids']);new={r['rid'] for r in read(resume/'requests.jsonl')}
    assert len(done)==52 and len(new)==278 and not done&new and done|new==set(meta)
    assert sha(base/'deep-inputs.jsonl')==json.loads((base/'cohort-audit.json').read_text())['deep_inputs_sha256']
    out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for source,allowed in [(old,done),(resume,new)]:
        chunks=read(source/'events/manifest.jsonl');selected=[c for c in chunks if c['request_id'] in allowed]
        assert {(c['request_id'],c['target_layer_id']) for c in selected}=={(rid,l) for rid in allowed for l in range(1,48)}
        assert len(selected)==len(allowed)*47
        source_rows=read(source/'requests.jsonl')
        for i,r in enumerate(source_rows):
            if r['rid'] not in allowed:continue
            response=json.loads((source/f'response-{i:02d}.json').read_text())
            assert len(response['output_ids'])==65
            assert response['meta_info'].get('finish_reason',{}).get('type')!='abort'
            assert r['input_ids']==meta[r['rid']]['input_ids']
        for c in selected:
            p=source/'events'/c['file'];assert p.stat().st_size==c['bytes'] and c['num_steps']==64
        sources.append((source,allowed,selected))
    for name,source_groups,ids in [('old52',sources[:1],done),('combined330',sources,done|new)]:
        root=out/name;events=root/'events';events.mkdir(parents=True)
        rows=[r for r in inputs if r['rid'] in ids]
        (root/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        save(root/'selection.json',dict(requests=[{k:v for k,v in r.items() if k!='input_ids'} for r in rows],policy='Frozen input union; exclude uncommitted old in-flight request',seed=20261113))
        shutil.copyfile(resume/'run-config.json',root/'run-config.json')
        manifest=[];attention_rows=0
        for source,allowed,chunks in source_groups:
            for c in chunks:
                (events/c['file']).symlink_to((source/'events'/c['file']).resolve());manifest.append(c)
            for f in sorted((source/'events').glob('attention-stage-layer*.jsonl')):
                target=events/f.name
                with target.open('a') as writer:
                    for line in f.open():
                        r=json.loads(line)
                        if r['request_id'] in allowed:
                            writer.write(line);attention_rows+=1
        (events/'manifest.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in manifest))
        assert attention_rows==len(ids)*47*64*3
        save(root/'view-audit.json',dict(requests=len(ids),chunks=len(manifest),attention_rows=attention_rows,
            source_runs=[dict(path=str(s),requests=len(a),manifest_sha256=sha(s/'events/manifest.jsonl'),server=json.loads((s/'server-command.json').read_text())) for s,a,c in source_groups],
            excluded_old_request_ids=sorted(set(c['request_id'] for c in read(old/'events/manifest.jsonl'))-done)))
    print(json.dumps(dict(output=str(out),requests=330,old_complete=52,resumed=278,chunks=330*47,attention_rows=330*47*64*3)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.base.resolve(),a.resume.resolve(),a.output.resolve())
