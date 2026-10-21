"""Export measured CUDA-event intervals, never additive component approximations."""
import argparse,csv,hashlib,json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--batch',type=int,default=4);p.add_argument('--layer',type=int,default=12)
a=p.parse_args();folder=a.run_dir/'long-replay-v01/results';rows=[]
for mode in ['lru-b4096','data-ready-b4096','indexer-data-ready-b4096']:
 candidates=list(folder.glob(f'RULER128K-*-c{a.batch}-l{a.layer:02d}-*-{mode}.json'));assert len(candidates)==1,candidates
 f=candidates[0];d=json.loads(f.read_text());assert d['usable_comparison']
 for s in d['probe_component_spans']:
  rows.append(dict(case=f.stem,mode=mode,group=d['group'],batch=a.batch,layer=a.layer,component=s['component'],
                   start_us=s['start_us'],end_us=s['end_us'],span_us=s['stream_span_us'],probe_elapsed_us=d['probe_elapsed_us'],
                   graph_median_us=sorted(d['elapsed_us'])[2],source_file=str(f.resolve()),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as stream:
 w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(f'{len(rows)} actual event intervals exported')
