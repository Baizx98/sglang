"""Resume the approved frozen cohort in a new directory, preserving paused traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from horizonkv_predictive_runtime import inventory,notify
from horizonkv_single_runtime import save,update


def prepare(base,root):
    old=base/'deep64';checkpoint=json.loads((old/'pause-checkpoint.json').read_text())
    assert checkpoint['status']=='paused'
    manifest=old/'events/manifest.jsonl'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==checkpoint['manifest_sha256']
    chunks=[json.loads(s) for s in manifest.read_text().splitlines()]
    lookup={}
    for chunk in chunks:
        key=(chunk['request_id'],chunk['target_layer_id'])
        assert key not in lookup,key
        lookup[key]=chunk
    done=checkpoint['completed_request_ids'];assert len(done)==len(set(done))==52
    for i,rid in enumerate(done):
        response=json.loads((old/f'response-{i:02d}.json').read_text())
        assert len(response['output_ids'])==65
        for layer in range(1,48):
            item=lookup[(rid,layer)];p=old/'events'/item['file']
            assert item['num_steps']==64 and p.stat().st_size==item['bytes']
    source=base/'deep-inputs.jsonl'
    audit=json.loads((base/'cohort-audit.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['deep_inputs_sha256']
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    assert len(rows)==330 and {r['rid'] for r in rows}==set(done)|set(checkpoint['remaining_request_ids'])
    remaining=[r for r in rows if r['rid'] not in set(done)]
    assert len(remaining)==278 and [r['rid'] for r in remaining]==checkpoint['remaining_request_ids']
    root.mkdir(parents=True,exist_ok=False)
    inputs=root/'frozen-remaining-inputs.jsonl'
    inputs.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in remaining))
    config=json.loads((old/'run-config.json').read_text())
    config.update(prior_completed_requests=52,overall_total_requests=330,
        resume_from=str(old.resolve()),resume_authorized_date='2026-11-16',extra_metrics=True,compress_trace=True)
    save(root/'run-config.json',config)
    save(root/'resume-audit.json',dict(source=str(source.resolve()),source_sha256=audit['deep_inputs_sha256'],
        checkpoint=str((old/'pause-checkpoint.json').resolve()),completed_verified=52,remaining=278,
        partial_prior_requests=checkpoint['partial_chunk_counts'],remaining_inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        policy='Keep completed old requests once; recapture all uncommitted/in-flight requests; no paused artifact overwritten'))
    return inputs,config


def main(base,root):
    devices=inventory(['GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822','GPU-79886749-8f57-63a5-8c8e-8a68a085565a'],require_idle=True)
    inputs,config=prepare(base,root)
    save(root/'environment.json',dict(gpus=devices,checked_unix_s=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_environment=str((base/'deep64/environment.json').resolve()),scope='Same PP2/TP1 BF16 configuration; fresh health and environment inventory'))
    (root/'kernel-before.log').write_text(subprocess.check_output(['dmesg'],text=True))
    update(root,status='running',phase='resume_starting',completed_requests=0,total_requests=278,child_pid=None,
        detail='User authorized resume on 2026-11-16; prior 52/330 preserved; remaining 278 in a new directory')
    cmd=[sys.executable,'scripts/keye_trace/horizonkv_predictive_runtime.py','--run-dir',str(root),
        '--requests-file',str(inputs),'--gpu-ids',*config['gpu_ids'],'--tp-size','1','--pp-size','2',
        '--steps','64','--seed','20261113','--extra-metrics','--compress-trace']
    with (root/'controller.log').open('w') as log:
        controller=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    with (root/'monitor.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,'scripts/keye_trace/horizonkv_predictive_runtime.py','--run-dir',str(root),'--monitor'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    save(root/'controller-pids.json',dict(controller=controller.pid,monitor=monitor.pid))
    save(base/'expansion-state.json',dict(status='running',phase='resumed_deep_capture',prior_completed_requests=52,
        current_run=str(root),remaining_requests=278,total_requests=330,controller_pid=controller.pid,monitor_pid=monitor.pid,checked_unix_s=time.time()))
    if (base/'PAUSED').exists():
        (base/'PAUSED').rename(base/('PAUSED-resumed-'+root.name))
    sent=notify('HorizonKV恢复实验：健康A6000+L40S均空闲且Recovery=None，52/330检查点通过；新目录续采剩余278条，64步Decode、多指标和无损压缩不变，约每20分钟通知。')
    print(json.dumps(dict(root=str(root),controller=controller.pid,monitor=monitor.pid,start_notification_sent=sent)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();main(args.base.resolve(),args.run_dir.resolve())
