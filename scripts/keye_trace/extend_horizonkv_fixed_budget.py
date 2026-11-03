"""Generate distinct RULER inputs, capture real states, extend LxC data only.

Three disjoint cohorts per length. Mixed GPUs are used ONLY for input capture.
All actual stage measurements use the healthy A6000 and the calibrated ledger.
"""
import argparse,hashlib,json,os,signal,subprocess,time,urllib.request,shutil
from pathlib import Path
from transformers import AutoTokenizer
from horizonkv_single_runtime import REPO,GPU,INPUT_GPUS,save,update,notify
from continue_horizonkv_fixed_budget import execute,export
from run_horizonkv_motivation import run_wave

PYTHON=str(REPO/'.venv/bin/python')

def require_successful_capture(result):
    for row in result:
        reason=row['response'].get('meta_info',{}).get('finish_reason',{})
        if isinstance(reason,dict) and reason.get('type')=='abort':
            raise RuntimeError(f"Input capture rejected for {row['case_id']}: {reason.get('message',reason)}")


def freeze(root):
    source=root/'expanded-inputs';source.mkdir(exist_ok=True)
    manifest=source/'long-input-manifest.json'
    if manifest.exists():return source,json.loads(manifest.read_text())
    tokenizer=AutoTokenizer.from_pretrained('/Tan/model/Keye-VL-2.0-30B-A3B',trust_remote_code=True)
    groups=[];waves=[]
    for length,maxc in [(8192,64),(16384,64),(32768,32),(65536,32),(131072,16),(261632,8)]:
        generated=source/'ruler-source'/str(length)
        if not (generated/'generation_manifest.json').exists():
            cmd=[PYTHON,'scripts/keye_trace/prepare_ruler_gate_e0.py','--ruler-repo',str(REPO/'data/external/RULER'),
                 '--output-root',str(generated),'--model','/Tan/model/Keye-VL-2.0-30B-A3B',
                 '--lengths',str(length-256),'--tasks','niah_multikey_1','--num-samples',str(3*maxc),'--seed','20261014']
            execute(root,f'generate_ruler_{length}',cmd,os.environ.copy())
        raw=generated/str(length-256)/'niah_multikey_1/validation.jsonl'
        rows=[json.loads(s) for s in raw.read_text().splitlines()]
        seen=set()
        for cohort in range(3):
            group=dict(label=f'RULER{length//1024}K-Q{cohort}',waves=[],case_ids=[],actual_input_tokens=[],length_bucket=length,cohort=cohort)
            for i,row in enumerate(rows[cohort*maxc:(cohort+1)*maxc]):
                prompt=tokenizer.apply_chat_template([{'role':'user','content':row['input']}],tokenize=False,add_generation_prompt=True)
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
                assert digest not in seen;seen.add(digest)
                assert len(ids)+160<=262144,(length,len(ids))
                assert len(ids)<=length,(length,len(ids))
                ordinal=len(waves);case=f'fixed{length}-q{cohort}-{i:03d}'
                request=dict(case_id=case,input_ids=ids,task='niah_multikey_1',workload='retrieval',prompt_tokens=len(ids),
                    input_ids_sha256=digest,expected_answer=row['outputs'],dataset_revision='c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a')
                requests=source/'requests';requests.mkdir(exist_ok=True);save(requests/f'wave{ordinal:04d}.json',request)
                group['waves'].append(ordinal);group['case_ids'].append(case);group['actual_input_tokens'].append(len(ids))
                waves.append(dict(wave=ordinal,group=group['label'],request_file=str((requests/f'wave{ordinal:04d}.json').resolve()),length_bucket=length))
            assert len(group['waves'])==maxc;groups.append(group)
    data=dict(groups=groups,waves=waves,seed=20261014,capture_concurrency=1,decode_steps=128,
              note='Official RULER generator, disjoint real requests per cohort; retain EOS, no padding/repeated histories')
    save(manifest,data);return source,data


def capture(root,source,manifest,long=False):
    selected=[w for w in manifest['waves'] if (w['length_bucket']>131072)==long]
    for w in selected:
        existing=source/f"input-wave{w['wave']:03d}.json"
        if existing.exists():require_successful_capture(json.loads(existing.read_text()))
    if all((source/f"input-wave{w['wave']:03d}.json").exists() for w in selected):return
    port=31126
    cmd=[PYTHON,'-m','sglang.launch_server','--model-path','/Tan/model/Keye-VL-2.0-30B-A3B','--trust-remote-code',
         '--host','127.0.0.1','--port',str(port),'--tp-size','2','--attention-backend','keye_sa',
         '--context-length','262144' if long else '131328','--max-total-tokens','263000' if long else '140000',
         '--max-running-requests','1','--chunked-prefill-size','512' if long else '1024',
         '--mem-fraction-static','0.98' if long else '0.85','--disable-cuda-graph','--disable-custom-all-reduce',
         '--disable-radix-cache','--disable-overlap-schedule','--dtype','bfloat16','--random-seed','20261014']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(INPUT_GPUS),KEYE_SM80_DSA='1',KEYE_SM80_EXACT_TOPK='1',
             KEYE_LAYER_REPLAY_DIR=str(source/'fixtures'),KEYE_LAYER_REPLAY_LAYERS='1,12,24,36,47',KEYE_LAYER_REPLAY_STEPS='128',
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    save(source/('capture-256k-command.json' if long else 'capture-command.json'),dict(command=cmd,gpus=INPUT_GPUS,timing_evidence=False))
    server=None
    try:
        with (source/('capture-256k.log' if long else 'capture.log')).open('a') as log:
            server=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(root,phase='input_capture_model_loading',child_pid=server.pid,detail='A6000+L40S inputs only, never timing')
        url=f'http://127.0.0.1:{port}';deadline=time.time()+1200
        while time.time()<deadline:
            if server.poll() is not None:raise RuntimeError('Input server exited during load')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(3)
        else:raise TimeoutError('Input server startup timed out')
        for i,w in enumerate(selected):
            target=source/f"input-wave{w['wave']:03d}.json"
            if target.exists():continue
            if shutil.disk_usage(source).free<40*2**30:raise RuntimeError('Input capture reached 40GiB free-space floor; preserve all data')
            update(root,phase='collecting_fixed_budget_inputs',child_pid=server.pid,completed_waves=i,total_waves=len(selected),
                   detail=f"{w['group']}, distinct request wave {w['wave']}; input timings excluded")
            row=json.loads(Path(w['request_file']).read_text())
            result=run_wave(url,[row],f"replay__wave{w['wave']:03d}",160,3600)
            require_successful_capture(result)
            save(target,result)
        notify(f'HorizonKV固定预算：{"256K" if long else "8–128K"}真实输入采集完成；接下来仅健康A6000计时。')
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        update(root,child_pid=None)


def measure(root,source,manifest,long=False):
    groups=[g['label'] for g in manifest['groups'] if (g['length_bucket']>131072)==long]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    for rank in [0,1]:
        cmd=[PYTHON,'scripts/keye_trace/run_horizonkv_fixed_budget.py','--run-dir',str(root),'--source',str(source),
             '--rank',str(rank),'--groups',*groups,'--concurrencies','1','2','4','8','16','32','64',
             '--layers','1','12','24','36','47','--steps','0','32','64','96','127']
        execute(root,f'expanded_{"256k" if long else "main"}_rank{rank}',cmd,env)
        export(root);notify(f'HorizonKV固定预算扩展：rank{rank}的{"256K" if long else "主矩阵"}已测完；保留不可行/短输出缺失标记。')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();root=a.run_dir.resolve()
    source,manifest=freeze(root)
    capture(root,source,manifest);measure(root,source,manifest)
    try:
        capture(root,source,manifest,long=True);measure(root,source,manifest,long=True)
    except Exception as exc:
        save(root/'long-context-input-limitation.json',dict(error=str(exc),scope='Input collection/long extension; not an A6000 deployment limit'))
        export(root)
        update(root,status='failed',phase='main_grid_done_long_extension_failed',child_pid=None,detail=str(exc))
        notify('HorizonKV主矩阵已测，256K扩展受阻：'+str(exc));raise
    records=export(root)
    update(root,status='completed',phase='fixed_budget_data_completed',child_pid=None,completed_cases=len(records),detail='All queued data stages complete; no figures. See missing-step and capacity markers; TP2 rank-local only.')
    notify(f'HorizonKV固定预算数据队列完成：{len(records)}条有效层测量；无图；结果需结合有效step/容量边界审计。')

if __name__=='__main__':main()
