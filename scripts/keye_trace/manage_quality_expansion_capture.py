"""Wait for verified sources, freeze the registered deep cohort, then capture it."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from transformers import AutoTokenizer
from horizonkv_predictive_runtime import inventory,notify
from horizonkv_single_runtime import save,update

IDS=['GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822','GPU-79886749-8f57-63a5-8c8e-8a68a085565a']
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def choose(rows,cal,test):
    result=[]
    for split,n in [('calibration',cal),('test',test)]:
        pool=sorted([r for r in rows if r['split']==split and r['prompt_tokens']>2048],key=lambda r:r['rid'])
        result.extend(pool[:n])
    return result

def main(base):
    base=base.resolve();state=base/'expansion-state.json';notice=time.time()+1200
    required=['text-inputs-docsplit/selection.json','new-bfcl-inputs/selection.json','external-inputs/selection.json','ruler-generated/generation_manifest.json']
    while True:
        missing=[s for s in required if not (base/s).exists()]
        save(state,dict(status='running',phase='preparing_sources',missing=missing,checked_unix_s=time.time()))
        if not missing:break
        if time.time()>notice:
            generated=len(list((base/'ruler-generated').glob('*/*/validation.jsonl')))
            notify(f'HorizonKV扩展：BFCL桥接与详细pilot已完成；新数据准备中，RULER已生成{generated}/78格；待就绪：{missing}。')
            notice=time.time()+1200
        time.sleep(20)
    text=read(base/'text-inputs-docsplit/requests.jsonl');bfcl=read(base/'new-bfcl-inputs/requests.jsonl');external=read(base/'external-inputs/requests.jsonl')
    assert len(text)==1223 and len(bfcl)==80 and len(external)==120, (len(text),len(bfcl),len(external))
    generated=json.loads((base/'ruler-generated/generation_manifest.json').read_text())
    assert len(generated['generated'])==78
    tok=AutoTokenizer.from_pretrained('/Tan/model/Keye-VL-2.0-30B-A3B',trust_remote_code=True)
    ruler=[]
    for g in generated['generated']:
        p=Path(g['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==g['sha256']
        for i,r in enumerate(read(p)):
            # RULER's index is an answer offset; ordinal groups the same
            # generation seed across lengths, keeping it within one split.
            rid=f'predictive__ruler__{g["task"]}__{g["length"]}__{i:02d}'
            prompt=tok.apply_chat_template([dict(role='user',content=r['input'])],tokenize=False,add_generation_prompt=True)
            ids=tok.encode(prompt,add_special_tokens=False)
            assert 2048<len(ids)<131264,(rid,len(ids))
            ruler.append(dict(rid=rid,case_id=rid,dataset='RULER',group=g['task'],task=g['task'],
                length_bucket_upper=g['length'],prompt_tokens=len(ids),input_ids=ids,expected_answer=r['outputs'],
                source_instance=f'{g["task"]}-seed20261113-ordinal{i}',split='calibration' if i<4 else 'test',
                input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    all_rows=text+bfcl+external+ruler
    assert len({r['rid'] for r in all_rows})==len(all_rows)
    assert len({r['input_ids_sha256'] for r in all_rows})==len(all_rows)
    deep=[];counts={}
    for task in sorted({r['task'] for r in text if r['dataset']=='LongBench'}):
        sample=choose([r for r in text if r['dataset']=='LongBench' and r['task']==task],2,6);deep+=sample;counts[task]=len(sample)
    for task in sorted({r['task'] for r in bfcl}):deep+=choose([r for r in bfcl if r['task']==task],2,4)
    v2=sorted([r for r in text if r['dataset']=='LongBench-v2'],key=lambda r:(r['task'],r['case_id']))
    # Round-robin domains, fixed before seeing new model outcomes.
    domains={}
    for r in v2:domains.setdefault(r['task'],[]).append(r)
    chosen=[]
    while len(chosen)<30:
        for domain in sorted(domains):
            if domains[domain]:chosen.append(domains[domain].pop(0))
            if len(chosen)==30:break
    deep+=chosen
    for dataset in ['MRCR','GraphWalks']:deep+=choose([r for r in external if r['dataset']==dataset],4,8)
    for task in sorted({r['task'] for r in ruler}):
        for length in [4096,8192,16384,32768,65536,131072]:deep+=choose([r for r in ruler if r['task']==task and r['length_bucket_upper']==length],1,1)
    for name,rows in [('all-inputs',all_rows),('deep-inputs',deep)]:
        path=base/(name+'.jsonl');assert not path.exists()
        with path.open('w') as stream:
            for r in rows:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    save(base/'cohort-audit.json',dict(total=len(all_rows),calibration=sum(r['split']=='calibration' for r in all_rows),test=sum(r['split']=='test' for r in all_rows),deep=len(deep),longbench_deep_counts=counts,
        all_inputs_sha256=hashlib.sha256((base/'all-inputs.jsonl').read_bytes()).hexdigest(),deep_inputs_sha256=hashlib.sha256((base/'deep-inputs.jsonl').read_bytes()).hexdigest(),policy='Frozen without outcomes; sparse-eligible deep subset; shortage explicitly recorded; no fitting to test observations'))
    root=base/'deep64';root.mkdir();config=dict(gpu_ids=IDS,tp_size=1,pp_size=2,pci_bus_tags=[':51:00',':C3:00',':c3:00'],seed=20261113,steps=64)
    save(root/'run-config.json',config);save(root/'environment.json',dict(gpus=inventory(IDS,require_idle=True),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    update(root,status='running',phase='starting_deep_capture',completed_requests=0,total_requests=len(deep),child_pid=None,detail='Approved deep shadow cohort; task generation quality remains pending')
    command=[sys.executable,'scripts/keye_trace/horizonkv_predictive_runtime.py','--run-dir',str(root),'--requests-file',str(base/'deep-inputs.jsonl'),'--gpu-ids',*IDS,'--tp-size','1','--pp-size','2','--steps','64','--seed','20261113','--extra-metrics','--compress-trace']
    with (root/'monitor.log').open('w') as log:monitor=subprocess.Popen([sys.executable,'scripts/keye_trace/horizonkv_predictive_runtime.py','--run-dir',str(root),'--monitor'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    notify(f'HorizonKV扩展：来源已冻结{len(all_rows)}请求，详细影子队列{len(deep)}/330条启动，64Decode步、额外Attention/表示指标、无损压缩trace。正式任务质量尚待实现和运行。')
    save(state,dict(status='running',phase='deep_capture',total=len(deep),monitor_pid=monitor.pid,checked_unix_s=time.time()))
    with (root/'controller.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(state,dict(status='stage_completed',phase='deep_capture_done_task_quality_pending',checked_unix_s=time.time()))

if __name__=='__main__':
    base=Path(sys.argv[1])
    try:main(base)
    except BaseException as exc:
        save(base/'expansion-manager-failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        notify('HorizonKV扩展队列管理器停止：'+str(exc)+'；保留数据等待修复，未宣布任务质量完成。')
        raise
