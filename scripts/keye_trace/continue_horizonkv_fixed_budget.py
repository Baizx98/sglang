"""Data-only fixed-budget queue: verified pilot, ledger checks, then extensions."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from horizonkv_single_runtime import REPO,GPU,save,update,notify


def execute(root,name,command,env):
    save(root/(name+'-launch.json'),dict(command=command,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),started_unix_s=time.time()))
    with (root/(name+'.log')).open('a') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(root,child_pid=child.pid,phase=name,detail=' '.join(command[2:]))
        code=child.wait()
    if code:raise RuntimeError(f'{name} exited {code}; see {root/name}.log')


def export(root):
    import csv
    records=[]
    for p in sorted((root/'results').glob('*.json')):
        d=json.loads(p.read_text())
        for v in d.get('results',[]):
            record={k:d[k] for k in ['case','group','concurrency','layer','rank','step']}
            record.update({k:v[k] for k in ['mode','cache_tokens_per_request','selected','miss_tokens','host_read_payload_bytes','host_write_payload_bytes','elapsed_median_us','front_probe_us','front_probe_share','indexer_probe_us','postselect_kv_probe_us','incremental_peak_bytes']})
            record['topk_equal']=v['topk_equal'];record['graph_output_bitwise_equal']=v['graph_output_bitwise_equal']
            record['gpu_work_us']=v['gpu_profile']['gpu_work_total_us']
            record['source_file']=str(p.resolve());records.append(record)
    if records:
        with (root/'pilot-data.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();root=a.run_dir.resolve()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    try:
        for rank in [0,1]:
            cmd=[str(REPO/'.venv/bin/python'),'scripts/keye_trace/run_horizonkv_fixed_budget.py','--run-dir',str(root),'--rank',str(rank),
                 '--groups','RULER32K','RULER64K','RULER128K','--concurrencies','1','4','8','16',
                 '--layers','1','12','24','36','47','--steps','32','39']
            execute(root,f'pilot_rank{rank}',cmd,env)
            records=export(root)
            notify(f'HorizonKV固定预算：rank{rank} pilot完成，累计{len(records)}组真实数据；未绘图，继续核验workspace和更多L×C。')
        records=export(root)
        assert records and all(r['topk_equal'] for r in records)
        ledger=json.loads((root/'budget.json').read_text())
        violations=[]
        for r in records:
            d=json.loads(Path(r['source_file']).read_text());tokens=sum(d['lengths'])
            envelope=ledger['workspace_fixed_bytes']+tokens*ledger['workspace_bytes_per_history_token']+d['concurrency']*2048*1024
            if r['incremental_peak_bytes']>envelope:violations.append(dict(case=r['case'],peak=r['incremental_peak_bytes'],envelope=envelope))
        save(root/'pilot-audit.json',dict(records=len(records),workspace_envelope_violations=violations,
             topk_all_equal=True,scope='TP2-shaped single-rank layer replay, no communication or plots'))
        if violations:raise RuntimeError('Provisional workspace envelope exceeded; revise ledger before expanding')
        # Mark the first stage explicitly, then run the remaining authorized matrix.
        update(root,status='running',phase='pilot_completed_extension_pending',child_pid=None,
               completed_cases=len(records),detail='Pilot measured; 8K/16K and C>16 require additional distinct real input capture; extension runner will follow.')
        notify(f'HorizonKV固定预算pilot完成：{len(records)}组；workspace预算核验通过。继续生成/采集新输入和扩展矩阵。')
        execute(root,'expanded_data_queue',[str(REPO/'.venv/bin/python'),'scripts/keye_trace/extend_horizonkv_fixed_budget.py','--run-dir',str(root)],env)
    except Exception as exc:
        save(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        export(root);update(root,status='failed',phase='fixed_budget_queue_failed',child_pid=None,detail=str(exc));notify('HorizonKV固定预算实验异常：'+str(exc));raise

if __name__=='__main__':main()
