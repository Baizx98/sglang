"""232K extension: capture one request, validate real replay, then expand."""
import argparse,json,os,traceback
from pathlib import Path
from extend_horizonkv_fixed_budget import freeze,capture,PYTHON
from continue_horizonkv_fixed_budget import execute,export
from horizonkv_single_runtime import GPU,INPUT_GPUS,save,update,notify
from run_horizonkv_fixed_budget import check_gpu_isolation


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();root=a.run_dir.resolve();root.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    try:
        check_gpu_isolation(root,'232K startup','before_capture')
        source,manifest=freeze(root,pools=[(232*1024,8)])
        pilot=dict(manifest,waves=manifest['waves'][:1])
        capture(root,source,pilot,long=True,capture_name='232k-pilot',prefill_chunk=128,score_chunk=8)
        # Release the mixed-device model before any A6000 layer replay.
        for rank in [0,1]:
            gate=root/f'gate-r{rank}';gate.mkdir(exist_ok=True)
            cmd=[PYTHON,'scripts/keye_trace/run_horizonkv_fixed_budget.py','--run-dir',str(gate),
                 '--source',str(source),'--groups',manifest['groups'][0]['label'],'--rank',str(rank),
                 '--layers','12','--concurrencies','1','--steps','0','--validate-only']
            execute(root,f'gate_rank{rank}',cmd,env)
            checks=[json.loads(p.read_text()) for p in (gate/'results').glob('*.json')]
            assert len(checks)==1 and checks[0].get('timed'),('Missing real pilot replay',rank)
            assert all(v['topk_equal'] for v in checks[0]['results'])
        save(root/'single-request-gate.json',dict(passed=True,groups=[manifest['groups'][0]['label']],
             ranks=[0,1],layer=12,step=0,scope='Real non-aborted capture and same-A6000 replay gates; not E2E quality'))
        notify('HorizonKV 232K：首条真实请求及两个rank的层12回放校验通过；继续采集其余23条。')
        check_gpu_isolation(root,'232K full capture','before_capture')
        capture(root,source,manifest,long=True,capture_name='232k',prefill_chunk=128,score_chunk=8)
        groups=[g['label'] for g in manifest['groups']]
        for rank in [0,1]:
            cmd=[PYTHON,'scripts/keye_trace/run_horizonkv_fixed_budget.py','--run-dir',str(root),
                 '--source',str(source),'--groups',*groups,'--rank',str(rank),
                 '--layers','1','12','24','36','47','--concurrencies','1','2','4','8','16','32','64',
                 '--steps','0','32','64','96','127']
            execute(root,f'232k_rank{rank}',cmd,env);export(root)
            notify(f'HorizonKV 232K：rank{rank}扫描结束，继续核验有效点与容量/短输出标记。')
        rows=export(root)
        assert rows and all(r['topk_equal'] for r in rows)
        markers=[json.loads(p.read_text()) for p in (root/'results').glob('*.json')]
        captured=sum((source/f"input-wave{w['wave']:03d}.json").exists() for w in manifest['waves'])
        save(root/'completion-audit.json',dict(valid_timed_rows=len(rows),input_captures=captured,
             missing_step_markers=sum(d.get('capacity_status')=='missing-common-decode-step' for d in markers),
             scope='Single A6000 TP2 rank-local, mixed inputs only; no figures or E2E results'))
        update(root,status='completed',phase='232k_data_completed',child_pid=None,
               completed_cases=len(rows),total_cases=len(markers),completed_waves=captured,total_waves=24,
               detail='232K data queue finished; audit valid steps and capacity markers before claims')
        notify(f'HorizonKV 232K扩展结束：{captured}/24输入、{len(rows)}个有效层计时；仅数据，待审计。')
    except Exception as exc:
        save(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        export(root);update(root,status='failed',phase='232k_extension_failed',child_pid=None,detail=str(exc))
        notify('HorizonKV 232K扩展停止：'+str(exc));raise


if __name__=='__main__':main()
