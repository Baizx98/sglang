"""Finish the running real-input collection, then validate and measure on A6000."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from horizonkv_single_runtime import GPU, REPO, notify, save, update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--batches",nargs="+",type=int,default=[1,4])
    args=parser.parse_args()
    root=args.run_dir.resolve()
    pid=json.loads((root/"collector-launch.json").read_text())["pid"]
    child=None
    try:
        while True:
            path=Path(f"/proc/{pid}/cmdline")
            if not path.exists() or b"collect_horizonkv_long_inputs.py" not in path.read_bytes():
                break
            time.sleep(5)
        assert (root/"long-input-complete.json").exists(), "input collector did not finish"
        python=str(REPO/".venv/bin/python")
        jobs=[("frozen_residency_kernel_check",[python,"scripts/keye_trace/test_horizonkv_layer_replay.py"],GPU),
              ("long_context_a6000_replay",[python,"scripts/keye_trace/run_horizonkv_long_replay.py","--run-dir",str(root),"--batches",*map(str,args.batches)],GPU),
              ("long_context_export",[python,"scripts/keye_trace/export_horizonkv_long_replay.py","--run-dir",str(root)],""),
              ("long_context_summary",[python,"scripts/keye_trace/summarize_horizonkv_long_results.py","--run-dir",str(root)],"")]
        notes=REPO.parent/"my_research_note/projects/horizonkv/branches/main/data"
        pipeline=root/"report-long-replay-v01/pipeline-events.csv"
        jobs += [("pipeline_export",[python,"scripts/keye_trace/export_horizonkv_pipeline.py","--run-dir",str(root),"--output",str(pipeline),"--batch",str(args.batches[-1])],""),
                 ("figure_preview",[python,str(REPO.parent/"horizonkv-paper/scripts/plot_motivation.py"),
                    "--points",str(root/"report-long-replay-v01/point-components.csv"),
                    "--moe",str(notes/"2026-10-21_1050_moe-tile-audit_v01_reviewed.csv"),
                    "--pipeline",str(pipeline),"--plot-data",str(root/"report-long-replay-v01/figure-data.csv"),
                    "--output-dir",str(root/"figure-preview"),"--batch",str(args.batches[-1]),"--core-only"],"")]
        save(root/"continuation-jobs.json",[{"phase":p,"command":c} for p,c,_ in jobs])
        for phase,command,devices in jobs:
            update(root,phase=phase,detail="只用健康A6000做实际计时；控制组不作为在线系统或方法收益。")
            notify(f"HorizonKV长上下文进入阶段：{phase}；输入已采集，故障卡未参与。")
            with (root/(phase+".log")).open("w") as log:
                child=subprocess.Popen(command,cwd=REPO,
                      env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONUNBUFFERED="1"),
                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(root,child_pid=child.pid)
            save(root/"continuation-current.json",dict(phase=phase,pid=child.pid))
            if child.wait():raise RuntimeError(f"{phase} failed; see log")
        audit=json.loads((root/"processed-long-replay-v01/audit.json").read_text())
        complete=json.loads((root/"long-replay-v01/complete.json").read_text())
        assert audit["cases"]==complete["cases"]==complete["expected_cases"]
        update(root,status="completed",phase="long_context_completed",child_pid=None,
               detail=f"{audit['cases']}组长上下文实际层测量与导出完成；待检查趋势和基线局限，不自动写入论文结论。")
        notify(f"HorizonKV长上下文完成：{audit['cases']}组，batch={args.batches}、5个采样层；原始时间线与直接表已保存，待审计结论。")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        update(root,status="failed",phase="long_context_failed",child_pid=None,detail=f"{type(exc).__name__}: {exc}")
        notify(f"HorizonKV长上下文队列停止：{type(exc).__name__}: {exc}；已完成数据保留，需核查日志。")
        raise


if __name__=="__main__":
    main()
