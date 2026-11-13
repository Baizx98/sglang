"""Run the same-state 4.2 study; monitor verified files and notify every 20 min."""
import argparse
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

from horizonkv_a6000 import GPU_IDS, notify as _notify
from horizonkv_single_runtime import REPO, save, update

notify=partial(_notify,title="HorizonKV 4.2")


def inventory(gpu_ids, require_idle=False):
    result=subprocess.check_output(['nvidia-smi','-i',','.join(gpu_ids),
        '--query-gpu=name,uuid,memory.total,memory.free,pci.bus_id,driver_version,gpu_recovery_action',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    fields=['name','uuid','memory_mib','free_mib','pci_bus','driver','recovery_action']
    rows=[dict(zip(fields,[v.strip() for v in row])) for row in csv.reader(result.splitlines())]
    if {r['uuid'] for r in rows}!=set(gpu_ids) or any(r['recovery_action']!='None' for r in rows):
        raise RuntimeError('Selected GPUs are missing or require recovery')
    if require_idle:
        apps=subprocess.check_output(['nvidia-smi','-i',','.join(gpu_ids),
            '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
        if apps.strip():raise RuntimeError('A selected GPU has another compute process')
    return rows


def bounded_inventory(root, gpu_ids):
    # A stuck GSP can block NVML in uninterruptible driver sleep. Do not let
    # progress notifications wait indefinitely for a subprocess to reap.
    with (root/'monitor-gpu-probe.txt').open('w') as log:
        proc = subprocess.Popen(['nvidia-smi', '-i', ','.join(gpu_ids), '--query-gpu=name,uuid,memory.used,temperature.gpu,power.draw,utilization.gpu,gpu_recovery_action',
            '--format=csv'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return dict(timed_out=True, pid=proc.pid)
    return dict(exit_code=proc.returncode, output=(root/'monitor-gpu-probe.txt').read_text())


def monitor(root):
    config=json.loads((root/'run-config.json').read_text()) if (root/'run-config.json').exists() else dict(gpu_ids=GPU_IDS)
    gpu_ids=config['gpu_ids']
    # Exclude the faulty, unselected card's later driver events.
    selected_bus_tags=config.get('pci_bus_tags',[])
    next_notice = time.time() + 1200
    known_xids = set(subprocess.check_output(['dmesg'], text=True).splitlines())
    while True:
        state = json.loads((root / 'state.json').read_text())
        terminal = state.get('status') in {'completed', 'failed', 'interrupted'}
        failure = None
        server_log = root/'server.log'
        if not terminal and server_log.exists():
            with server_log.open('rb') as handle:
                handle.seek(max(0, server_log.stat().st_size-32768))
                recent = handle.read().lower()
            if any(value in recent for value in [b'scatter gather kernel index out of bounds',
                    b'device-side assert triggered', b'cuda error:', b'cuda out of memory']):
                failure = 'CUDA failure in owned server log'
        gpu_probe = bounded_inventory(root, gpu_ids)
        with (root/'gpu-monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(checked_unix_s=time.time(), **gpu_probe))+'\n')
        if not terminal:
            if gpu_probe.get('timed_out') or gpu_probe.get('exit_code') != 0:
                failure = 'GPU status query failed or timed out'
            else:
                for gpu in csv.DictReader(gpu_probe['output'].splitlines(), skipinitialspace=True):
                    if gpu['uuid'] in gpu_ids and gpu['gpu_recovery_action'] != 'None':
                        failure = f"{gpu['uuid']} Recovery={gpu['gpu_recovery_action']}"
            lines = subprocess.check_output(['dmesg'], text=True).splitlines()
            new_xids = [line for line in lines if 'NVRM: Xid' in line and line not in known_xids
                and (not selected_bus_tags or any(tag in line for tag in selected_bus_tags))]
            if new_xids:
                failure = 'New NVIDIA Xid: '+new_xids[-1]
            known_xids.update(lines)
        if failure:
            pid=state.get('child_pid')
            proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and b'sglang.launch_server' in proc.read_bytes():
                # Stop only the recorded service group; no GPU reset or retry.
                os.killpg(pid,signal.SIGKILL)
            update(root,status='failed',phase='error_watchdog',child_pid=None,detail=failure)
            save(root/'watchdog-error.json', dict(checked_unix_s=time.time(),failure=failure,gpu_probe=gpu_probe))
            notify('HorizonKV 4.2：检测到错误，已停止本轮服务，不重试、不复位。'+failure)
            return
        if terminal or time.time() >= next_notice:
            metrics = root / 'events' / 'manifest.jsonl'
            chunks = sum(1 for _ in metrics.open()) if metrics.exists() else 0
            record = dict(checked_unix_s=time.time(), state=state, chunks=chunks)
            record['gpus'] = gpu_probe
            pid = state.get('child_pid')
            record['server_alive'] = bool(pid and Path(f'/proc/{pid}').exists())
            record['ntfy_sent'] = notify(
                f"HorizonKV 4.2：{state.get('status')}；{state.get('phase')}；"
                f"请求 {state.get('completed_requests', 0)}/{state.get('total_requests', '?')}；"
                f"已落盘层Trace {chunks}；{state.get('detail', '')}")
            save(root / 'monitor-status.json', record)
            with (root / 'monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            next_notice = time.time() + 1200
        if terminal:
            return
        time.sleep(10)


def freeze(root):
    source = REPO / 'data/horizonkv-motivation/2026-10-27_fixed-budget-x16-v01/expanded-inputs/long-input-manifest.json'
    manifest = json.loads(source.read_text())
    selected = []
    for length in [8192, 16384, 32768, 65536, 131072]:
        for cohort in range(3):
            wave = next(w for w in manifest['waves'] if w['group'] == f'RULER{length//1024}K-Q{cohort}')
            row = json.loads(Path(wave['request_file']).read_text())
            row.update(group=f'RULER{length//1024}K', length_bucket_upper=length,
                       source_file=wave['request_file'])
            selected.append(row)
    source2 = REPO / 'data/horizonkv-motivation/2026-10-14_pilot-v01/workloads-v02/requests.jsonl'
    rows = [json.loads(s) for s in source2.read_text().splitlines()]
    for task in ['Single-Document QA', 'gov_report', 'repobench-p']:
        group = sorted([r for r in rows if r['task'] == task and r['length_bucket_upper'] == 32768], key=lambda r: r['case_id'])[:3]
        assert len(group) == 3
        selected.extend(dict(r, group=task, source_file=str(source2)) for r in group)
    # Short/long RULER requests alternate within each length; outcomes never enter selection.
    assert len(selected) == 24
    seen = set()
    for i, row in enumerate(selected):
        digest = hashlib.sha256(json.dumps(row['input_ids'], separators=(',', ':')).encode()).hexdigest()
        assert digest not in seen
        seen.add(digest)
        assert len(row['input_ids']) == row['prompt_tokens']
        row.update(rid=f'predictive__{i:02d}__{row["case_id"]}', input_ids_sha256=digest)
    (root / 'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    save(root / 'selection.json', dict(seed=20261112, policy='First request per disjoint RULER cohort, five lengths; first three case IDs per QA/summary/code 32K bucket. All selected before new outcomes.',
        source_files=[str(source), str(source2)], requests=[{k:v for k,v in r.items() if k not in {'input_ids'}} for r in selected]))
    return selected


def post(url, payload, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            # Administrative endpoints such as flush_cache return plain text.
            return raw.decode()


def run(root, gpu_ids=GPU_IDS, tp_size=2, pp_size=1, requests_file=None, steps=32, seed=20261112, extra_metrics=False):
    devices=inventory(gpu_ids, require_idle=True)
    if requests_file is None:
        rows = freeze(root)
    else:
        rows = [json.loads(s) for s in requests_file.read_text().splitlines() if s]
        assert rows and len({r['rid'] for r in rows}) == len(rows)
        for row in rows:
            assert len(row['input_ids']) == row['prompt_tokens'] > 2048
        save(root/'selection.json', dict(seed=seed, source=str(requests_file),
            source_sha256=hashlib.sha256(requests_file.read_bytes()).hexdigest(),
            policy='Externally frozen requests; outcome-independent selection',
            requests=[{k:v for k,v in r.items() if k != 'input_ids'} for r in rows]))
        (root/'requests.jsonl').write_text(requests_file.read_text())
    config_path=root/'run-config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update(gpu_ids=gpu_ids,tp_size=tp_size,pp_size=pp_size,steps=steps,seed=seed,
        candidate_widths=[2048,2304,2560,3000,3072,4096,6144,8192] if requests_file else [2048,2304,2560,3072,4096,6144,8192])
    save(config_path,config)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu_ids), KEYE_SM80_DSA='1',
        KEYE_SM80_EXACT_TOPK='1', KEYE_LOOKAHEAD_TRACE_DIR=str(root/'events'),
        KEYE_PREDICTIVE_STUDY='1', KEYE_LOOKAHEAD_RID_PREFIX='predictive__',
        KEYE_LOOKAHEAD_DECODE_STEPS=str(steps), KEYE_LOOKAHEAD_MAX_K='8192',
        KEYE_SM80_PREFILL_SCORE_CHUNK='64', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1')
    if extra_metrics:
        assert tp_size == 1, 'Attention diagnostic currently supports TP1 only'
        env['KEYE_PREDICTIVE_EXTRA_METRICS']='1'
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', '/Tan/model/Keye-VL-2.0-30B-A3B',
        '--trust-remote-code', '--host', '127.0.0.1', '--port', '31242', '--tp-size', str(tp_size), '--pp-size', str(pp_size),
        '--attention-backend', 'keye_sa', '--context-length', '131328', '--max-total-tokens', '140000',
        '--max-running-requests', '1', '--chunked-prefill-size', '1024', '--mem-fraction-static', '0.90',
        '--disable-cuda-graph', '--disable-custom-all-reduce', '--disable-radix-cache',
        '--disable-overlap-schedule', '--dtype', 'bfloat16', '--random-seed', str(seed)]
    if pp_size > 1:
        # Keye is advertised as a VLM, so the framework's default warm-up sends
        # an image. This adapter is text-only; run the explicit Decode health
        # request below instead of expanding the experiment to vision inputs.
        cmd.append('--skip-server-warmup')
    save(root/'server-command.json', dict(command=cmd, env={k:v for k,v in env.items() if k.startswith(('KEYE_', 'CUDA_VISIBLE', 'OMP_', 'OPENBLAS_'))},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        gpu_inventory=devices, tp_size=tp_size, pp_size=pp_size,
        scope='Fresh full-model PP/TP exact trajectory with same-state shadow prediction; instrumentation excluded from performance claims'))
    server = None
    try:
        with (root/'server.log').open('w') as log:
            server = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(root, status='running', phase='loading_model', child_pid=server.pid, total_requests=len(rows), completed_requests=0)
        url = 'http://127.0.0.1:31242'
        deadline = time.time() + 1200
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode} during load')
            try:
                with urllib.request.urlopen(url+'/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            time.sleep(3)
        else:
            raise TimeoutError('TP2 server readiness')
        # Warm-up exercises true Decode, without adding observations to the frozen study.
        response = post(url+'/generate', dict(input_ids=rows[0]['input_ids'][:512], rid='health__',
            sampling_params=dict(temperature=0, max_new_tokens=4, ignore_eos=True), stream=False))
        save(root/'health.json', response)
        if not response.get('output_ids') or response['meta_info'].get('finish_reason',{}).get('type') == 'abort':
            raise RuntimeError('Nonempty real-model health generation required')
        notify(f'HorizonKV 4.2：PP={pp_size}/TP={tp_size}真实Decode健康检查通过；开始{len(rows)}请求、47相邻层对、{steps}步影子实验。')
        for i, row in enumerate(rows):
            update(root, phase='collecting_same_state_trace', completed_requests=i,
                detail=f"{row['group']}，{row['prompt_tokens']} tokens，{row['rid']}")
            post(url+'/flush_cache', {})
            response = post(url+'/generate', dict(input_ids=row['input_ids'], rid=row['rid'],
                sampling_params=dict(temperature=0, max_new_tokens=steps+1, ignore_eos=True), stream=False))
            save(root/f'response-{i:02d}.json', response)
            if len(response.get('output_ids', [])) != steps+1 or response['meta_info'].get('finish_reason',{}).get('type') == 'abort':
                raise RuntimeError(f"Incomplete generation: {row['rid']}")
            chunks = [json.loads(s) for s in (root/'events/manifest.jsonl').read_text().splitlines()]
            req_chunks = [r for r in chunks if r['request_id'] == row['rid']]
            if len(req_chunks) != 47 or {r['target_layer_id'] for r in req_chunks} != set(range(1,48)):
                raise RuntimeError(f"Incomplete 47-layer trace for {row['rid']}: {len(req_chunks)}")
            import torch
            for chunk in req_chunks:
                data = torch.load(root/'events'/chunk['file'], map_location='cpu', weights_only=False)
                if [r['step'] for r in data['rows']] != list(range(steps)):
                    raise RuntimeError(f"Noncontiguous trace: {chunk['file']}")
                if any(r['score_max_abs'] != 0 for r in data['rows']):
                    raise RuntimeError(f"Candidate/full score mismatch: {chunk['file']}")
                first = data['rows'][0]
                if first['selfcheck_max_abs'] != 0 or not first['selfcheck_equal']:
                    raise RuntimeError(f"Full-score self-check failed: {chunk['file']}")
            update(root, completed_requests=i+1)
        update(root, phase='trace_collected', detail=f'{len(rows)} requests captured; shutting down owned server before analysis.')
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        update(root, child_pid=None)
    update(root, phase='analyzing')
    subprocess.run([sys.executable, 'scripts/keye_trace/analyze_horizonkv_predictive.py', '--run-dir', str(root)], cwd=REPO, check=True)
    update(root, status='completed', phase='analysis_completed', detail='Fresh raw traces and processed metrics written; paper figures/report archival pending review by active agent.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--gpu-ids',nargs='+',default=GPU_IDS)
    parser.add_argument('--tp-size',type=int,default=2)
    parser.add_argument('--pp-size',type=int,default=1)
    parser.add_argument('--requests-file',type=Path)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--extra-metrics',action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.monitor:
        monitor(root)
    elif args.freeze_only:
        freeze(root)
    else:
        def stop(_sig, _frame):
            raise KeyboardInterrupt('Owned study interrupted')
        signal.signal(signal.SIGTERM, stop)
        try:
            run(root,args.gpu_ids,args.tp_size,args.pp_size,args.requests_file,args.steps,args.seed,args.extra_metrics)
        except BaseException as exc:
            save(root/'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
            update(root, status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', phase='controller_stopped', detail=str(exc))
            notify('HorizonKV 4.2 实验停止：'+str(exc)+'；已保存日志，等待当前任务检查修复。')
            raise
