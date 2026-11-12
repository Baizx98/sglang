"""Run the same-state 4.2 study; monitor verified files and notify every 20 min."""
import argparse
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

from horizonkv_a6000 import GPU_IDS, inventory, notify
from horizonkv_single_runtime import REPO, save, update


def monitor(root):
    next_notice = time.time() + 1200
    while True:
        state = json.loads((root / 'state.json').read_text())
        terminal = state.get('status') in {'completed', 'failed', 'interrupted'}
        if terminal or time.time() >= next_notice:
            metrics = root / 'events' / 'manifest.jsonl'
            chunks = sum(1 for _ in metrics.open()) if metrics.exists() else 0
            record = dict(checked_unix_s=time.time(), state=state, chunks=chunks)
            try:
                record['gpus'] = inventory()
            except Exception as exc:
                record['gpu_error'] = type(exc).__name__
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
        return json.load(response)


def run(root):
    inventory(require_idle=True)
    rows = freeze(root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(GPU_IDS), KEYE_SM80_DSA='1',
        KEYE_SM80_EXACT_TOPK='1', KEYE_LOOKAHEAD_TRACE_DIR=str(root/'events'),
        KEYE_PREDICTIVE_STUDY='1', KEYE_LOOKAHEAD_RID_PREFIX='predictive__',
        KEYE_LOOKAHEAD_DECODE_STEPS='32', KEYE_LOOKAHEAD_MAX_K='8192',
        KEYE_SM80_PREFILL_SCORE_CHUNK='64', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1')
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', '/Tan/model/Keye-VL-2.0-30B-A3B',
        '--trust-remote-code', '--host', '127.0.0.1', '--port', '31242', '--tp-size', '2',
        '--attention-backend', 'keye_sa', '--context-length', '131328', '--max-total-tokens', '140000',
        '--max-running-requests', '1', '--chunked-prefill-size', '1024', '--mem-fraction-static', '0.85',
        '--disable-cuda-graph', '--disable-custom-all-reduce', '--disable-radix-cache',
        '--disable-overlap-schedule', '--dtype', 'bfloat16', '--random-seed', '20261112']
    save(root/'server-command.json', dict(command=cmd, env={k:v for k,v in env.items() if k.startswith(('KEYE_', 'CUDA_VISIBLE', 'OMP_', 'OPENBLAS_'))},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        gpu_inventory=inventory(), scope='Fresh full-model TP2 exact trajectory with same-state shadow prediction; instrumentation excluded from performance claims'))
    server = None
    try:
        with (root/'server.log').open('w') as log:
            server = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(root, status='running', phase='loading_tp2_model', child_pid=server.pid, total_requests=len(rows), completed_requests=0)
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
        notify('HorizonKV 4.2：双 A6000 TP2 模型加载及真实 Decode 健康检查通过；开始24请求、全47相邻层对、32步同状态影子实验。')
        for i, row in enumerate(rows):
            update(root, phase='collecting_same_state_trace', completed_requests=i,
                detail=f"{row['group']}，{row['prompt_tokens']} tokens，{row['rid']}")
            post(url+'/flush_cache', {})
            response = post(url+'/generate', dict(input_ids=row['input_ids'], rid=row['rid'],
                sampling_params=dict(temperature=0, max_new_tokens=33, ignore_eos=True), stream=False))
            save(root/f'response-{i:02d}.json', response)
            if len(response.get('output_ids', [])) != 33 or response['meta_info'].get('finish_reason',{}).get('type') == 'abort':
                raise RuntimeError(f"Incomplete generation: {row['rid']}")
            chunks = [json.loads(s) for s in (root/'events/manifest.jsonl').read_text().splitlines()]
            req_chunks = [r for r in chunks if r['request_id'] == row['rid']]
            if len(req_chunks) != 47 or {r['target_layer_id'] for r in req_chunks} != set(range(1,48)):
                raise RuntimeError(f"Incomplete 47-layer trace for {row['rid']}: {len(req_chunks)}")
            update(root, completed_requests=i+1)
        update(root, phase='trace_collected', detail='24 fresh requests captured; shutting down owned server before analysis.')
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
            run(root)
        except BaseException as exc:
            save(root/'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
            update(root, status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', phase='controller_stopped', detail=str(exc))
            notify('HorizonKV 4.2 实验停止：'+str(exc)+'；已保存日志，等待当前任务检查修复。')
            raise
