"""Audit and summarize completed long-context replay without new GPU measurements."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main(root):
    src=root/'processed-long-replay-v01'
    out=root/'report-long-replay-v01';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((src/'latency-and-dependencies.csv').open()))
    component_rows=list(csv.DictReader((src/'component-work.csv').open()))
    audit=json.loads((src/'audit.json').read_text())
    assert len(rows)==audit['cases']
    assert len(component_rows)==12*len(rows)
    for name,expected in audit['case_sha256'].items():
        assert digest(root/'long-replay-v01/results'/name)==expected,name
    components=collections.defaultdict(dict)
    for row in component_rows:
        assert row['component'] not in components[row['case']]
        components[row['case']][row['component']]=float(row['actual_gpu_work_us'])
    names=sorted({k for values in components.values() for k in values})
    assert len(names)==12
    index={r['case']:r for r in rows}
    wide=[];groups=collections.defaultdict(list)
    for r in rows:
        work=components[r['case']]
        assert abs(sum(work.values())-float(r['gpu_work_total_us']))<1e-6
        repeats=[float(r[f'repeat_{i}_us']) for i in range(1,6)]
        assert statistics.median(repeats)==float(r['elapsed_median_us'])
        if r['mode'] not in {'native-unsorted','native-fast-topk-v2'}:
            assert r['usable']==r['output_bitwise_equal']==r['selected_set_equal']=='True'
        baseline=index[r['case'].removesuffix(r['mode'])+r['baseline_mode']]
        enriched=dict(r)
        for metric in ['attention_probe_start_us','moe_probe_start_us']:
            enriched['paired_advance_'+metric]=float(baseline[metric])-float(r[metric])
        enriched.update({name+'_gpu_us':work.get(name,0) for name in names})
        wide.append(enriched)
        groups[(r['group'],r['concurrency'],r['snapshot'],r['mode'])].append(enriched)
    summary=[]
    metrics=['elapsed_median_us','gpu_work_total_us','indexer_gpu_work_us','indexer_gpu_work_ratio',
             'demand_pipeline_gpu_work_ratio','delta_vs_baseline_us','reduction_vs_baseline_pct',
             'paired_advance_attention_probe_start_us','paired_advance_moe_probe_start_us',
             'logical_miss_tokens','selected_tokens','host_read_payload_bytes','host_write_payload_bytes',
             'persistent_main_kv_bytes','index_k_bytes','layer_parameter_bytes','unique_experts',
             'unique_expert_weight_bytes','selected_recall','probe_overhead_ratio']+[n+'_gpu_us' for n in names]
    for key,values in sorted(groups.items()):
        row=dict(zip(['group','concurrency','snapshot','mode'],key))
        row.update(snapshot_points=len(values),request_cohorts=len({r['wave'] for r in values}),
                   usable_points=sum(r['usable']=='True' for r in values),
                   exact_set_points=sum(r['selected_set_equal']=='True' for r in values))
        for m in metrics:row[m+'_median']=statistics.median(float(r[m]) for r in values)
        summary.append(row)
    kernel_counts=collections.Counter()
    for f in sorted((root/'long-replay-v01/timelines').glob('*.json')):
        trace=json.loads(f.read_text())['traceEvents']
        kernels={e['name'] for e in trace if e.get('cat')=='kernel'}
        assert any('_dsa_split_online' in k for k in kernels),f
        assert any('_dsa_reduce_online' in k for k in kernels),f
        kernel_counts['sparse_attention_confirmed']+=1
    assert kernel_counts['sparse_attention_confirmed']==len(rows)
    write_csv(out/'point-components.csv',wide)
    write_csv(out/'group-summary.csv',summary)
    checks=dict(cases=len(rows),component_rows=len(component_rows),summary_rows=len(summary),
                exact_cases=sum(r['mode'] not in {'native-unsorted','native-fast-topk-v2'} for r in rows),
                optimized_diagnostic_cases=sum(r['mode'] in {'native-unsorted','native-fast-topk-v2'} for r in rows),usable_cases=audit['usable_cases'],
                kernel_checks=dict(kernel_counts),all_case_hashes_verified=True,
                all_component_sums_verified=True,all_repeat_medians_verified=True,
                source_files={n:digest(src/n) for n in ['latency-and-dependencies.csv','component-work.csv','audit.json']},
                environment=audit['environment'],
                direct_files={n:digest(out/n) for n in ['point-components.csv','group-summary.csv']},
                aggregation='Each point retains five graph repeats. Group medians across recorded input/layer points; merged larger batches use distinct C1 requests; not independent request counts or confidence intervals.',
                units='Times and component GPU work: us; *_ratio and recall: dimensionless; storage/payload: bytes; capacities: token slots per layer; counts: integers.',
                caveats='Single A6000 TP1 layer replay; C4 merges four independently collected C1 histories. Mixed A6000+L40S timings excluded. Reference exact TopK and current GPU cache implementation; fused Host gather includes transfer. Oracles omit work, not method gains. Sparse kernels checked on every profile. Diagnostic failures retained, not task-quality evidence.',
                paper_ready=False)
    (out/'report-audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in checks.items() if isinstance(v,(bool,int))},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir.resolve())
