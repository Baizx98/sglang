"""Audit and aggregate fixed-budget local-rank measurements; no plotting."""
import argparse,csv,hashlib,json,re,statistics,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

CASE=re.compile(r'RULER(\d+)K-Q(\d+)-c(\d+)-l(\d+)-r(\d+)-s(\d+)')
TOPLEVEL=['input_norm','qkv_projection','qk_norm_rope','indexer_query_prepare',
          'indexer_score','indexer_topk','kv_writeback','kv_resolve_gather',
          'attention','output_projection','post_attention_norm','moe_ffn']


def mean(values):return statistics.mean(values)
def quantile(values,q):return float(np.percentile(values,q))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def aggregate(points):
    if not points:return dict(n=0)
    sums={k:sum(p[k] for p in points) for k in ['probe_us','front_us','index_us','kv_us','attention_us','moe_us']}
    result=dict(n=len(points),cohorts=len({p['cohort'] for p in points}),
        layers=len({p['layer'] for p in points}),steps=len({p['step'] for p in points}),
        mode='|'.join(sorted({p['mode'] for p in points})),
        actual_length_min=min(p['length_min'] for p in points),actual_length_max=max(p['length_max'] for p in points),
        actual_length_mean=mean([p['length_mean'] for p in points]),
        elapsed_mean_us=mean([p['elapsed_us'] for p in points]),
        elapsed_median_us=statistics.median(p['elapsed_us'] for p in points),
        elapsed_p25_us=quantile([p['elapsed_us'] for p in points],25),
        elapsed_p75_us=quantile([p['elapsed_us'] for p in points],75))
    for name in ['probe','front','index','kv','query','score','topk','diff','fetch','resolve',
                 'resolve_extra','writeback','attention','moe','other','gaps','gpu_work',
                 'gpu_index','gpu_kv','gpu_attention','gpu_moe']:
        result[name+'_mean_us']=mean([p[name+'_us'] for p in points])
    for name in ['front','index','kv','attention','moe']:
        result[name+'_share']=sums[name+'_us']/sums['probe_us']
    for name in ['gpu_index','gpu_kv','gpu_attention','gpu_moe']:
        result[name+'_share']=sum(p[name+'_us'] for p in points)/sum(p['gpu_work_us'] for p in points)
    selected=sum(p['selected'] for p in points);miss=sum(p['miss'] for p in points)
    result.update(miss_rate=miss/selected,hit_rate=1-miss/selected,
        miss_payload_mean_bytes=mean([p['payload'] for p in points]),
        cache_over_k_min=min(p['cache_over_k'] for p in points),
        cache_over_k_max=max(p['cache_over_k'] for p in points),
        cache_fraction_mean=mean([p['cache_fraction'] for p in points]),
        persistent_cache_mean_GiB=mean([p['cache_bytes']/2**30 for p in points]),
        index_history_mean_GiB=mean([p['index_bytes']/2**30 for p in points]),
        probe_perturbation_mean_pct=mean([p['probe_overhead_pct'] for p in points]),
        probe_perturbation_median_pct=statistics.median(p['probe_overhead_pct'] for p in points),
        repeat_span_median_pct=statistics.median(p['repeat_span_pct'] for p in points),
        # Useful logical payload rate, NOT measured PCIe link utilization.
        logical_miss_payload_per_fetch_GB_s=(sum(p['payload'] for p in points)/sum(p['fetch_us'] for p in points)/1000
                                            if sum(p['fetch_us'] for p in points)>0 else None))
    by_cohort=defaultdict(list);by_layer=defaultdict(list)
    for p in points:by_cohort[p['cohort']].append(p);by_layer[p['layer']].append(p)
    cohort_elapsed=[mean([p['elapsed_us'] for p in v]) for v in by_cohort.values()]
    cohort_front=[sum(p['front_us'] for p in v)/sum(p['probe_us'] for p in v) for v in by_cohort.values()]
    layer_elapsed=[mean([p['elapsed_us'] for p in v]) for v in by_layer.values()]
    result.update(cohort_elapsed_min_us=min(cohort_elapsed),cohort_elapsed_max_us=max(cohort_elapsed),
        cohort_front_share_min=min(cohort_front),cohort_front_share_max=max(cohort_front),
        sampled_layer_elapsed_min_us=min(layer_elapsed),sampled_layer_elapsed_median_us=statistics.median(layer_elapsed),
        sampled_layer_elapsed_max_us=max(layer_elapsed),sampled_layer_elapsed_std_us=statistics.pstdev(layer_elapsed))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    points=[];roots=set();seen=set();closure_errors=[];workspace_violations=[];impl_hashes=defaultdict(set)
    for source in a.inputs:
        for row in csv.DictReader(source.open()):
            assert row['case'] not in seen;seen.add(row['case'])
            L,cohort,C,layer,rank,step=map(int,CASE.fullmatch(row['case']).groups())
            path=Path(row['source_file']);d=json.loads(path.read_text());roots.add(path.parent.parent)
            assert d['timed'] and len(set(d['input_ids']))==C
            assert d['shape']['q_heads']==16 and d['shape']['kv_heads']==2 and d['shape']['moe_intermediate']==384
            assert 'GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822' in d['gpu_inventory_at_start']
            assert not d['source_dirty_at_start']
            for name,value in d['implementation_sha256'].items():impl_hashes[name].add(value)
            assert not d['gpu_isolation_before']['foreign'] and not d['gpu_isolation_after']['foreign']
            v=d['results'][0];t=v['timing'];sp={x['component']:x for x in t['probe_component_spans']}
            assert v['topk_equal'] and row['topk_equal']=='True' and v['graph_output_bitwise_equal']
            assert abs(statistics.median(t['elapsed_us'])-float(row['elapsed_median_us']))<1e-4
            assert v['selected']==C*2048 and 0<=v['miss_tokens']<=v['selected']
            assert v['host_read_payload_bytes']==v['miss_tokens']*1024
            duration=lambda key:sp[key]['stream_span_us'] if key in sp else 0.
            probe=t['probe_elapsed_us'];index=sp['indexer_topk']['end_us']-sp['indexer_query_prepare']['start_us']
            kv=sp['kv_resolve_gather']['end_us']-sp['indexer_topk']['end_us'];front=index+kv
            assert abs(front-v['front_probe_us'])<.001 and abs(front/probe-v['front_probe_share'])<1e-6
            gaps=probe-sum(duration(k) for k in TOPLEVEL)
            if gaps<-.2:closure_errors.append(dict(case=d['case'],gaps_us=gaps))
            work=defaultdict(float)
            for key,val in v['gpu_profile']['component_gpu_work_us'].items():work[key.split('/')[-1]]+=val
            gpu_total=v['gpu_profile']['gpu_work_total_us'];assert abs(sum(work.values())-gpu_total)<.01
            budget=json.loads((path.parent.parent/f"budget-{d['case']}.json").read_text())
            assert budget['status'] in ['H','D']
            used=sum(budget[k] for k in ['weights_bytes','index_bytes','workspace_and_stage_bytes',
                 'reserve_bytes','persistent_main_bytes','metadata_bytes'])
            assert used<=budget['device_bytes'],d['case']
            ledger=json.loads((path.parent.parent/'budget.json').read_text())
            envelope=ledger['workspace_fixed_bytes']+sum(d['lengths'])*ledger['workspace_bytes_per_history_token']+C*2048*1024
            if v['incremental_peak_bytes']>envelope:
                workspace_violations.append(dict(case=d['case'],incremental_peak=v['incremental_peak_bytes'],envelope=envelope))
            assert (budget['status']=='H')==(v['mode']=='h-native')
            b=budget['cache_tokens_per_request'];diff=duration('kv_diff');fetch=duration('kv_fetch')
            p=dict(case=d['case'],L=L,C=C,cohort=cohort,layer=layer,rank=rank,step=step,
                mode=v['mode'],status=budget['status'],length_min=min(d['lengths']),length_max=max(d['lengths']),length_mean=mean(d['lengths']),
                elapsed_us=v['elapsed_median_us'],probe_us=probe,front_us=front,index_us=index,kv_us=kv,
                query_us=duration('indexer_query_prepare'),score_us=duration('indexer_score'),topk_us=duration('indexer_topk'),
                diff_us=diff,fetch_us=fetch,resolve_us=duration('kv_resolve_gather'),
                resolve_extra_us=duration('kv_resolve_gather')-diff-fetch,
                writeback_us=duration('kv_writeback'),attention_us=duration('attention'),moe_us=duration('moe_ffn'),
                other_us=probe-front-duration('attention')-duration('moe_ffn'),gaps_us=gaps,
                gpu_work_us=gpu_total,gpu_index_us=sum(work[k] for k in ['indexer_query_prepare','indexer_score','indexer_topk']),
                gpu_kv_us=sum(work[k] for k in ['kv_writeback','kv_diff','kv_fetch','kv_resolve_gather']),
                gpu_attention_us=work['attention'],gpu_moe_us=work['moe_ffn'],
                selected=v['selected'],miss=v['miss_tokens'],payload=v['host_read_payload_bytes'],
                cache_over_k=b/2048,cache_fraction=b/max(d['lengths']),cache_bytes=budget['persistent_main_bytes'],
                index_bytes=budget['index_bytes'],probe_overhead_pct=(probe/v['elapsed_median_us']-1)*100,
                repeat_span_pct=(max(t['elapsed_us'])-min(t['elapsed_us']))/v['elapsed_median_us']*100)
            points.append(p)
    assert len(points)==5250 and not closure_errors
    assert all(len(v)==1 for v in impl_hashes.values()),impl_hashes
    markers=defaultdict(set);lengths={p['L'] for p in points}
    for root in roots:
        for path in (root/'results').glob('*.json'):
            m=CASE.fullmatch(path.stem)
            if not m:continue
            L,cohort,C,layer,rank,step=map(int,m.groups())
            if L not in lengths:continue
            d=json.loads(path.read_text())
            if not d.get('timed'):markers[(L,C,rank)].add(d['capacity_status'])
    group=defaultdict(list)
    for p in points:group[(p['L'],p['C'],p['rank'])].append(p)
    cell_rows=[];layer_rows=[]
    for L in sorted({p['L'] for p in points}):
        for C in [1,2,4,8,16,32,64]:
            for rank in [0,1]:
                ps=group[(L,C,rank)]
                for window in ['cold0','warm32plus','all_snapshots']:
                    chosen=[p for p in ps if window=='all_snapshots' or (p['step']==0)==(window=='cold0')]
                    status='|'.join(sorted({p['status'] for p in ps})) if ps else '|'.join(sorted(markers[(L,C,rank)]))
                    assert status, (L,C,rank)
                    assert not ps or len(ps)==75
                    cell_rows.append(dict(length_bucket_K=L,concurrency=C,rank=rank,window=window,status=status,**aggregate(chosen)))
                for layer in [1,12,24,36,47]:
                    chosen=[p for p in ps if p['layer']==layer and p['step']>0]
                    if chosen:layer_rows.append(dict(length_bucket_K=L,concurrency=C,rank=rank,layer=layer,**aggregate(chosen)))
    # Invalid cells use blanks, never zero latency or zero waiting.
    fields=list(next(r for r in cell_rows if r['n']).keys())
    cell_rows=[{k:r.get(k) for k in fields} for r in cell_rows]
    write_csv(a.out/'cells.csv',cell_rows);write_csv(a.out/'layers.csv',layer_rows)
    warm=[p for p in points if p['step']>0];dwarm=[p for p in warm if p['status']=='D']
    fits=[]
    for C in [1,2,4,8,16,32,64]:
        for rank in [0,1]:
            cs=[r for r in cell_rows if r['window']=='warm32plus' and r['n'] and r['concurrency']==C and r['rank']==rank]
            if len(cs)<3:continue
            x=np.array([r['actual_length_mean']/1024 for r in cs]);y=np.array([r['score_mean_us'] for r in cs]);coef=np.polyfit(x,y,1);pred=np.polyval(coef,x)
            fits.append(dict(concurrency=C,rank=rank,length_cells=len(cs),score_us_per_K=float(coef[0]),
                             intercept_us=float(coef[1]),r2=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())))
    ranks=[]
    for L,C,rank in list(group):
        if rank or not group[(L,C,rank)]:continue
        x=aggregate([p for p in group[(L,C,0)] if p['step']>0]);y=aggregate([p for p in group[(L,C,1)] if p['step']>0])
        ranks.append(dict(L=L,C=C,rank1_over_rank0=y['elapsed_mean_us']/x['elapsed_mean_us']))
    perturb=[p['probe_overhead_pct'] for p in warm]
    warm_cells=[r for r in cell_rows if r['window']=='warm32plus' and r['n']]
    report=dict(source_inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in a.inputs],
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),analysis_script_sha256=sha(__file__),
        common_replay_implementation_sha256={k:next(iter(v)) for k,v in impl_hashes.items()},
        points=len(points),warm_points=len(warm),cold_points=len(points)-len(warm),
        valid_cells=sum(bool(v) for v in group.values()),cell_rows_count=len(cell_rows),sampled_layer_rows=len(layer_rows),
        stage_closure_errors=closure_errors,workspace_envelope_violations=workspace_violations,
        score_fits=fits,rank_elapsed_ratios=ranks,
        warm_probe_perturbation_pct={str(q):quantile(perturb,q) for q in [0,25,50,75,95,100]},
        warm_repeat_span_pct={str(q):quantile([p['repeat_span_pct'] for p in warm],q) for q in [50,95,100]},
        warm_repeat_span_over10pct_points=sum(p['repeat_span_pct']>10 for p in warm),
        warm_probe_difference_over30pct_points=sum(abs(p['probe_overhead_pct'])>30 for p in warm),
        probe_difference_extremes=[dict(case=p['case'],probe_difference_pct=p['probe_overhead_pct'])
                                   for p in sorted(warm,key=lambda p:abs(p['probe_overhead_pct']),reverse=True)[:5]],
        warm_front_share_over_half_cells=sum(r['front_share']>.5 for r in warm_cells),
        warm_index_share_over_half_cells=sum(r['index_share']>.5 for r in warm_cells),
        warm_index_larger_than_moe_cells=sum(r['index_mean_us']>r['moe_mean_us'] for r in warm_cells),
        D_warm_zero_miss_points=sum(p['miss']==0 for p in dwarm),D_warm_points=len(dwarm),
        D_warm_zero_miss_fetch_mean_us=mean([p['fetch_us'] for p in dwarm if p['miss']==0]) if any(p['miss']==0 for p in dwarm) else None,
        measurement_contract='Local batch-layer CUDA graph median; separate instrumented probe and eager activity profiler. No scaling probe spans into uninstrumented latency.',
        aggregation_contract='Cold step0 separate; warm snapshots32/64/96/127. Descriptive means; shares=sum(stage)/sum(probe). Three request cohorts; layers/steps/ranks are correlated, no IID CI.',
        units='*_us microseconds; *_share and rates unitless; *_pct percentages; *_GiB 2^30 bytes; length K=1024 tokens',
        validity='5250 actual main measurements only; no pilot, archived402, x8, rejected232K/256K; all gates and isolation-boundary checks audited')
    (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['points','warm_points','cold_points','warm_probe_perturbation_pct','D_warm_zero_miss_points','D_warm_zero_miss_fetch_mean_us']},indent=2))


if __name__=='__main__':main()
