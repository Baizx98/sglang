"""Fixed deployment-budget TP2-shaped local layer measurements; no plotting."""
import argparse,json,csv,gc,statistics,subprocess,time,hashlib
from functools import lru_cache
from pathlib import Path
from types import MethodType
import torch
from safetensors import safe_open
from horizonkv_layer_replay import initialize,load_layer,read_frames,ReplayData,ReplayLayer,graph_measure,MODEL
from horizonkv_tp2_local import load_rank
from horizonkv_single_runtime import GPU,REPO,save,update,notify
from horizonkv_trace_work import kernel_work_from_trace

OLD=REPO/'data/horizonkv-motivation/2026-10-21_bs16-v01'

def no_reduce(self,h,*args,**kwargs):
    return self.forward_normal(h,should_allreduce_fusion=True)

@torch.inference_mode()
def rank_prefix(layer,data):
    """Exact same-batch native prefix, gathered from real Host KV outside timing."""
    from horizonkv_host_kernels import HostPointer,gather,write_current
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged
    from sglang.srt.layers.attention.keye_dsa_sm80 import keye_dsa_paged_attention,KeyeDSAWorkspace
    att=layer.self_attn;c=data.batch
    h=data.hidden.clone();res=None if data.residual is None else data.residual.clone()
    if res is None:res=h;h=layer.input_layernorm(h)
    else:h,res=layer.input_layernorm(h,res)
    qkv,_=att.qkv_proj(h);q,k,v=qkv.split([att.q_size,att.kv_size,att.kv_size],dim=-1)
    q,k=att._apply_qk_norm(q,k);q,k=att.rotary_emb(data.positions,q,k)
    iq,ik,weights=att.sa_indexer._get_q_k_w_bf16(h,data.positions,False)
    data.index.index_copy_(0,data.current,ik)
    scores=keye_indexer_score_paged(iq,data.index,data.page_table,weights,data.lengths,att.sa_indexer.softmax_scale)
    selected=att.sa_indexer._topk_sm80(scores,data.lengths)
    hk,hv=HostPointer(data.hostk),HostPointer(data.hostv)
    write_current(hk,hv,k,v,data.lengths,data.maxlen)
    sk=torch.empty(c,2048,2,128,device='cuda',dtype=torch.bfloat16);sv=torch.empty_like(sk)
    gather(hk,hv,selected,sk,sv,data.lengths,data.maxlen)
    slots=torch.arange(c*2048,device='cuda',dtype=torch.int32).view(c,-1)
    output=keye_dsa_paged_attention(q.view(c,16,128),sk.view(-1,2,128),sv.view(-1,2,128),slots,
        num_q_heads=16,head_dim=128,sm_scale=att.scaling,is_decode=True,workspace=KeyeDSAWorkspace())
    partial,_=att.o_proj(output)
    return dict(selected=selected.clone(),attention=output.clone(),query=q.clone(),partial=partial.clone(),residual=res.clone())

@torch.inference_mode()
def recompute_prior(layer,data,frames):
    """Actual batch-shape selection history drives LRU; C1 traces are not reused as truth."""
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged
    data.prior={rid:[] for rid in data.frame['request_ids']}
    for f in frames[:-1]:
        h=f['attention_input'].cuda();positions=f['positions'].cuda();lengths=f['seq_lens'].int().cuda()
        iq,ik,w=layer.self_attn.sa_indexer._get_q_k_w_bf16(h,positions,False)
        current=(data.page_table[:,0]+lengths-1).long();data.index.index_copy_(0,current,ik)
        scores=keye_indexer_score_paged(iq,data.index,data.page_table,w,lengths,layer.self_attn.sa_indexer.softmax_scale)
        selected=layer.self_attn.sa_indexer._topk_sm80(scores,lengths).cpu()
        for row,rid in enumerate(f['request_ids']):data.prior[rid].append((f['steps'][row],selected[row]))


@lru_cache(maxsize=512)
def cached_frames(root,wave,layer):
    return read_frames(root,wave,layer,batch=1)

def assemble(root,group,c,layer_id,step,rank):
    """Merge distinct real C1 frames in memory, never repeat an input request."""
    m=json.loads((root/'long-input-manifest.json').read_text())
    group=next(g for g in m['groups'] if g['label']==group)
    waves=group['waves'][:c]
    if len(waves)<c:raise LookupError(f'Only {len(waves)} distinct requests available for C={c}')
    parts=[cached_frames(root,w,layer_id) for w in waves]
    common=set.intersection(*({f['steps'][0] for f in fs} for fs in parts))
    if step not in common:
        raise LookupError(f'request cohort lacks complete step {step}; common={sorted(common)}')
    def combine(fs):
        out={}
        for key,v in fs[0].items():
            vals=[f[key] for f in fs]
            if key=='peer':continue
            if isinstance(v,torch.Tensor):out[key]=torch.cat(vals,dim=-1 if key=='positions' else 0)
            elif isinstance(v,list):out[key]=sum(vals,[])
            else:
                assert all(x==v for x in vals),(key,vals);out[key]=v
        assert len(set(out['request_ids']))==c
        return out
    frames=[]
    for s in sorted(common):
        if s>step:continue
        fs=[next(f for f in x if f['steps'][0]==s) for x in parts]
        a,b=combine(fs),combine([f['peer'] for f in fs]);a['peer']=b;b['peer']=a
        # Use a non-cyclic shallow peer structure for captures/serialization.
        frame=a if rank==0 else b
        frame['peer']=dict(b if rank==0 else a);frame['peer'].pop('peer',None)
        frames.append(frame)
    return frames,frames[-1]


def local_shape_record(layer,rank):
    return dict(rank=rank,q_heads=layer.self_attn.num_heads,kv_heads=layer.self_attn.num_kv_heads,
                moe_intermediate=layer.mlp.experts.intermediate_size_per_partition,
                parameter_bytes=sum(p.numel()*p.element_size() for p in layer.parameters()))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source',type=Path,default=OLD);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--layers',type=int,nargs='+',default=[12]);p.add_argument('--concurrencies',type=int,nargs='+',default=[1,4,8,16])
    p.add_argument('--groups',nargs='+',default=['RULER32K','RULER64K','RULER128K'])
    p.add_argument('--steps',type=int,nargs='+',default=[32]);p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();root=args.run_dir;root.mkdir(parents=True,exist_ok=True)
    for d in ['results','timelines']:(root/d).mkdir(exist_ok=True)
    source_info=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),
        gpu_inventory=subprocess.check_output(['nvidia-smi','-i',GPU,
            '--query-gpu=uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.width.max,gpu_recovery_action',
            '--format=csv'],text=True,timeout=20),
        sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['run_horizonkv_fixed_budget.py','horizonkv_layer_replay.py','horizonkv_tp2_local.py','horizonkv_fixed_budget_ledger.py','horizonkv_host_kernels.py']})
    save(root/f'rank{args.rank}-process-source.json',source_info)
    config=initialize();torch.manual_seed(20261014)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]):
        torch.ones(16,device="cuda").add_(1);torch.cuda.synchronize()
    assert str(torch.cuda.get_device_properties(0).uuid)==GPU.removeprefix('GPU-')
    record={};total=len(args.layers)*len(args.groups)*len(args.concurrencies)*len(args.steps);done=0
    for lid in args.layers:
        layer=load_rank(config,lid,args.rank);layer.mlp.forward=MethodType(no_reduce,layer.mlp)
        peer_layer=load_rank(config,lid,1-args.rank);peer_layer.mlp.forward=MethodType(no_reduce,peer_layer.mlp)
        record[str(lid)]=local_shape_record(layer,args.rank)
        for group in args.groups:
            for c in args.concurrencies:
                for step in args.steps:
                    case=f'{group}-c{c}-l{lid}-r{args.rank}-s{step}';target=root/'results'/f'{case}.json'
                    if target.exists():done+=1;continue
                    update(root,status='running',phase='tp2_local_validation' if args.validate_only else 'fixed_budget_pilot',child_pid=__import__('os').getpid(),total_cases=total,completed_cases=done,detail=case)
                    group_meta=next(g for g in json.loads((args.source/'long-input-manifest.json').read_text())['groups'] if g['label']==group)
                    if len(group_meta['waves'])<c and (root/'budget.json').exists():
                        from horizonkv_fixed_budget_ledger import point_budget
                        early=point_budget(json.loads((root/'budget.json').read_text()),c,max(group_meta['actual_input_tokens'])+step,args.rank)
                        save(target,dict(case=case,timed=False,capacity_status=early['status'] if early['status'].startswith('X') else 'insufficient-distinct-inputs',budget=early));done+=1;continue
                    try:
                        frames,frame=assemble(args.source,group,c,lid,step,args.rank)
                    except LookupError as exc:
                        save(target,dict(case=case,timed=False,capacity_status='missing-common-decode-step',detail=str(exc)))
                        done+=1;continue
                    if (root/'budget.json').exists() and not args.validate_only:
                        from horizonkv_fixed_budget_ledger import point_budget
                        early=point_budget(json.loads((root/'budget.json').read_text()),c,int(frame['seq_lens'].max()),args.rank)
                        if early['status'].startswith('X'):
                            save(target,dict(case=case,timed=False,capacity_status=early['status'],budget=early));done+=1;continue
                    data=ReplayData(args.source,frames,frame,local_rank=args.rank)
                    recompute_prior(layer,data,frames)
                    native=rank_prefix(layer,data)
                    peer_frames=[]
                    for f in frames:
                        peer=dict(f['peer']);peer['peer']={k:v for k,v in f.items() if k!='peer'};peer_frames.append(peer)
                    peer_data=ReplayData(args.source,peer_frames,peer_frames[-1],local_rank=1-args.rank)
                    recompute_prior(peer_layer,peer_data,peer_frames)
                    peer_native=rank_prefix(peer_layer,peer_data)
                    assert torch.equal(native['selected'],peer_native['selected'])
                    data.reduced_o_source=native['partial']+peer_native['partial'];data.reduced_o=data.reduced_o_source.clone()
                    ref_input,ref_res=layer.post_attention_layernorm(data.reduced_o_source.clone(),native['residual'].clone())
                    reference_local=layer.mlp(ref_input.clone()).clone()
                    reference_full=reference_local+peer_layer.mlp(ref_input.clone())
                    source_full_error=float((reference_full-frame['output_hidden'].cuda()).abs().max())
                    source_anchor_error=None
                    if c==1:
                        # Fixtures were captured on A6000+L40S. Anchor weight loading
                        # and TP reduction with their actual Attention outputs;
                        # an A6000 recomputation of the L40S prefix is not bitwise
                        # equivalent. Timed paths still match same-GPU native below.
                        captured_o,_=layer.self_attn.o_proj(data.source_attention_output)
                        captured_peer_o,_=peer_layer.self_attn.o_proj(peer_data.source_attention_output)
                        anchor_input,_=layer.post_attention_layernorm(captured_o+captured_peer_o,native['residual'].clone())
                        anchor_output=layer.mlp(anchor_input.clone())+peer_layer.mlp(anchor_input.clone())
                        source_anchor_error=float((anchor_output-frame['output_hidden'].cuda()).abs().max())
                        assert torch.allclose(anchor_output,frame['output_hidden'].cuda(),rtol=.02,atol=.01),(case,'captured Attention anchor',source_anchor_error)
                    del peer_data,peer_frames
                    # Calibrated deployment ledger is created before the timed matrix.
                    ledger=json.loads((root/'budget.json').read_text()) if (root/'budget.json').exists() else None
                    choices=[('h-native',None)] if args.validate_only else []
                    if ledger and not args.validate_only:
                        from horizonkv_fixed_budget_ledger import point_budget
                        point=point_budget(ledger,c,data.maxlen,args.rank)
                        save(root/f'budget-{case}.json',point)
                        if point['status'].startswith('X'):
                            save(target,dict(case=case,capacity_status=point['status'],budget=point,timed=False));done+=1;del data;continue
                        choices=[('h-native',None)] if point['status']=='H' else [('l-cached',point['cache_tokens_per_request'])]
                    elif not args.validate_only:raise RuntimeError('Calibrate deployment budget before timing')
                    results=[]
                    for mode,cap in choices:
                        runner=ReplayLayer(layer,data,mode,cap or 4096);runner.reset();value=runner.forward();torch.cuda.synchronize()
                        # Captures have already reduced the MoE output on both ranks.
                        # Reconstruct the other rank contribution outside timing.
                        complete_output=value[0]
                        errors=dict(normalization=float((runner.last_attention_input-data.source_attention_input).abs().max()),
                                    query=float((runner.last_query-data.source_query).abs().max()),
                                    attention=float((runner.last_attention_output-data.source_attention_output).abs().max()),
                                    output=float((complete_output-reference_local).abs().max()),source_full_output=source_full_error,
                                    source_captured_attention_anchor=source_anchor_error)
                        selected_same=torch.equal(runner.last_topk,native['selected'])
                        source_set_same=torch.equal(runner.last_topk.cpu().sort(-1).values,frame['topk'].sort(-1).values)
                        assert selected_same,(case,errors)
                        assert torch.allclose(runner.last_attention_output,native['attention'],rtol=.02,atol=.005),(case,errors)
                        # Compare rank-local MoE output to captured real TP2 rank output.
                        assert torch.allclose(complete_output,reference_local,rtol=.02,atol=.01),(case,errors)
                        wanted=runner.last_topk;valid=wanted>=0
                        miss=0 if mode=='h-native' else int((valid&(runner.initial_cache['map'].gather(1,wanted.long().clamp_min(0))<0)).sum())
                        torch.cuda.reset_peak_memory_stats();base_allocated=torch.cuda.memory_allocated()
                        timing,output=graph_measure(runner.forward,runner.reset,[runner],profile_path=root/'timelines'/f'{case}.json')
                        spans={s['component']:s for s in timing['probe_component_spans']}
                        assert torch.allclose(output[0],reference_local,rtol=.002,atol=.001),(case,'graph output')
                        front=spans['kv_resolve_gather']['end_us']-spans['indexer_query_prepare']['start_us']
                        results.append(dict(mode=mode,errors=errors,topk_equal=selected_same,source_c1_topk_equal=source_set_same,cache_tokens_per_request=cap,
                           selected=int(valid.sum()),miss_tokens=miss,host_read_payload_bytes=miss*1024,
                           host_write_payload_bytes=0 if mode=='h-native' else c*1024,
                           elapsed_median_us=statistics.median(timing['elapsed_us']),
                           front_probe_us=front,front_probe_share=front/timing['probe_elapsed_us'],
                           indexer_probe_us=spans['indexer_topk']['end_us']-spans['indexer_query_prepare']['start_us'],
                           postselect_kv_probe_us=spans['kv_resolve_gather']['end_us']-spans['indexer_topk']['end_us'],
                           graph_output_bitwise_equal=torch.equal(output[0],reference_local),
                           timing=timing,gpu_profile=kernel_work_from_trace(root/'timelines'/f'{case}.json'),
                           base_allocated_bytes=base_allocated,incremental_peak_bytes=torch.cuda.max_memory_allocated()-base_allocated,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                        del runner
                    save(target,dict(case=case,group=group,concurrency=c,layer=lid,rank=args.rank,step=step,lengths=frame['seq_lens'].tolist(),
                         shape=record[str(lid)],results=results,timed=True,input_ids=frame['request_ids'],
                         source_commit=source_info['commit'],implementation_sha256=source_info['sha256'],source_dirty_at_start=source_info['dirty'],
                         gpu_inventory_at_start=source_info['gpu_inventory'],
                         scope='one A6000, actual TP2 rank-local kernels; true reduced O injected; NCCL omitted; not full-model latency'))
                    done+=1;del data,frames,frame;gc.collect();torch.cuda.empty_cache()
        del layer,peer_layer;gc.collect();torch.cuda.empty_cache()
    save(root/f'rank{args.rank}-shapes.json',record)
    update(root,phase='tp2_local_validation_completed' if args.validate_only else 'fixed_budget_pilot_completed',completed_cases=done,total_cases=total,child_pid=None)
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
