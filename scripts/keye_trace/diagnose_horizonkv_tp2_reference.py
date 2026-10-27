"""Locate captured-versus-replayed TP2 differences without changing acceptance gates."""
import argparse
from types import MethodType
import torch
from run_horizonkv_fixed_budget import (
    OLD, initialize, load_rank, assemble, ReplayData, recompute_prior,
    rank_prefix, no_reduce, save,
)


def metrics(a, b):
    a, b = a.float(), b.float()
    delta = (a - b).abs()
    return dict(max_abs=float(delta.max()), rms=float(delta.square().mean().sqrt()),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)),
                failed_elements=int((delta > .01 + .02*b.abs()).sum()),
                elements=a.numel(), bitwise_equal=torch.equal(a,b))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=36)
    p.add_argument('--group',default='RULER128K');p.add_argument('--step',type=int,default=32)
    p.add_argument('--output',required=True);a=p.parse_args()
    config=initialize();layers=[load_rank(config,a.layer,r) for r in [0,1]]
    for layer in layers:layer.mlp.forward=MethodType(no_reduce,layer.mlp)
    frames,f=assemble(OLD,a.group,1,a.layer,a.step,0)
    peer=[]
    for x in frames:
        y=dict(x['peer']);y['peer']={k:v for k,v in x.items() if k!='peer'};peer.append(y)
    data=[ReplayData(OLD,fs,fs[-1],local_rank=r) for r,fs in enumerate([frames,peer])]
    natives=[];captured_partials=[];out={'layer':a.layer,'group':a.group,'step':a.step,'ranks':[]}
    for r in [0,1]:
        recompute_prior(layers[r],data[r],[frames,peer][r]);n=rank_prefix(layers[r],data[r]);natives.append(n)
        cp,_=layers[r].self_attn.o_proj(data[r].source_attention_output);captured_partials.append(cp)
        out['ranks'].append(dict(query=metrics(n['query'],data[r].source_query),
            attention=metrics(n['attention'],data[r].source_attention_output),
            captured_topk_equal=torch.equal(n['selected'].cpu().sort(-1).values,[f,f['peer']][r]['topk'].sort(-1).values)))
    original=f['output_hidden'].cuda()
    for name,reduced in [('native',natives[0]['partial']+natives[1]['partial']),
                         ('captured_attention',captured_partials[0]+captured_partials[1])]:
        h,res=layers[0].post_attention_layernorm(reduced.clone(),natives[0]['residual'].clone())
        parts=[layer.mlp(h.clone()) for layer in layers]
        out[name]=metrics(parts[0]+parts[1],original)
    save(a.output,out);print(out)
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
