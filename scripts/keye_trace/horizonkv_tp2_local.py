"""Single-device TP2-rank shape construction, with no fake communication.

Initialize the real runtime as one process, override rank/size getters only
while constructing/loading actual modules, then measure their local operations.
Collective outputs must be supplied from real fixtures or separately summed
rank partials outside timing; never claim NCCL or complete TP2 latency.
"""
from contextlib import contextmanager
import sys

@contextmanager
def rank_shape(rank):
    names={'get_tensor_model_parallel_world_size':2,'get_tensor_model_parallel_rank':rank,
           'get_attention_tp_size':2,'get_attention_tp_rank':rank,
           'get_moe_tensor_parallel_world_size':2,'get_moe_tensor_parallel_rank':rank}
    changed=[]
    for module in list(sys.modules.values()):
        if module is None or not getattr(module,'__name__','').startswith('sglang.'):
            continue
        for name,value in names.items():
            if name in vars(module):
                changed.append((module,name,getattr(module,name)))
                setattr(module,name,lambda value=value:value)
    try:yield
    finally:
        for module,name,value in reversed(changed):setattr(module,name,value)


def load_rank(config,layer_id,rank):
    # Import before overriding: constructors refer to module-local getters.
    import sglang.srt.models.keye_topk_mask_moe
    import sglang.srt.layers.moe.fused_moe_triton.layer
    from horizonkv_layer_replay import load_layer
    with rank_shape(rank):
        layer=load_layer(config,layer_id)
    assert layer.self_attn.num_heads==16 and layer.self_attn.num_kv_heads==2
    assert layer.mlp.experts.intermediate_size_per_partition==384
    return layer

if __name__=='__main__':
    import torch,json,argparse
    from horizonkv_layer_replay import initialize
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    config=initialize();layer=load_rank(config,12,a.rank)
    print(json.dumps({'rank':a.rank,'qheads':layer.self_attn.num_heads,
                     'kvheads':layer.self_attn.num_kv_heads,
                     'expert_intermediate':layer.mlp.experts.intermediate_size_per_partition,
                     'parameter_bytes':sum(p.numel()*p.element_size() for p in layer.parameters()),
                     'parameters':{n:list(p.shape) for n,p in layer.named_parameters()}}),flush=True)
    torch.distributed.destroy_process_group()
