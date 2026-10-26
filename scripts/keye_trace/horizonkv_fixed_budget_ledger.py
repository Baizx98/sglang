"""Full-deployment memory ledger, independent of single-layer free memory."""
import math

def point_budget(ledger,c,length,rank=0):
    g=ledger['device_bytes'];weights=ledger['rank_weights_bytes'][str(rank)]
    tokens=c*length;index=tokens*48*128
    stage=c*2048*1024
    workspace=ledger['workspace_fixed_bytes']+tokens*ledger['workspace_bytes_per_history_token']+stage
    reserve=ledger['context_and_safety_reserve_bytes']
    common_map=tokens*4
    fullmain=tokens*48*1024
    common=dict(concurrency=c,length=length,rank=rank,device_bytes=g,weights_bytes=weights,
        index_bytes=index,workspace_and_stage_bytes=workspace,reserve_bytes=reserve,
        complete_main_kv_bytes=fullmain,budget_kind=ledger['budget_kind'])
    if weights+index+workspace+reserve+common_map>g:
        return dict(common,status='X-index-or-workspace',cache_tokens_per_request=None)
    if weights+index+workspace+reserve+common_map+fullmain<=g:
        return dict(common,status='H',cache_tokens_per_request=length,persistent_main_bytes=fullmain,
                    metadata_bytes=common_map)
    # Each layer has its own mapping, tags/ages, and per-request selection scratch.
    mapping=48*tokens*4+48*c*2048*12
    available=g-weights-index-workspace-reserve-mapping
    cap=max(0,math.floor(available/(48*c*(1024+8))))
    cap=min(length,cap//64*64)
    return dict(common,status='D' if cap>=2048 else 'X-cache-kernel-floor',cache_tokens_per_request=cap,
        cache_over_topk=cap/2048,persistent_main_bytes=cap*c*48*1024,
        metadata_bytes=mapping+cap*c*48*8,
        note='b<k is a current LRU-kernel floor, not a universal hardware limit')
