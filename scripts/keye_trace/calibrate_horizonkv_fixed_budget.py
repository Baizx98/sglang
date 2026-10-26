"""Count actual TP-rank parameter allocations using meta-device constructors."""
import json,argparse,struct,subprocess
from pathlib import Path
import torch
from horizonkv_layer_replay import initialize,MODEL
from horizonkv_tp2_local import rank_shape
from horizonkv_single_runtime import save
from horizonkv_fixed_budget_ledger import point_budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    config=initialize()
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.models.keye_topk_mask_moe import KeyeVL2MoeForConditionalGeneration
    get_global_server_args().device='meta'
    records={}
    for rank in [0,1]:
        with rank_shape(rank),torch.device('meta'):
            model=KeyeVL2MoeForConditionalGeneration(config)
        records[str(rank)]={name:dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size()) for name,t in model.named_parameters()}
        del model
    ledger=dict(rank_weights_bytes={r:sum(t['bytes'] for t in d.values()) for r,d in records.items()},
        device_bytes=49140*2**20,context_and_safety_reserve_bytes=2*2**30,
        workspace_fixed_bytes=256*2**20,workspace_bytes_per_history_token=16,
        budget_kind='actual TP2 meta parameter bytes incl vision; provisional workspace envelope pending observed peak calibration',
        precision='BF16 modules except explicit FP32 parameters',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        note='Per-rank 49140MiB budget; no pooling; weights remain fully budgeted while replaying one layer.')
    save(a.run_dir/'rank-parameter-ledger.json',records);save(a.run_dir/'budget.json',ledger)
    grid=[point_budget(ledger,c,l,r) for r in [0,1] for l in [8192,16384,32768,65536,131072,261632] for c in [1,2,4,8,16,32,64,128,256]]
    save(a.run_dir/'budget-grid.json',grid)
    print(json.dumps(ledger),flush=True)
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
