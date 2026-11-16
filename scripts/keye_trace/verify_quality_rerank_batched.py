"""Native GPU Top-K replay with batched membership checks and Parquet output."""
import argparse,json,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from analyze_horizonkv_predictive import load_chunk
from horizonkv_predictive_runtime import inventory,notify
from horizonkv_single_runtime import save
from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer

GPU='GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822'

@torch.inference_mode()
def main(root, batch_chunks=8):
    inventory([GPU],require_idle=True);torch.set_num_threads(4)
    manifest=[json.loads(s) for s in (root/'events/manifest.jsonl').read_text().splitlines()]
    widths=json.loads((root/'run-config.json').read_text())['candidate_widths']
    out=root/f'gpu-replay-batch{batch_chunks}';out.mkdir(exist_ok=False);writer=None;count=ties=max_tie=0;started=time.time();next_notice=started+1200
    try:
        for start in range(0,len(manifest),batch_chunks):
            cells=manifest[start:start+batch_chunks];j=start+len(cells)-1;c=cells[-1]
            with ThreadPoolExecutor(max_workers=4) as pool:
                batch=list(pool.map(load_chunk,[root/'events'/cell['file'] for cell in cells]))
            rows=[row for data in batch for row in data['rows']];steps=len(rows)
            rids=[data['request_id'] for data in batch for row in data['rows']]
            layers=[data['target_layer_id'] for data in batch for row in data['rows']]
            step_ids=[row['step'] for data in batch for row in data['rows']]
            ids=torch.stack([r['candidates'] for r in rows]).cuda().long()
            scores=torch.stack([r['candidate_scores'] for r in rows]).cuda()
            native=torch.stack([r['exact_indices'] for r in rows]).cuda().long()
            lengths=torch.tensor([r['context_len'] for r in rows],device='cuda',dtype=torch.int32)
            cutoffs=torch.tensor([r['native_cutoff'] for r in rows],device='cuda')
            members=torch.zeros((steps,int(lengths.max())),device='cuda',dtype=torch.bool)
            members.scatter_(1,native,True)
            covered=members.gather(1,ids.clamp_min(0)) & (ids>=0)
            blocks=[]
            for width in widths:
                offsets=KeyeIndexer._topk_with_lengths(None,scores[:,:width],lengths.clamp(max=width),2048).long()
                selected=ids[:,:width].gather(1,offsets)
                nonnegative=(selected>=0).all()
                unique=(selected.sort(dim=1).values.diff(dim=1)>0).all()
                hits=covered[:,:width].sum(dim=1)
                final_hits=members.gather(1,selected).sum(dim=1)
                kept=torch.zeros_like(covered[:,:width]);kept.scatter_(1,offsets,True)
                strict=(covered[:,:width]&~kept&(scores[:,:width]>cutoffs[:,None])).sum(dim=1)
                loss=hits-final_hits
                assert bool(torch.stack([nonnegative,unique,(strict==0).all(),(loss>=0).all()]).all()),(c,width)
                values=torch.stack([hits,final_hits,loss,strict],dim=1).cpu().numpy()
                ties+=int((values[:,2]>0).sum());max_tie=max(max_tie,int(values[:,2].max()))
                blocks.append(pd.DataFrame(dict(request_id=rids,layer=layers,
                    step=step_ids,candidate_k=width,candidate_recall=values[:,0]/2048,
                    gpu_rerank_recall=values[:,1]/2048,boundary_tied_tokens_lost=values[:,2],strict_tokens_lost=values[:,3])))
            frame=pd.concat(blocks,ignore_index=True);table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(out/'observations.parquet',table.schema,compression='zstd')
            writer.write_table(table);count+=len(frame)
            if start%64==0 or j+1==len(manifest):
                inventory([GPU])
                save(out/'progress.json',dict(status='running',chunks=j+1,total_chunks=len(manifest),observations=count,tie_cases=ties,elapsed_s=time.time()-started))
            if time.time()>next_notice:
                notify(f'HorizonKV结果核查：GPU精排{j+1}/{len(manifest)}层chunk，{count}组，严格高分遗漏0；CPU合并统计同时处理中。');next_notice=time.time()+1200
    finally:
        if writer:writer.close()
    result=dict(status='completed',chunks=len(manifest),observations=count,strict_losses=0,boundary_tie_cases=ties,max_boundary_tied_tokens_lost=max_tie,
        gpu_uuid=GPU,gpu_name=torch.cuda.get_device_name(),elapsed_s=time.time()-started,
        scope='Native exact Top-K helper on captured candidate scores; no task accuracy or timing claim')
    save(out/'audit.json',result);save(out/'progress.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--batch-chunks',type=int,default=8);a=p.parse_args()
    try:main(a.run_dir.resolve(),a.batch_chunks)
    except BaseException as e:
        save(a.run_dir/'gpu-replay-failure.json',dict(error=str(e),traceback=traceback.format_exc()))
        notify('HorizonKV GPU结果核查停止：'+str(e));raise
