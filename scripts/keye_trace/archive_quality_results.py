"""Archive validated metric tables and compact provenance, never raw traces."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import pyarrow.parquet as pq


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root,dest):
    tables=root/'tables';gpu=root/'combined330/gpu-replay-batch16'
    prov=json.loads((tables/'provenance.json').read_text());audit=json.loads((gpu/'audit.json').read_text());join=json.loads((root/'cpu-gpu-join-audit.json').read_text())
    assert audit['observations']==7941120 and audit['strict_losses']==0 and join['cpu_gpu_rerank_different_rows']==0
    dest.mkdir(parents=True,exist_ok=False);files=[]
    def copy(source,name,rows=None):
        target=dest/name;shutil.copyfile(source,target);assert sha(source)==sha(target)
        if rows is not None and target.suffix=='.parquet':assert pq.read_metadata(target).num_rows==rows
        assert target.stat().st_size<100_000_000,(target,target.stat().st_size)
        files.append(dict(file=name,rows=rows,bytes=target.stat().st_size,sha256=sha(target),source=str(source.resolve())))
    for item in prov['outputs']:
        source=tables/item['file'];assert sha(source)==item['sha256']
        copy(source,item['file'],item['rows'])
    for name in ['request-metadata.csv','facts.json','report-derived-facts.json','h-vs-step-counterexamples.csv']:
        copy(tables/name,name)
    copy(tables/'provenance.json','analysis-provenance.json')
    copy(gpu/'observations.parquet','native-gpu-rerank-observations.parquet',7941120)
    copy(gpu/'audit.json','native-gpu-rerank-audit.json')
    copy(root/'cpu-gpu-join-audit.json','cpu-gpu-join-audit.json')
    copy(root/'health-summary.json','sampled-device-health.json')
    copy(root/'combined330/view-audit.json','capture-union-audit.json')
    scripts=['prepare_quality_result_views.py','analyze_horizonkv_predictive.py','summarize_predictive_distributions.py','process_quality_results.py','verify_quality_rerank_batched.py','report_quality_results.py',Path(__file__).name]
    metadata=dict(processed_root=str(root.resolve()),archive_commit_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=files,
        scripts=[dict(path='scripts/keye_trace/'+s,sha256=sha(Path('scripts/keye_trace')/s)) for s in scripts],
        units=prov['units'],statistics=prov['statistics'],required_budget=prov['required_budget'],scope=prov['scope'],
        validation='330 unique requests; old uncommitted request excluded; 47 layers x64 steps; 330 PP boundaries; 7941120 native GPU records joined one-to-one with CPU, coverage and rerank Recall equal; strict loss0; all archive hashes and Parquet row counts verified',
        excluded_temporary_outputs=['combined330/gpu-replay','combined330/gpu-replay-batch8','combined330/superseded-worker-start-failure.json'])
    (dest/'archive-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# 330请求详细质量数据包','',
        '仅保存已验证指标与元数据。原始trace/日志/输入token IDs留在keye_sglang；本包不是正式任务得分或性能结果。',
        '', '330请求=52暂停前完成+278续跑，118校准/212测试；47层×64步=992640层步。方法3955050行、候选7941120行、Attention2977920行、表示992640行。',
        '', '## 优先阅读','',
        '- [总来源与验证](archive-provenance.json)：逐文件hash/来源、处理脚本hash、配置与范围。',
        '- [处理来源与原始chunk hash](analysis-provenance.json)：两批采集审计、字段类型及所有处理表。',
        '- [逐请求元数据](request-metadata.csv)、[CPU/GPU逐行核查](cpu-gpu-join-audit.json)、[设备采样](sampled-device-health.json)。',
        '- [集合/误差关联](report-derived-facts.json)、[h未胜前步复用的3个反例](h-vs-step-counterexamples.csv)。',
        '', '## 字段、单位与统计口径','',
        '`request_id/layer/step/method或candidate_k`是逐观测主键；layer/step零基。dataset/task_name/split/length_bucket来自冻结清单，长度和预算单位tokens。Recall/Jaccard/Precision/cosine无量纲，方差按原0–1值计算（fraction²），L2为比值，attention_max_abs为原生输出单位，score MAE为Indexer分数单位。',
        '', '方法主比较step1–63，其余step0–63；`all47`为全部47目标层，`historical42`为预先指定的3及5–45层，不能据其均值声称全层结果。请求均值先请求内平均再等权汇总；每层的分布另存。跨任务/长度混合的数据集均值依赖当前抽样比例，各task×length结果都保留。',
        '', '所有连续指标提供count/missing_count/mean/median/variance(ddof1)/std/minimum/maximum/P01/P05/P10/P25/P50/P75/P90/P95/P99，linear分位数；缺失或单样本方差保留NA。池化层步相关，统计不是CI；同文档多题也可能相关。Recall重点看下尾，误差重点看上尾，不能只读均值。',
        '', '`fully_covered`的均值是候选全包含原生Top-k的比例。`candidate_precision`分母为effective_candidates，`missing_native_tokens`为遗漏数量。Required-M是实测离散网格上的最小满足值，超过8192为NA并记录censored_*；必须同时读删失比例，不能把条件均值当全样本所需预算。',
        '', 'Spearman仅step0/15/31/63采样，共62040非空，其余有意未测；没有Attention概率质量或完整logits KL。Attention是同Q/K/V下对原生Top-k DSA的输出比较，预测并未替换模型主轨迹。',
        '', '## 文件命名','',
        '`*-observations.parquet`保留逐层步；`*-request-layer.parquet`为逐请求×层；`*-task-layer.parquet`为dataset×task×length×split×层；`*-by-request.csv`为请求内均值；`*-request-distribution.csv`为这些请求均值的分布；`*-pooled-distribution.csv`为相关层步分布；`*-dataset-request-distribution.csv`再按dataset/split汇总请求均值。native-gpu文件为独立原生helper回放。',
        '', '## 完整文件索引','', '| 文件 | 行数（如适用） | 字节 |','| --- | ---: | ---: |']
    for item in files:lines.append(f'| [{item["file"]}]({item["file"]}) | {item["rows"] if item["rows"] is not None else "—"} | {item["bytes"]} |')
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(files=len(files)+2,bytes=sum(f['bytes'] for f in files),destination=str(dest))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.destination.resolve())
