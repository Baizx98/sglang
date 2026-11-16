"""Render a Chinese evidence report from the frozen processed tables."""
import argparse,json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd


def table(frame,formats=None):
    formats=formats or {};lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        cells=[]
        for col,v in zip(frame.columns,row):
            if pd.isna(v):cells.append('NA')
            elif col in formats:cells.append(formats[col](v))
            elif isinstance(v,float):cells.append(f'{v:.6f}')
            else:cells.append(str(v))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main(tables,gpu,report,archive_link):
    pct=lambda v:f'{v*100:.2f}%'
    timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    facts=json.loads((tables/'facts.json').read_text());native=json.loads((gpu/'audit.json').read_text())
    method=pd.read_csv(tables/'method-all47-by-request.csv');candidate=pd.read_csv(tables/'candidate-all47-by-request.csv')
    attention=pd.read_csv(tables/'attention-all47-by-request.csv');meta=pd.read_csv(tables/'request-metadata.csv')
    test=method[method.split=='test'];means=test.pivot_table(index=['dataset','request_id'],columns='method',values='recall')
    summary=means.groupby('dataset').mean()[['previous_layer','previous_step','horizonkv']]
    summary['n']=means.groupby('dataset').size();summary=summary[['n','previous_layer','previous_step','horizonkv']].reset_index()
    summary.columns=['数据集','测试请求','前层直接复用','前步直接复用','前层h预测']
    cs=candidate[candidate.split=='test'];coverage=cs.pivot_table(index=['dataset','request_id'],columns='candidate_k',values='recall_at_M').groupby('dataset').mean()
    coverage=coverage[[2048,3000,3072,4096,8192]].reset_index();coverage.columns=['数据集','M2048','M3000','M3072','M4096','M8192']
    ruler=cs[cs.dataset=='RULER'].pivot_table(index=['length_bucket','request_id'],columns='candidate_k',values='recall_at_M').groupby('length_bucket').mean()
    ruler=ruler[[2048,3000,3072,4096,8192]].reset_index();ruler.columns=['RULER长度桶','M2048','M3000','M3072','M4096','M8192']
    # Attention quantiles use the underlying observations; quantiles of request
    # means have a different meaning and are exported separately.
    att=pd.read_parquet(tables/'attention-observations.parquet');att=att[att.split=='test']
    at=att.groupby(['dataset','method']).agg(mean_cos=('attention_cosine','mean'),min_cos=('attention_cosine','min'),l2_mean=('attention_relative_l2','mean'),l2_p90=('attention_relative_l2',lambda x:x.quantile(.9)),l2_p99=('attention_relative_l2',lambda x:x.quantile(.99)),l2_max=('attention_relative_l2','max')).reset_index()
    diagnostic=[]
    for scope,part in [('all47',att),('historical42',att[att.layer.isin([3]+list(range(5,46)))])]:
        for dataset,group in part[part.method=='horizonkv_M3072'].groupby('dataset'):
            diagnostic.append(dict(dataset=dataset,scope=scope,mean_cosine=group.attention_cosine.mean(),min_cosine=group.attention_cosine.min(),l2_p99=group.attention_relative_l2.quantile(.99),l2_max=group.attention_relative_l2.max()))
    diagnostic=pd.DataFrame(diagnostic)
    tail=pd.read_parquet(tables/'candidate-observations.parquet',columns=['request_id','dataset','split','layer','step','candidate_k','recall_at_M'])
    tail=tail[(tail.split=='test')&tail.candidate_k.isin([3072,8192])]
    tt=tail.groupby(['dataset','candidate_k']).agg(mean=('recall_at_M','mean'),median=('recall_at_M','median'),p01=('recall_at_M',lambda x:x.quantile(.01)),p10=('recall_at_M',lambda x:x.quantile(.1)),minimum=('recall_at_M','min'),fully_covered=('recall_at_M',lambda x:(x==1).mean())).reset_index()
    cross=tail[tail.candidate_k==3072].merge(att[att.method=='horizonkv_M3072'][['request_id','layer','step','attention_relative_l2']],on=['request_id','layer','step'],validate='one_to_one')
    low=cross[cross.recall_at_M<.9];high=cross[cross.recall_at_M>=.99]
    association=dict(low_recall_observations=len(low),low_recall_low_error_fraction=float((low.attention_relative_l2<.01).mean()),high_recall_observations=len(high),high_recall_over_1pct_error_fraction=float((high.attention_relative_l2>.01).mean()))
    scope_counts=meta.groupby(['dataset','split']).size().unstack(fill_value=0).reset_index()
    budget=pd.read_csv(tables/'required-budget-by-task.csv')
    worst=att[att.method=='horizonkv_M3072'].nlargest(10,'attention_relative_l2')
    report.write_text(f'''---
title: 330请求跨层h预测与Attention质量完整结果
project: horizonkv
research_branch: main
created: {timestamp} +08:00
updated: {timestamp} +08:00
version: v01
status: draft
authors:
  - bzx
source_repos:
  - repo: /home10T/bzx/workspace/keye_sglang
    commit: 3189d4892
related_files:
  - 2026-11-13_1120_predictive-quality-expansion-plan_v01_draft.md
  - 2026-11-13_1150_predictive-quality-expansion-execution_v01_draft.md
  - 2026-11-16_1008_predictive-quality-resume_v01_draft.md
supersedes: null
---

# 330请求完整选择质量与Attention误差结果

## 1. 问题、假设与完成范围

验证前层归一化h经过目标层Indexer预测是否优于直接复用、候选扩展如何随任务/长度变化、集合遗漏是否对应Attention输出扰动。完整采集并合并330/330条详细影子请求：11月13日完成52条，11月16日续跑278条；仅使用旧52条已提交请求，旧第53条在途产物全部排除，由新批重采。

本次已结束的是330条详细影子采集及其处理。2359条是冻结输入总清单；1747条独立测试×最多8配置的自由生成任务评分、IndexCache/MLI机制对照、校准LM-loss选层、完整logits KL和Attention概率质量尚未完成。不能把本报告的Recall/Attention误差称为任务准确率或端到端收益。

## 2. 方法、环境与完整性审计

Keye-VL-2.0-30B-A3B，BF16，健康A6000 `GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822`与L40S `GPU-79886749-8f57-63a5-8c8e-8a68a085565a`，PP2/TP1、C1 eager、seed20261113。context131328，逐请求65个生成token（首token来自Prefill，64步Decode），ignore_eos仅为固定长度诊断。模型实际Attention始终消费原生精确Top-2048，历史Key使用真实目标层状态，不跨层共享KV。预测为当前token预留一位，再对历史目标层Key评分。

- 15510个唯一请求层文件，992640个层步；方法记录{facts['method_rows']}行，候选{facts['candidate_rows']}行，Attention{facts['attention_rows']}行，表示/评分{facts['representation_rows']}行。
- 330条PP23→24边界逐步核对；所有候选评分与同状态原生评分差为0，15510个首步自检通过。
- CPU稳定精排严格高分遗漏0；GPU原生Top-K helper核查{native['observations']}组，严格高分遗漏{native['strict_losses']}，边界同分差异{native['boundary_tie_cases']}组、最大{native['max_boundary_tied_tokens_lost']}token。与CPU逐行连接后，7941120组覆盖率和精排Recall全部相同（最大差0）；同分差异指与原生真值集合的边界差异，不是CPU/GPU分歧。
- 新旧token输入、manifest、chunk、处理脚本和输出的来源/哈希在[数据包]({archive_link})，不把中断记录混入完成集。采集实现459f448bc/a0398779b，模型/预测代码两批相同；处理3189d4892及后续报告脚本提交见provenance。

详细子集的校准/测试组成：

{table(scope_counts)}

LongBench涵盖12子任务、RULER13任务×6长度、BFCL4类别，LBv2按领域抽样30题；公开MRCR只使用每needle的0号shard，并未核实为IndexCache的MRCR v2。GraphWalks仅选完整输入能适配预算的实例。正式任务输入清单仍保留2359条，不把这330条冒称全量官方benchmark结果。

## 3. 统计口径

以下主结果使用独立测试子集，校准集另表保存。方法比较使用共同step1–63，候选/Attention使用step0–63。数据集表先在每条请求内平均47层和相关步，再等权平均请求；RULER每个任务×长度一条测试请求，长度表每点13条。表中n是请求数，不是独立层步数。

每项指标导出count/missing、均值、中位数、样本方差(ddof1)、标准差、min/max、P1/P5/P10/P25/P50/P75/P90/P95/P99；分位数为linear定义。不足两点方差NA，未测Spearman不填0。逐层/逐请求层、任务×长度×split、请求均值分布、池化层步分布全部保存；池化层步相关，不能当独立样本置信区间。

全47层是主口径，历史42层（3、5–45，零基）仅作预先指定消融，不因本轮结果删层。候选Precision的分母是实际有效候选数；Required-M只在8个实测M上取最小值，超过8192右删失，不能当8192截顶平均。Attention相对L2以原生精确Top-k DSA输出为分母，不是dense Attention。

## 4. 测试请求上的选择质量

### 4.1 三种方法的Recall@2048

{table(summary,{c:pct for c in summary.columns[2:]})}

全330条的请求平均中，h预测高于直接前层复用{facts['h_beats_previous_layer_requests']}条，高于前步复用{facts['h_beats_previous_step_requests']}条；这是描述性胜负计数，包含校准集，不能与上表测试均值混为同一口径。

### 4.2 扩候选后的覆盖率

{table(coverage,{c:pct for c in coverage.columns[1:]})}

RULER控制任务构造、按长度展开的测试结果：

{table(ruler,{c:pct for c in ruler.columns[1:]})}

短上下文M超过全部历史时100%是全历史饱和点，不能据此宣称固定小候选可泛化到长输入。八月BFCL桥接M3072覆盖99.122%、历史42层精排99.837%已复现，说明旧高覆盖可重现；本轮RULER与真实任务扩展仍需分别报告长度和层尾部。上述差异尚未被正交实验完全归因，不单独归因于长度或PP。

### 4.3 覆盖尾部：均值与完全覆盖不是一回事

下表是测试集相关层步的描述性分布，不是请求均值分布或CI：

{table(tt)}

Required-M@90/95/99/99.9及各自右删失比例在required-budget-by-task.csv和逐层步Parquet中。读取所需预算的均值/分位数时必须同时读censored_*_mean，它们是未达目标的比例，而非可忽略的缺失。

## 5. 同状态Attention扰动

下表汇总测试层步，cosine和相对L2均无量纲；L2=0.01为1%相对误差。P99是误差上尾，min cosine/max L2保留所有层：

{table(at)}

预先指定的历史42层诊断（仅M3072、测试层步；不作为自动部署mask）：

{table(diagnostic)}

全层与42层必须并列；下面的极大误差集中在零基层1，不能因均值好或事后删层而隐藏。其他层/任务的尾部也保存在逐层表中，仍需独立质量验证。

M3072的最坏测试样本入口：

{table(worst[['request_id','dataset','task_name','layer','step','attention_cosine','attention_relative_l2']])}

集合指标与输出误差关联的描述性检查：M3072测试层步中，Recall<90%的{association['low_recall_observations']}组里，L2<1%的比例为{association['low_recall_low_error_fraction']*100:.2f}%；Recall≥99%的{association['high_recall_observations']}组里，L2>1%的比例为{association['high_recall_over_1pct_error_fraction']*100:.2f}%。该交叉统计只说明两个指标并非一一对应，不能代替任务质量；没有测Attention概率质量，不能断言遗漏一定是低权重token。

零Attention输出范数记录{facts['zero_attention_norms']}组。h/score cosine、相对L2、MAE/NMAE及Spearman也已按上述层级导出；Spearman仅step0/15/31/63测量，非空{facts['spearman_sampled_rows']}行，其他步为有意未测，不能补成0。

## 6. 对原假设与相关工作的结论边界

已确认：完整330请求的选择和同状态Attention证据可追溯；候选扩展收益应分任务、长度和层查看；精排能保留严格高分候选真值，但不能补回候选外遗漏。高平均Recall或cosine均不足以证明每层/每步低误差。

需要正式任务评估才能判断：即便某些复用方案的集合Recall较低，它们是否仍保持答案质量；h预测更高Recall是否换来更好的最终质量；校准选层/精确回退能否控制累积扰动。IndexCache与MLI文献的任务分数与本报告的影子Recall不是同一指标，模型、选层和训练条件也不同；本轮未实现它们的完整任务对照，不声称已解释其全部收益。

下一步仍按已批准方案：仅用校准数据冻结层配置和协议，再开展近似路径真正参与Decode的配对任务评分，区分原生基线、统一候选、校准回退及IndexCache/MLI适配。不得根据这里看到的测试尾部修改规则后继续把同一测试集当未见数据；如改规则需另留独立验证。

## 7. 验证与归档

本次没有新增模型输入采集，使用已有完整trace进行后处理及原生GPU helper核查。所有原始日志/trace留在实验仓库；[数据包索引]({archive_link})提供字段、单位、源路径、配置、代码/脚本hash、处理方法与验证状态。数据包包含逐观测Parquet、逐请求/逐层和分布统计CSV/Parquet；正文draft.md未改写。

本报告不报告混卡性能、自由生成任务得分、logits KL或异步重叠收益。设备温度/显存采样只作健康诊断，不作峰值或成本效率结论。先前读取/批量处理的受控优化临时产物未纳入归档，最终完整审计是本次结果依据。
''')
    savepath=tables/'report-derived-facts.json';savepath.write_text(json.dumps(association,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',type=Path,required=True);p.add_argument('--gpu',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--archive-link',required=True)
    a=p.parse_args();main(a.tables,a.gpu,a.report,a.archive_link)
