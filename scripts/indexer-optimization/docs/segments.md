# Keye Decode 的整型粗筛与段级准入

## 问题与假设

真实 Top2048 的位置分布决定了连续段的理想跳过率。当前分段实验分别验证段内最大分能否从低带宽的 INT8 评分恢复，以及仅观察段内少数位置是否足以决定准入。所有方案用原始 BF16 K 精排，按原生 SGLang 评分的 Top2048 阈值核查遗漏。参考实现用逻辑 token 位置和物理 slot 映射，保留负 gate 的评分语义，验证不要求同分 token 的输出次序一致。

## 方法

- **Token 粗筛**。每个物理 K 和每个 Q head 各用二次幂 scale 量化至有符号 INT8。整数 Tensor Core 一次算出所有粗分，取 4096 个 token 作 BF16 精排。任何未入围的 token 都不会被复查，因此方法本身是近似的。
- **段内最大分**。连续逻辑 token 按 64 或 128 分段，取该段内 INT8 分数的最大值，保留预算内的整段，再加载原始 K 和精排。非融合路径写回所有粗分并用独立 kernel 做段内最大值，融合路径在评分 CTA 内直接写一个段最大值。S128 消融用相同的 CTA token 数，两条路径只差最大值归约的位置和输出宽度。S64 融合路径的 CTA 从 128 缩到 64，S64 的性能差包含分块差异。
- **段内 BF16 抽样**。每段只计算逻辑偏移 0、16、32、48 四个位置的 BF16 分数，按抽样最大值给 50% 的段准入。该方案扫描总键的约 6.25% 做粗筛，不持久化摘要。漏掉孤立高分 token 的概率需要实测。
- **残差上界剪枝**。设量化后的向量为 `qhat`、`khat`，则点积误差的实数上界为 `||q||·||k-khat|| + ||q-qhat||·||khat||`。按 gate 的符号选取最大化本项的点积端点，再执行单调 BF16 取整、ReLU 和带符号 gate。实现对范数与点积界留出浮点余量，并对 gate 归约另留余量。每行取最多 8192 个均匀抽样 token 的真实 BF16 第 2048 大分作为保守阈值，保留上界等于阈值和上界非有限值的 token。该误差包络对应正常有限的模型输入，并以原生分数逐键检验实现。当前原型静态发射两个 MMA 的整个 tile，通过掩码减少 BF16 K 的读取。

## 处理范围与资源

- INT8 K code 64 B，FP32 scale、原向量范数、重建向量范数、残差范数各 4 B，共额外 80 B/slot。原始 128 B/slot 同时保留，持久空间增加 62.5%。粗筛变体数学上只需 scale，但本原型和上界变体共用 80 B 格式。
- K 摘要由历史 K 单独离线生成。在线每步额外量化新追加的 K，刷新其物理 slot。独立计时每次对 B 个当前 slot 读取 BF16 K 并写摘要。历史预计算一次后可复用，摊销值为预计算总耗时除以复用步数。
- 主计时从 Q、BF16 K、weights、page table、lengths 就绪开始，含动态 Q 量化、粗分、候选生成、BF16 精排、最终 Top2048 及逻辑位置映射。历史构建和独立摘要更新使用分别记录的 CUDA Graph 计时。没有运行模型推理端到端评测。
- 流量按材质、唯一访问数量估算。Tensor Core tile、物理分页、缓存、读合并决定实际显存流量。当前未采 NVIDIA 性能计数器。

## RTX A6000 同卡对照

在健康 RTX A6000 上对同一组 RULER 真实输入运行 11 次试验，每次 12 次 CUDA Graph replay。以下为中位数，单位 µs。原生评分后分别调用 fast_topk_v2、PyTorch、FlashInfer 0.7.4、vLLM Persistent 和 vLLM 的 Ampere CuTe 选择器。表中原生对照取各格中所有试次都满足 BF16 分数及合法索引检查的最快路径。FlashInfer、Persistent 和 CuTe 在这四格都通过，fast_topk_v2 在 128K 两格发生严格高分遗漏。

| B / 长度 | 原生加最快通过的选择器 | Token INT8→4096 | 残差上界 | S64 50% 融合，Recall | S128 50% 融合，Recall |
|---|---:|---:|---:|---:|---:|
| 1 / 32K | 29.61（Persistent） | 106.24 | 105.73 | 140.20，96.68% | 121.77，91.70% |
| 16 / 32K | 129.79（Persistent） | 217.34 | 281.52 | 284.84，96.45% | 282.20，92.43% |
| 1 / 128K | 51.11（Persistent） | 129.37 | 151.72 | 134.66，100% | 130.56，100% |
| 16 / 128K | 490.84（CuTe） | 511.40 | 730.54 | 650.33，100% | 642.99，99.9969% |

Token INT8 与残差上界在这四格均得到与原生 BF16 相同的 Top2048 集合，但耗时高于各格最快通过的原生路径。B16/128K 下，S128 预算 50% 的 99.9969% Recall 对应 1 个严格高分遗漏。保留 S64 25% 段时四格 Recall 为 75.76%–96.63%。在同等分段配置下融合使 S128 路径节省 4.69–69.12 µs，单独比较“到段最大值”的阶段节省 4.27–71.00 µs，说明局部归约收益不足以抵消两阶段评分成本。全历史摘要预计算耗时 7.85–637.44 µs，每步追加摘要耗时 2.13–2.99 µs。以上两项独立于表中的评分到 Top2048 延迟。

同样四格复测 S64 每段抽 4 个 BF16 位置，再精排排名前 50% 的段。下表使用这次单独运行中的原生对照。

| B / 长度 | 原生加最快通过的选择器（µs） | 段内 BF16 抽样（µs） | Recall | 严格高分遗漏 |
|---|---:|---:|---:|---:|
| 1 / 32K | 29.70（Persistent） | 131.93 | 87.94% | 247 |
| 16 / 32K | 129.54（Persistent） | 227.84 | 89.57% | 3,415 |
| 1 / 128K | 50.69（Persistent） | 120.15 | 92.48% | 154 |
| 16 / 128K | 487.42（CuTe） | 429.91 | 90.38% | 3,150 |

BF16 抽样仅在 B16/128K 缩短延迟，约 1.13×，该格有 3,150 个分数严格高于参考阈值的 token 漏选。34 个请求行都未通过原生 BF16 Top2048 的精确校验。

## L40S 诊断

L40S 的同层同批对照使用 RULER 32K/128K、B1/16、层 12、Decode 步 0，每格 5 次试验、每次 8 次 CUDA Graph replay。固定 4096 个预选 token 在这 34 个请求行均得到原生 BF16 Top2048，B16/128K 从原生 BF16 + PyTorch Top2048 的 486.40 µs 降到 403.71 µs，另外三个组合更慢。B16/128K 原生 fast_topk_v2 路径 458.50 µs，和 BF16 参考集合重合约 95.61%，因此与两条严格对拍路径分开记录。

S64 保留 25% 段后集合 Recall 为 75.84%–96.63%，保留 50% 时为 96.48%–100%。S128 保留 50% 时为 91.70%–100%。融合段内最大值在这些组合节省 3.84–41.22 µs，和各自相同分段参数的非融合路径拥有相同 Recall，融合后仍慢于原生 BF16 + PyTorch Top2048。报告 Recall 按槽位数加权，四格之间的范围表示各格聚合结果。

补测将 32K/64K/128K、B4/16、层 1/12/24/36/47 和 Decode 步 7/31 组合为 60 格共 600 个请求行。Token 粗筛和残差界路径均与原生 BF16 参考集合完全相同。S64 预算 25% 的加权 Recall 为 80.79%，S64 和 S128 预算 50% 时各为 97.41% 和 93.94%。在合成量化碰撞测试中，同一维度的三个 BF16 取值 0.984375、1、1.015625 被量化到相同代码，但原生结果不同。Token 粗筛在 8K 的 Recall 为 49.96%–51.32%，32K 时 12.11%–12.62%，严格高分遗漏合计 10,538 个。残差界路径在这 10 行保持正确。

仅评分到 S128 段最大值的配对消融节省 3.71–40.70 µs，说明融合确实减少了独立归约和粗分写回。真实 B16/128K 样例中，段内抽 4 个 BF16 位置把耗时从原生 BF16 + PyTorch Top2048 的 486.91 µs 降到 351.74 µs，代价是 Recall 降到 90.37%。另外三格 87.94%–92.48% 且路径更慢。

L40S 的独立诊断覆盖 5 层、两种长度、B1/16，共 20 格、170 个请求行。INT8 粗分的格内绝对误差均值为 0.0240–0.0626，p99 为 0.0785–0.2062，单点最大值为 0.4636。各格内残差界跳过 35.14%–58.49% 的 BF16 精排键，保留分数逐项相等，所有剪掉的键严格低于原生第 2048 大分。就这组数据，完整扫描 80 B/slot 摘要、读取 BF16 抽样和未跳过的 BF16 原键后，逻辑 key 流量下界仍约为原生的 1.10 倍。20 格中追加摘要均与历史全量刷新逐位一致。

## 复现

在实验仓库根目录运行。以下选择本次 L40S 设备并使用对应实验环境。

```bash
segment_gpu=GPU-79886749-8f57-63a5-8c8e-8a68a085565a
CUDA_VISIBLE_DEVICES="$segment_gpu" .venv/bin/python scripts/indexer-optimization/bench.py \
  --module scripts/indexer-optimization/segments.py --select 'int8_|bf16_sample4' \
  --source real --batch 1 16 --length 32768 131072 --layer 12 --step 0 \
  --trials 5 --graph-iterations 8 --stages --skip-flashinfer \
  --output data/indexer-optimization/results/segment-screening.jsonl

CUDA_VISIBLE_DEVICES="$segment_gpu" .venv/bin/python scripts/indexer-optimization/bench.py \
  --module scripts/indexer-optimization/segments.py --select 'int8_' --check-only \
  --source real --batch 4 16 --length 32768 65536 131072 --layer 1 12 24 36 47 \
  --step 7 31 --request-offset 8 --skip-flashinfer --correctness-repeats 2 \
  --output data/indexer-optimization/results/segment-screening-holdout.jsonl

CUDA_VISIBLE_DEVICES="$segment_gpu" .venv/bin/python scripts/indexer-optimization/segment_diagnostics.py \
  --source real --batch 1 16 --length 32768 131072 --layer 1 12 24 36 47 --step 0 \
  --output data/indexer-optimization/results/segment-screening-diagnostics.jsonl

CUDA_VISIBLE_DEVICES="$segment_gpu" .venv/bin/python scripts/indexer-optimization/segment_diagnostics.py \
  --source quantization-alias --batch 1 4 --length 8192 32768 \
  --output data/indexer-optimization/results/segment-screening-quantization-alias.jsonl
```

L40S 已保留原始 `data/indexer-optimization/results/int8-l40s-{checks,real,diagnostics,holdout,final,quantization-alias}.jsonl`。真实输入来自 `data/horizonkv-motivation/2026-10-21_bs16-v01`，以 rank00 历史 K 与步增量为基础，用各不相同的真实请求组成 batch 后重算 Q 和 gate。

A6000 的对应环境和运行参数见 [run_suite.sh](../run_suite.sh) 和原始结果旁的 `int8-a6000.metadata.json`、`bf16-screen-a6000.metadata.json`。从实验仓库根目录汇总两次独立测量。CSV 每行保留原始文件名，参照延迟来自该文件的相同工作负载。

```bash
.venv/bin/python scripts/indexer-optimization/summarize_segment_results.py \
  data/indexer-optimization/results/int8-a6000.jsonl \
  data/indexer-optimization/results/bf16-screen-a6000.jsonl \
  --output data/indexer-optimization/results/int8-a6000-summary.csv
```
