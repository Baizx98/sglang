# Keye Decode Indexer 实验记录

实验产物路径均以 `data/indexer-optimization/results/` 为根。研究目录之外沿用已有 Keye 实现。所有 A6000 的时间均为健康卡 `GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822` 上的单层回放，L40S 用于语义与候选筛选诊断。

| 阶段 | 问题与实验 | 结果与决定 |
|---|---|---|
| 11‑18 初始基线 | 固定 Keye 原生 16 头、64 维、k=2048，对照部署 BF16 分数和各精确选择器。拆开测评分、Top‑K，联合测一次完整调用。 | 128K/B16 的原生打分 412.33 µs，单独选择 100.18 µs。相比只取消分数写回，减少每个物理 Key 的读取字节更有潜力。`synthetic-baseline-early.jsonl` |
| 固定段 Oracle | 用真实 Top‑2048 回溯理想筛选器至少必须访问的固定段，按实际尾长计数。 | 32/64/128K、段长 64 时，最多可跳过比例中位数分别为 35.5%、57.4%、71.5%。据此测试固定段筛选和完整 token 上界。`oracle-fixed-segments.csv` |
| 调优与布局 | 对照 dense 打分、更换选择器、连续 tile 压缩页表、尾部描述符更新及不同读写布局。 | 页表有局部连续性；(最新页表、尾部更新和访问成本都需计入)。B16 混长逐 token 打散时，开发版压缩路径的打分＋选择从连续的 302.25 增至 634.11 µs。`mixed.jsonl` |
| 量化粗排、段筛选与上界 | 对比 INT8 token 粗排、分段粗排、融合段最大值、BF16 样本代表和有证书的恢复。 | 低预算覆盖不足。INT8 token 候选只在部分分布上复现正确集合，同码反例揭示漏选。精确条件需要计算与元数据。各预算和持出结果见[段筛选记录](segments.md)。 |
| 指数无损缓存 | 按物理 token 紧凑存储、原位增量更新、评分内还原，完整调用纳入投影和写入。 | 开发用 128K 层 12。冻结 32K 的 BF16 策略，以及 64/128K 的压缩策略（B4 用 vLLM persistent，B16 用 FlashInfer）。真实多步 append 和包括异常数值、重排、退场在内的边界测试见[缓存实现](packed_cache.md)。 |
| 持出层复测 | 层 36、47，步 2、4，B4/B16；重新测部署版和 0.7.4 FlashInfer，并加入 vLLM Ampere CuTe。每次在通过精确校验的所有部署 scorer 组合中取最快者。 | 新版 128K/B4 四点的配对加速为 1.101–1.106×，B16 为 1.139–1.148×，输出集合逐点完全相同。64K 为约 1.06×/1.13×，32K 的 BF16 回退在 1× 附近。`heldout-{old,new}-{full128,full-short}.jsonl` |
| FlashTopK 式融合与恢复 | 将均匀样本阈值、融合 BF16 打分及候选压实移植到 SM86，阈值过高或候选溢出时走完整恢复。 | 普通、平分、候选不足、短行和碎页单独对照，功能及目标卡测量见[融合实验](fused_threshold.md)。 |
| 隔离压缩与解包 | 在同一选择器、相同打分 tile 上对比 BF16、先显式还原整张 BF16、在寄存器中边打分边解码。 | 128K/B16 同一打分＋选择，BF16 491.78 µs、显式还原 1189.89 µs、边算边还原 422.31 µs。三条路径的计时放在 `ablation-packed.jsonl`，逐条件检查见研究笔记。 |

[完整实验报告、适用范围和直接数据](../../../../my_research_note/projects/horizonkv/branches/main/experiments/2026-11-18_2000_keye-indexer-lossless-results_v01_draft.md)。复现时从仓库根运行 `scripts/indexer-optimization/run_suite.sh --output-dir <新目录>`，保存该目录的元数据、原始 JSONL 和汇总 CSV。历史输入逐组校验可使用 [reproduce_trace_statistics.py](../reproduce_trace_statistics.py)。
