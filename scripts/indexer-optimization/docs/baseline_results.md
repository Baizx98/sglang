# Keye 打分与选择基线

## 输入和算术合同

每个 Decode 请求有一条当前查询，16×64 BF16。每个既往位置使用一条 64 维 BF16 Index Key；16 个查询头共享它。物理槽来自逻辑页表。以 `keye_indexer_score_paged` 作为部署分数定义，并与 PyTorch 分块参考单独核对。选择器须返回长度 2048 的 int32 逻辑下标，有效下标唯一；短行剩余位置填 -1。正确性校验同时检查所有严格高于第 k 名分数的位置都被选中，并且被选中的每个位置均达到该分数。重复执行的集合及顺序、与参考集合的交集另行记录。

## 合成分解

健康 RTX A6000，UUID `GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822`，SM86、TP1。固定 seed 20261014；每个实现的 CUDA Graph 有 12 次调用，预热后做 11 轮交错计时。下表是 SGLang 优化打分器和 fast_topk_v2 各自的图内中位数及联合实测，单位 µs。L2 为 6 MiB，32K/B1 的 4 MiB Key 可以受益于缓存驻留。

| 长度 | B | 打分 | 选择 | 打分＋选择 |
|---:|---:|---:|---:|---:|
| 32K | 1 | 5.97 | 21.50 | 27.05 |
| 32K | 4 | 28.93 | 24.83 | 53.16 |
| 32K | 16 | 105.56 | 26.37 | 133.12 |
| 64K | 1 | 15.02 | 31.06 | 45.48 |
| 64K | 4 | 54.53 | 33.71 | 89.69 |
| 64K | 16 | 207.87 | 43.43 | 254.55 |
| 128K | 1 | 29.27 | 47.27 | 75.69 |
| 128K | 4 | 106.06 | 63.40 | 176.55 |
| 128K | 16 | 412.33 | 100.18 | 516.44 |

32K/B1 时选择耗时高于打分。到 128K/B16 时打分约 412 µs，读取长历史成为可优化的大项。该阶段使用的 PyTorch 是 2.9.1+cu130，Triton 3.5.1，SGLang kernel 0.4.1.post1，driver 580.126.09。每轮原始计时位于 `data/indexer-optimization/results/synthetic-baseline-early.jsonl`。

## 真实请求与布局

RULER 输入来自 `data/horizonkv-motivation/2026-10-21_bs16-v01` 的 Keye 层输入和增量 Index Key。原始采集为 A6000+L40S TP2。每个 batch 从不同 C1 请求合并，在 A6000 单卡重算一层。B>1 时重做投影会改变个别 BF16 舍入，因此所有后端在同一组重算 Q/K/门权上对照。另可用逐请求投影复现 C1 捕获的分布。相同输入上对测 PyTorch、SGLang fast_topk_v2、FlashInfer ragged Top‑k 和 vLLM persistent Top‑k；历史复测与最终版库区分记录。

开发测试为层 12、step 1、连续布局的 128K 桶。B4 下，部署打分器配 vLLM persistent 的完整 Indexer 用时 188.67 µs，初版 12 位缓存路径计入写尾行后为 175.19 µs。B16 下，最快原生组合用 FlashInfer 0.6.8.post1，分别为 544.85 和 468.31 µs。此次固定回放一个当前位置，旧 API 还保存用于对照的 BF16 缓存。完整调用计入投影和当前 K 写入。原始记录位于 `results/full.jsonl`（所有 results 路径均相对实验产物根 `data/indexer-optimization/`）。

在 32K/B1 上轮换四个真实请求，Key 合计 15.74 MiB。部署打分从反复读取同一请求的 5.97 µs 上升到 10.15 µs。因此小 batch 结论应说明缓存条件。混合 32/64/128K、B16 的同一批逻辑输入改用随机物理槽时，开发版压缩尾写＋vLLM 的打分和选择用时随布局明显变化：

| 物理布局 | 原生＋vLLM / µs | 压缩＋vLLM / µs |
|---|---:|---:|
| 连续 | 356.35 | 302.25 |
| 打散 128-token 块 | 356.52 | 305.24 |
| 逐 token 打散 | 358.74 | 634.11 |

以上来自 `results/mixed.jsonl`。另对已有 32K/C4 的 HBM 运行 trace 做地址核查，在 20 个请求的 647093 个相邻逻辑 token 对中，可直接验证 641523 对的物理槽也相邻，覆盖全部相邻对的 99.14%。未观察到的地址未插补。该检查使用旧部署分配器与独立请求组，适用于解释连续布局的由来；128K 回放的页表仍按指定布局重建。原始映射来自 `data/horizonkv-motivation/2026-10-15_capacity-c4-v01/hbm/trace/rank_00.{jsonl,bin}`，逐请求统计来自 `results/observed-slot-adjacency.csv`。

## 最终复测

B0 使用完整 PyTorch 分块评分加 torch.topk。真实层 12、步 0、32K/B1 的评分为 159.40 µs，联合选择为 209.07 µs（`torch-b0.jsonl`）。同一目标卡上的主表另按条件选取最快通过精确校验的部署 scorer＋selector。128K/B4 由 vLLM persistent 最快，B16 则是 Ampere CuTe DSL。在 0.7.4 环境的层 36/47、步 2/4 上，完整基线 188.08/543.10 µs，冻结的压缩方案分别为 170.37/475.26 µs；逐点及两种 FlashInfer 版本见[结果报告](../../../../my_research_note/projects/horizonkv/branches/main/experiments/2026-11-18_2000_keye-indexer-lossless-results_v01_draft.md)。

同一批混长输入（B16，层 47、步 4）在连续、128-token 块打散和逐 token 打散时，最新版基线完整 Indexer 为 404.4/405.4/410.7 µs，压缩路径为 362.5/363.0/692.8 µs。该测试从投影开始，开发阶段的 `mixed.jsonl` 从分数路径开始；两组数字对应各自的执行边界。
