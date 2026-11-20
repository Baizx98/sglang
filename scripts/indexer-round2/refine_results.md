# Keye 相邻层候选精排

## 输入与计时

回放原 48 条 RULER 请求，32K、64K、128K 各 16 条。目标零基层 36 的既有数据缺少相邻第 35 层输入，因此本次在零基层 1 采用真实相邻层预测。用 step 2 帧的真实 Decode token ID 查原 Checkpoint 的 BF16 embedding，再运行原生第 0 层输入 RMSNorm，复原 text Decode 中第 0 层进入 Attention 的向量。逐条核对 token 等于该请求的 `response.output_ids[2]`。使用第 1 层 Indexer 投影前层向量，在排除当前 token 的第 1 层 Index K 历史上打分。按预测分数降序取 `M-1` 项并将当前 token 插在首位。目标层精排使用第 1 层真实记录的 `attention_input`。按原回放规则从 offset 8 取不同请求，B16 绕回补满。已保存各 batch 的候选逻辑下标与原输入 token。

A6000 矩阵包含三档长度、B4/16、M3072/8192、连续槽、乱序 128-token 块、逐 token 乱序槽，共 36 格。候选与布局对所有后端保持一致。 `refine` 从当前 Q、K、门权及候选就绪开始，计量候选间接读取、BF16 点积与 FP32 门权归约、选择及逻辑 ID 输出。 `project_write_refine` 也包含目标 Indexer 投影和当步 K 写入。真实历史 K 在 GPU 已驻留。图内每轮 24 次调用，预热后做 11 次交错 CUDA event 计时。独立预测和串行过程另计。所有约 32K–128K 历史仍执行前瞻预测打分。本节计时适用于单卡层回放。

强基线包括原生和多 tile 候选打分器、FlashInfer fused page、SGLang fused page、vLLM persistent 加逻辑映射、vLLM Ampere CuTe 的原生候选映射、PyTorch 以及已有无损 12 位 Index K 读取。精确选集要求所有返回 token 属于候选且没有重复，排序满足原生 BF16 评分。在相同第 k 名分数上接受等价选集。另用 PyTorch gather、BF16 矩阵乘、FP32 计算独立判定。

## 精确候选排序

精确 CUDA selector 用有序 FP32 位编码定位第 2048 名。保留严格高于阈值的项，并对阈值同分者分配剩余名额。输出阶段从候选数组直接写逻辑 ID。较窄的 radix 使用 11/7/7/7 位三次细化，较宽的用 11/11/10 位两次细化。另试了在单个评分 CTA 内按物理槽排序，保留逻辑 ID 并交给已有 FlashInfer 选择器。两档 M 的短行、正负零、极端指数和全同分在 A6000 Graph 验证通过。所有评分 tile 的逐位结果通过验证。

在 A6000 测试前按 L40S 16 点试测冻结 64-token、4-warp 评分及 1024 线程、11/11/10 位 selector。A6000 比例为每格最快正确基线的中位延迟除以方法中位延迟。72 个计时格、2592 个精确方法记录均通过两套严格排名参考。

| 区域 | 固定方法 geomean | 固定方法更快 | 每格最快新方法 geomean |
| --- | ---: | ---: | ---: |
| 精排 | 0.9282× | 7/36 | 0.9292× |
| 目标投影、写入及精排 | 0.9797× | 8/36 | 0.9801× |

仅精排区间内，固定方法在 M3072 为 0.9981×，M8192 为 0.8632×。按物理槽的 CTA 内局部排序最佳固定值为 0.7011×。候选间接读取和排序方案没有稳定缩短执行时间。

## 融合目标投影消融

新组合保留原 BF16 GEMM、MRoPE 和 Hadamard，把两种归一化并进一个 Triton kernel。精排路由根据独立 L40S 试测按候选宽度和 batch 冻结。M3072/B4 使用 64-token、4-warp 评分和 vLLM CuTe fused page，M3072/B16 使用相同评分和 vLLM persistent，M8192 使用原生评分和 SGLang fused page。旧对照纳入本节所有已实现的精确精排后端，仍由原生目标投影生成输入。同路由消融只换目标投影。另测单独投影写入区间，并以 Q、K、门权 BF16 位差和输出严格排名两个口径分别验证。

| Batch | M | 每格最快旧组合的中位数 / µs | 融合投影固定路由中位数 / µs | 相对每格最快旧组合 geomean |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3072 | 63.92 | 43.95 | 1.456× |
| 4 | 8192 | 71.89 | 52.61 | 1.367× |
| 16 | 3072 | 67.03 | 47.79 | 1.402× |
| 16 | 8192 | 89.22 | 68.22 | 1.308× |

A6000 全 36 格均更快，逐格比值的 geomean 为 1.3821×，范围 1.3028–1.4633×。固定相同评分和选择器、只替换投影的消融为 1.3920×。单独的投影与写入在 B4 约从 46.2–46.4 µs 降至 26.2–26.3 µs，在 B16 从 41.7–42.9 µs 降至 20.9–21.4 µs。联合路径的收益主要来自归一化阶段。原始行及双参考核验位于 `data/indexer-round2/results/refine-projection-a6000/raw.jsonl`，第 1–36 行。逐格表为同目录的 `paired.csv`。

36 格都与原生投影的 Q、当前 K 和门权逐位一致，最终逻辑下标也都通过 Triton 和 PyTorch 的严格排名。额外在 L40S 独立检查层 1/12/36、step 0/2/4/8、三档长度和 B4/16，共 72 个组合。其中 3 个 B16 组合出现 13、25、16 个 Q BF16 元素变化，K 和门权逐位一致。层 1 的差异出现在 128K、step 8。既有 A6000 全量 Indexer 结果也在层 36、32K/B16、step 2 观测到当前 K 差异。融合投影的精确声明仅覆盖本次 A6000 排名实测点。额外逐位检查表为 `data/indexer-round2/results/refine-projection-crossstep-l40s.csv`。

## 近似候选处理

输入序列已由同一预测器排序。直接前缀路径实际把前 2048 个逻辑 ID 拷贝到连续输出。保护前缀路径将前 1024 或 1536 个逻辑 ID 固定选入，再计算其余候选的真实目标分数，以 FlashInfer 选择补足。候选的排序耗时包含在独立预测计时，精排计时覆盖前缀连续化、需要时的真实分数与候选变换，以及逻辑 ID 输出。

| 候选宽度 | 近似方法 | 精排 geomean | 与精确候选内选集重合 | 与原生全史 Top-K 重合 | 全史重合变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| 3072 | 直接前缀 | 9.90× | 68.17% | 8.77% | −3.60 pp |
| 3072 | 保护 1024 后精排 | 0.79× | 85.15% | 12.37% | 0.00 pp |
| 3072 | 保护 1536 后精排 | 0.85× | 76.91% | 12.37% | 0.00 pp |
| 8192 | 直接前缀 | 16.22× | 29.89% | 8.77% | −19.62 pp |
| 8192 | 保护 1024 后精排 | 0.75× | 66.27% | 28.38% | 0.00 pp |
| 8192 | 保护 1536 后精排 | 0.78× | 48.27% | 27.68% | −0.70 pp |

表内相对全史精确选集的变化，以同候选下精确精排为参照，按格等权平均。第 1 层是候选质量较差的一层。M3072/M8192 的候选覆盖均值在本矩阵约为 12.39%/28.39%。同分时精排在覆盖上有小差异。保护 1024 的全史重合在本次样本不变，返回选集仍出现替换。直接截取前缀的选择局部节省为 12.67×，计入原目标投影和写入后为 1.57×。局部交集与 Attention 代理均无法验证生成任务质量。

忽略全史分数边界的同分，若保护前缀之后最多还有 (k-P) 个原生 Top-K token，则尾部 (k-P) 个精排名额足以选回它们。本批次任何候选集最多包含 720 个原生 Top-K token，保护 (P=1024) 时尾部有 1024 个核实名额，解释了该代理指标没有下降。新加入前缀的低分候选依然可以改变 Attention 输出。

在同一 48 条 C1 请求上，用捕获的两卡 Attention Q 和 Main K/V 离线计算第 1 层 FP32 Attention。对照使用原始 C1 捕获的 Top-K，近似使用 B16 预测候选的前 2048 个逻辑 ID。对照 CPU 输出与原 GPU 输出的平均相对 L2 为 1.09%，近似前缀相对 CPU 对照的平均相对 L2 为 90.36%，平均余弦为 0.6853。32K/64K/128K 的平均相对 L2 为 42.78%/113.00%/115.29%。该代理只替换同状态的选集，使用浮点 CPU GQA、标准缩放和 softmax。逐请求数据见 `data/indexer-round2/results/refine-a6000-frozen/prefix_attention.csv`，脚本为 `refine_attention_proxy.py`。

## 独立预测与同步串行

预测计时从已到达的前层 Attention 输入开始，包括目标 Indexer 的原生投影、历史全量评分、PyTorch FP32 `topk(sorted=True)` 和当前 token 拼接。按现有实现仍计算并丢弃预测 K。同步串行在同一张 CUDA Graph 中把有序候选交给原生候选评分和 FlashInfer 逻辑映射，另测直接前缀及目标层投影写入。独立预测与精排可以重叠多少需要异步运行时的实测。

| A6000、128K 连续槽 | 有序预测 / µs | 预测后直接前缀 / µs | 预测后原生精排 / µs | 预测后原生投影、写入与精排 / µs |
| --- | ---: | ---: | ---: | ---: |
| B4/M3072 | 265.13 | 266.50 | 285.61 | 340.14 |
| B4/M8192 | 314.11 | 316.50 | 341.42 | 397.91 |
| B16/M3072 | 627.54 | 629.76 | 658.69 | 709.55 |
| B16/M8192 | 699.99 | 701.10 | 759.55 | 806.66 |

A6000 的这四格单测排序、逻辑 ID 写出和当前 token 拼接为 105.13–241.83 µs，单测原生精排为 17.75–55.98 µs。直接截取前缀的同步串行延迟仍由预测主导。A6000 连续布局的 12 点位于 `data/indexer-round2/results/refine-predictor-a6000/predictor.csv`。全量样本和方法计时边界见同目录 `predictor.jsonl` 与 `predictor-environment.json`。L40S 的独立 12 点在 `data/indexer-round2/results/refine-predictor-l40s/predictor.csv`。

## 数据与重现

A6000 首轮完整表位于 `data/indexer-round2/results/refine-a6000-frozen/`。 `paired.csv` 保留固定方法与逐格最优的结果，`methods.csv` 保留全部方法与双参考检验，`approximate.csv` 给出逐格延迟与重合指标，`raw.jsonl` 保留原始每轮计时和逐请求质量统计。CSV 的 `raw_line` 是原始 JSONL 的 1 基行号。 `metadata.json` 记录模型、输入 manifest、脚本 SHA-256、设备与库版本。

```bash
export PYTHONPATH="$PWD/scripts/indexer-round2"
export INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/refine_bench.py"
scripts/indexer-round2/run_full.sh --full-prepare --approximate \
  --primary direct_n64w4_radix_wide1024 \
  --output data/indexer-round2/results/refine-a6000-rerun

INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/refine_projection_bench.py" \
  scripts/indexer-round2/run_full.sh \
  --candidates data/indexer-round2/results/refine-a6000-rerun \
  --output data/indexer-round2/results/refine-projection-a6000-rerun

INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/refine_predictor.py" \
  scripts/indexer-round2/run_full.sh \
  --output data/indexer-round2/results/refine-predictor-a6000-rerun
```
