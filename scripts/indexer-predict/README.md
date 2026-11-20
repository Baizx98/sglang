# Keye 跨层 Full Indexer 实验

预测输入为真实前层归一化 hidden state，历史为实际目标层的 BF16 Index Key。每个方法输出一个已知的当前逻辑位置和历史 Top-(M−1)。当前 Index Key 只能由真实目标层生成并追加。目标层精排和 Main KV 回取都在计时区间之外。

## 复现

在实验仓库根目录运行。先使用已有 0→1 层真实输入验证环境。

```sh
CUDA_VISIBLE_DEVICES=GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822 \
  .venv/bin/python scripts/indexer-predict/verify.py
INDEXER_FULL_TARGET="$PWD/scripts/indexer-predict/benchmark.py" \
  scripts/indexer-round2/run_full.sh --existing --length 65536 131072 --batch 4 16 \
  --candidates 3072 8192 --fp16-scores --segment-audit 32 64 128 \
  --output data/indexer-predict/reproduction-pilot
.venv/bin/python scripts/indexer-predict/summarize.py \
  --inputs data/indexer-predict/reproduction-pilot/results.jsonl \
  --output data/indexer-predict/reproduction-summary
```

真实相邻多层输入用 `capture.py` 在 A6000 与 L40S 的 PP2/TP1 服务上采集。基于已有质量实验清单和 RULER 固定生成输入，用 `freeze_requests.py` 冻结 64K、128K、近 256K 每组 4 条校准与 16 条持出。采集保留实际前层 hidden state、目标层真实 Top-2048、位置、历史 Index Key 与新增 Key，不保存 Main KV。输入采集与 A6000 单卡性能测量使用独立进程和时段。

```sh
.venv/bin/python scripts/keye_trace/prepare_ruler_gate_e0.py \
  --ruler-repo data/external/RULER --output-root "$PWD/data/indexer-predict/reproduction-ruler" \
  --model /Tan/model/Keye-VL-2.0-30B-A3B --lengths 261376 --tasks niah_multikey_1 \
  --num-samples 20 --seed 20261120
.venv/bin/python scripts/indexer-predict/freeze_requests.py \
  --previous data/horizonkv-predictive/2026-11-13_1150_quality-expansion/deep-inputs.jsonl \
  --near256 data/indexer-predict/reproduction-ruler/261376/niah_multikey_1/validation.jsonl \
  --output data/indexer-predict/reproduction-cohort
.venv/bin/python scripts/indexer-predict/capture.py \
  --requests data/indexer-predict/reproduction-cohort/near256-gate.jsonl \
  --output "$PWD/data/indexer-predict/reproduction-capture"
INDEXER_FULL_TARGET="$PWD/scripts/indexer-predict/benchmark.py" \
  scripts/indexer-round2/run_full.sh --capture data/indexer-predict/reproduction-capture \
  --length 262144 --batch 1 --layers 1 12 24 47 --candidates 3072 8192 \
  --output data/indexer-predict/reproduction-near256-pilot
```

只有写入 `completed.jsonl` 并通过步骤与张量尺寸核验的真实请求会进入回放。测试用 `--split test`。回放 B16 使用 16 个不同请求。分子和分母均从预测 hidden state 开始，默认采用同一种跳过预测 K 的 Q/gate 实现。历史布局基线包括分页 BF16、同等连续 BF16、原有逐 token 指数压缩，以及可运行的 Torch、FlashInfer 和 vLLM CuTe 选择器。有边界同分时也要求匹配原预测路径的 PyTorch 有序 Top-M 集合。

## 指标

- `true_top2048_intersection` 为预测候选与采集轨迹上目标层真实 Top-2048 的交集个数。额外候选损失为精确预测交集减去测试方法交集，再除以 2048。与精确预测的 Top-M 集合重合独立保存。
- `logical_key_bytes` 是每次扫描触及的逻辑 Key 表示、其量化/例外元数据和已触及例外行的最小唯一字节数。它与缓存分配、FP32/FP16 分数输出及选择器工作区分列。GPU 实际 DRAM 事务由硬件计数器另行测量。
- 初始化使用同步墙钟，Append 使用真实目标层 Key 单测。稳态完整预测用 CUDA Graph 与 CUDA Event 测量。按格先取重复中位数，再比较逐格最快正确基线。
- 段级审计以已知的精确第 M−1 名做阈值上限分析，或给保留段内的键精确评分后观察候选覆盖。其结果是离线可行性证据。

## 后层质量和新路线

`--fit-int4 2` 比较在编码时拟合两轮最小平方尺度的 INT4 单组和双组，评分同普通 INT4。开启 `--progressive 1.25 1.5 2` 后，从 INT6 的高 4 位生成初选，再只读取入围位置的额外两位重评分。两次选择均计入。需要在 `--quant-formats` 中包含 `6x1`。加 `--exact-tuning` 可以对比掩码解码和 L2 缓存修饰符。无损停用任何造成预测集合差异的路径。

`--fp4` 测试 E2M1。用 `pq_fit.py` 在校准输入上拟合 256 个共享二维码字，`--pair-pq` 测试 34 B/token（32 B 码字＋2 B scale）的表示。评分时读取 512 B 码本，把每个码字还原为两个 INT8 分量再计算点积。编码额外使用 64 KiB 查找表。`codec_coverage.py` 可比较格式对真实任务候选的影响，且不记录性能。`--direct-paged-build` 直接把真实物理页转成紧凑缓存，初始化不分配中间 BF16 slab。

`temporal_oracle.py --rolling` 使用真实相邻步，给出上一轮阈值及段筛查的离线诊断，并检验只更新被评分段时的滚动误差。性能回放的 `--temporal 0.25 0.5 --temporal-segments 32` 使用已准备的真实上一轮段分数作为状态，在保留段上用**当前** Q 重扫。部分刷新与全扫刷新各自计时，完整周期的性能需按实际刷新次数核算。时间路由和全历史量化路由分别报告。

`policy.py freeze` 只读取校准结果，按长度、层和 M 固定方法。用 `policy.py test` 检查预先指定的 held-out batch/step 矩阵和请求不重叠，再报告加速及额外漏项。试验门槛默认是额外漏项均值 ≤1 pp 与 P95 ≤3 pp。两个门槛都属于同状态候选覆盖，服务任务结果另计。

采集结束且 A6000 空闲后，按以下顺序运行冻结矩阵。两次运行共用同一个输出根目录和实际码本。每格同时对拍 Q-only 全融合后处理、仅归一化融合和原生投影，分子分母共用通过逐位核验的最高融合程度。全融合保留原生 MRoPE 两半不同的 BF16 FMA 舍入，再做 Hadamard。低位全扫把分组 INT8 Q 编码放入该 epilogue，每格对拍 Q codes、FP32 scale 和完整历史分数。`projection_audit.py` 可单独复核真实输入的编码一致性。

```sh
.venv/bin/python scripts/indexer-predict/pq_fit.py \
  --capture data/indexer-predict/capture-64-128k-calibration \
  --output data/indexer-predict/pq-codebooks-v1
bash scripts/indexer-predict/run_frozen.sh calibration data/indexer-predict/frozen-run
bash scripts/indexer-predict/run_frozen.sh test data/indexer-predict/frozen-run
```

`capture.py --quality-policy` 可用 PP2 服务运行真实长上下文 RULER 生成。请求需要实际参考答案，非空；每条以原生、精确 BF16 预测及低位预测三种模式在同一脚本上生成。策略 JSON 有 `candidate_width`、`append_reserve`、`default`、可选逐层 `layers` 和 `audit_layers`。逐层可配置 `method=native/bf16/quantized`，低位模式可给 `bits`、`groups`、`fit_steps`、`oversample`、`tile`、`half_scores`。服务实际执行候选内目标层选择并生成文本。它保留原生 BF16 Key 作实际精排，额外维持低位预测缓冲，服务时延不用于性能结论。

`quality_cases.py` 用冻结路由为每个长度桶导出 RULER 生成策略及持出请求。默认每桶按 token 哈希选择一条独立请求，M3072，仅已校准的目标层启用预测。紧凑路径使用 FlashInfer 选择器，替代计时选择器在边界同分时可能产生不同候选。该配置适合实际输出的诊断，较大任务集需要单独报告。
