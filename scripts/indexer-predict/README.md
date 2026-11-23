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

`benchmark.py --verify-only` 走相同的投影、评分和候选检查，跳过 CUDA 事件、图计时和初始化计时。产物只列通过对拍的方法、集合指标与失败原因，不能用于冻结性能路由。

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

## FP32 逐 Key 筛选原型

`--certified-bits 6 8 --certified-capacity 4 8 12` 在基准中另测固定容量的精确候选路径，需要包含 `--quant-formats 6x1 8x1`。粗扫用 INT6/INT8 及每 Key 两个向上舍入的 FP16 范数，同一 Triton kernel 输出近似分数和带符号 gate 的 FP32 上界。种子和压实后的剩余候选使用原有 BF16 候选 scorer。遇到缓冲溢出或原始截止分数同分时，设备上按行回退到原始密集评分及 Torch 有序选择。实际 BF16 读取预算包含种子、复查及回退，驻留包含原始 Key、低位 Key 与范数副缓冲。分数缓冲字节数为持久分数和上界缓冲的大小，另有临时索引、scatter 和选择器开销。CUDA Graph 计时包含路由和回退准备。

`certified_device_audit.py --check-graph` 运行真实输入对拍、范数追加及 CUDA Graph 功能检查，无性能计时。当前原型的运行入口与冻结低位路由独立。

## FP16 评分内前缀直方图

`--fp16-scores --prefix-histogram-bits 9` 为匹配 `--prefix-modes` 的低位格式增加完整路径对照。评分 CTA 直接从同一批 FP16 分数累积 radix 前缀直方图。合并得到第 M−1 名所属桶，再扫描分数并压实其所属桶及更高分位置。超容量的请求走原密集选择。选中分数与原低位分数的最高分多重集合相同，截止同分可能改变逻辑 ID，所以额外覆盖损失仍须实测。与原 FlashInfer 同格式路径的配对计时由摘要导出，另报定长直方图和压实缓冲空间。

`prefix_topm_audit.py --prefix-bits 9 --score-histogram --check-graph` 在真实冻结的 FP16 格式上验证融合、压实、结果与 CUDA Graph。默认冻结矩阵继续使用已登记的原选择器。

## SM86 原生 INT4 评分

`--native-int4 --quant-formats 4x1 --quant-tiles 128` 增加配对评分实现，可加 `--fit-int4 2` 和 `--fp16-scores`。Q 的 INT8 整数分成带符号的高半字节和无符号的低半字节，在同一缓存表示上执行两次 `m16n8k64`，再合成原整数点积。连续扫描保持原 BF16 scale、16 头归约和 FP16/FP32 输出。每条请求增加 1024 B Q 片段及一次打包 kernel，均计入全路径。逐格要求与旧 INT4 分数逐位一致，配对使用同一个 FlashInfer 选择器。FlashInfer 在同分边界可选不同逻辑位置。`nibble_scan_audit.py` 在冻结路由采用单组线性 INT4 的层步上验证两种输出格式、分数逐位一致和 CUDA Graph。该原型通过可选参数测量，冻结矩阵保持原路由。

`--native-int4-fused-query` 让 Q 投影后处理直接输出 MMA 片段，省掉独立 Q 打包与中间 INT8 Q 写回，基准逐格对拍两种实现。使用原校准队列的结果运行 `policy.py replace-native-int4`，对校准更快且满足原覆盖门槛的单组线性 INT4 路由替换实现。输出带旧策略及新校准文件的哈希。将新策略保存到单独结果根目录后运行 `run_frozen.sh test-native-int4 <结果根目录>`，可在原持出队列完整复测。原策略模型生成适配器与这一计时入口独立。

`--native-int8` 为单组 INT8 增加同序归约的 warp 扫描。每个 warp 按 8 个位置加载 K，两次 S8×S8 `m16n8k32` 合成 64 维点积，然后在 warp 内完成 BF16 舍入、ReLU 与 gate 加权。保留 INT8 Q 表示，原 Key 载荷与工作区不增加。用 `--quant-tiles 128 --fp16-scores --quant-formats 8x1` 校准相同真实队列，`policy.py replace-native-int8` 在原策略上登记符合条件的实现替换，`run_frozen.sh test-native-warp <结果根目录>` 同时测量已登记的 INT4/INT8 路线。两个原生变体均与原低位 scorer 对拍 FP16/FP32 分数位模式。

`quality_cases.py` 可将登记的原生实现导出到质量服务配置。服务在 SM86 启用原生整数评分，在其他 GPU 上使用同一表示的 Triton scorer。`verify_server_mma.py` 用真实持出张量调用服务的候选生成与 Key 追加入口，检查原生分数位模式、选中分数及当前保留位置。这个验证不做时延统计。
