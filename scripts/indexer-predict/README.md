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
