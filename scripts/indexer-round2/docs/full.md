# 完整 Decode Indexer 实验

完整路径从当前层 `attention_input` 开始，包含 Q/K/gate 投影、归一化、MRoPE、Hadamard、当前 Index Key 写入、历史评分和逻辑 Top-2048 输出。A6000 单卡使用 CUDA Graph，预热后随机交错各方法，记录设备事件的中位数和 MAD。32K、64K、128K × B4/B16 × 连续槽、128-token 块打散、逐 token 打散共 18 格。输入取 Keye 真实请求，层 36、步 2、请求偏移 8。冻结策略和矩阵见 [full-policy.json](../full-policy.json)。

各格从通过原生评分及独立 PyTorch 评分的实现中取最强对照。包含 SGLang 原生评分、Triton tile 调优评分、FlashInfer 0.7.4、vLLM persistent 和 Ampere CuTe 选择器，以及第一轮无损位打包。选择校验检查有效下标、重复、选集、严格高分遗漏和允许的边界同分。独立槽布局也给原生评分和旧压缩评分建立同位置对照。

## 独立 Index Key 存储和投影融合

`slab_index.py` 在请求接纳时为每行分配逻辑下标连续的 GPU 存储。逻辑下标加该行的起始偏移即可寻址，评分读不到 Main KV 的页表。Main KV 保留自身原有映射，返回逻辑 Top-2048。启动阶段从原源页表复制有效 Index Key，或在连续布局上做无损位打包。稳态只保留一份各自的 Index Key 数据，初始化期间可能同时持有原表示、重排缓冲和目标表示。当前测试包含从现有 Index Key 重建后的稳态路径，一次性重建耗时单独统计。

每请求预留 64 个后续 token 并按 128 token 对齐。每步追加按逻辑位置写入。长度归零即完成本层读取路径的退休，复用该行时从逻辑位置 0 开始覆盖。无损编码的异常槽由预留容量约束，超出行容量或异常槽容量时返回溢出标记。需要容纳更多 token 时，上层须在继续写入前重建或搬迁并重新记录图和指针。当前原型测过追加、退休、复用及溢出诊断，尚无完整的服务分配器、自动增长和迁移调度。所报一次性重建从 BF16 源表示开始，未覆盖增长时从既有压缩缓存搬迁的支出。

`projection.py` 保留原有 BF16 GEMM、MRoPE 和 Hadamard，仅用同一 Triton kernel 完成 Q 的 RMSNorm 和当前 K 的 LayerNorm。冻结数据中 Q 和 gate 逐位一致，B16/32K 的当前 K 有 21/1024 个 BF16 元素不同，最大绝对差 0.00390625。其 Top-2048 在本矩阵全等，不能据此判定投影或跨步缓存逐位等价。将 MRoPE 和 Hadamard 也合入同一 kernel 的探索版出现更大非逐位结果，18 格均存在严格高分遗漏。

| 路径 | 最快原布局有效基线的几何平均加速 | Top-2048 全等格数 |
| --- | ---: | ---: |
| 独立无损 slab + 融合归一化 | 1.221× | 18/18 |
| 固定原 Index Key 布局，按已登记规则应用归一化融合与旧无损缓存 | 1.122× | 18/18 |
| 逐格取最快精确新路径，事后上限 | 1.221× | 18/18 |
| 独立无损 slab + 原生投影 | 1.054× | 18/18 |

带归一化融合路径的单格加速范围 1.090–1.325×。同样给予基线独立 Index Key 布局后，几何平均加速为 1.178×，保留原生投影时为 1.016×。稳态 Index Key 状态占输入 BF16 Key 字节的 78.7–79.1%，含对齐、容量预留、异常槽和逐请求起始偏移与容量。重建中位数的跨格范围约 0.341–2.159 ms。单格差异和选择器均见原始数据。

## INT4 全扫和候选内 BF16 精排

`int4_scan.py` 对每个 Key 用 32 B 存放 64 个有符号 INT4 值，再用 2 B 存放 BF16 缩放因子。Q 在每步每头量化到 INT8，解包后的 INT4 在 INT8 Tensor Core 上完整扫描。按低位分数筛选 4096 或 8192 个候选，随后用驻留的无损 BF16 数据按原公式重评分，返回逻辑 Top-2048。每步更新 INT4 和精确的无损表示。除精确表示外每个预留 Key 槽增加 34 B，缓存容量不能只按粗扫流量计算。

4096 档在 18 格几何平均加速为 1.254×，最差一格保留 98.926% 的原选集，0/18 格精确。128K/B16 连续槽 307.63 µs，原布局最快基线 476.84 µs，约 1.550×。8192 档为 1.166×，最差一格覆盖率 99.982%，9/18 格精确。两档均按有损路线报告。任务质量另由实际生成验证。

### 直接选择 INT4 分数

[int4-direct-policy.json](../int4-direct-policy.json) 固定使用相同的 18 格、INT4/INT8 量化规则和带归一化融合的投影。每步在计算当前 Q 的 INT8 码时写入当前 K 的 INT4 码和 BF16 缩放因子。完整历史扫描后直接从粗分选择逻辑 Top-2048。主策略用 FlashInfer 的 ragged Top-K，与在线任务测试使用同一选择器。预设消融改用 B4 的 vLLM persistent 和 B16 的 Ampere CuTe。2560 候选消融使用原生 BF16 公式精排。

直接路径持久状态为每预留槽 34 B，加每请求 8 B 的起始偏移和容量。BF16 所有历史 K 只在从已有缓存初始化时参与编码，稳态省去为二次精排准备的精确表示。Q 的 INT8 工作区、粗分和选择器工作区另计。每格校验从粗分返回的结果和原生 BF16 Top-2048，比较原有物理布局及同样允许独立布局的最快正确强基线。

| 路径 | 对原布局强基线 | 对同等独立布局强基线 | BF16 选集最低格覆盖率 | 对自身粗分选择正确 |
| --- | ---: | ---: | ---: | ---: |
| FlashInfer 直接选取，融合归一化 | 1.610× | 1.556× | 83.899% | 18/18 |
| 固定 vLLM 直接选取，融合归一化 | 1.723× | 1.665× | 83.899% | 18/18 |
| FlashInfer 直接选取，原生投影 | 1.338× | 1.293× | 83.899% | 18/18 |
| INT4 选 2560 候选后 BF16 精排 | 1.301× | 1.257× | 91.833% | 未测 |

直接选取三条路径的原生 BF16 选集最低单请求覆盖率均为 82.129%，平均格覆盖率 84.394%。相对最强原布局基线，FlashInfer 主路径 12/18 格达到 1.5×，固定 vLLM 为 17/18。128K/B16 连续槽的 FlashInfer 直选为 261.97 µs，最强原布局基线和同布局基线分别为 478.04 和 470.78 µs。单层 Index Key 驻留由 267.17 MB 降至 71.03 MB，粗分及 Q 量化工作区占 8.39 MB，选择器工作区另计。全矩阵 INT4 驻留的平均比例为 26.62%。已有 BF16 表示重排与 INT4 初始化两项中位数之和为 0.178–1.737 ms。已实测 CUDA Graph 重放能编码并写入新的当前 K，码和缩放因子与完整重新编码一致。

数据位于 `data/indexer-round2/results/int4-direct-frozen/full.{jsonl,csv,summary.json,metadata.json}`。

独立复验在 [int4-direct-holdout-policy.json](../int4-direct-holdout-policy.json) 固定前一矩阵更快的 vLLM 方案，B4 用 persistent，B16 用 Ampere CuTe。复验改用层 47、步 4、请求偏移 0。B4 的请求与前一矩阵分开，B16 沿用每个长度桶的全部 16 个捕获请求，改变评估层和步。原生投影加同一 vLLM 选择器作为对照。

| 留出复验 | 对原布局强基线 | 对同等独立布局强基线 | 同布局 ≥1.5× 的格数 |
| --- | ---: | ---: | ---: |
| 固定 vLLM 直接选取，融合归一化 | 1.718× | 1.662× | 18/18 |
| FlashInfer 直接选取，融合归一化 | 1.622× | 1.569× | 12/18 |
| 固定 vLLM 直接选取，原生投影 | 1.418× | 1.371× | 6/18 |
| FlashInfer 直接选取，原生投影 | 1.347× | 1.303× | 6/18 |

固定 vLLM 加融合归一化的最小格加速为 1.513× 和 1.507×。128K/B16 连续槽为 259.33 µs，对照为 478.72 和 475.56 µs。该留出数据的 BF16 Top-2048 格均覆盖率为 60.531%，最低格为 57.285%，最低单请求为 54.688%，0/18 格与 BF16 全等。四条路径均在 18 格通过自身粗分选集检查，并在 180 个请求布局组合上返回同一选集。粗分误差会随层和步变化，任务质量使用独立在线生成证据。驻留的 INT4 Key 状态平均占原 BF16 Key 字节的 26.619%。

复验数据位于 `data/indexer-round2/results/int4-direct-holdout/full.{jsonl,csv,summary.json,metadata.json}`。

## 周期刷新 Top-2048

[periodic-policy.json](../periodic-policy.json) 使用同样 18 个输入条件，按真实步 0–4 更新每个请求的序列长度和当前 K。每个条件在同一五步窗口重测原生和第一轮无损基线，取五步均通过严格校验的最快方法。稳定的物理映射根据步 4 的容量分配，超过步 0 的位置初始清零并由逐步投影覆盖。周期 3 在步 0/3 刷新，周期 4 在步 0/4 刷新。两者的计时都包含两次完整刷新和三次复用。额外报告第一个完整周期和末尾片段。复用步仍用原生 K-only 路径写入当前 K，保留上一刷新步的逻辑 ID。图内引用已存在的设备索引，原型独立部署时每请求每层需要 8 KiB 锚点 ID。

先导数据 `data/indexer-round2/results/periodic-frozen.jsonl` 中的当前 K 在两组 B16 的部分步上与原投影存在 21 和 10 个 BF16 元素差。128K/B16 步 0 的选集有 1 个严格漏项。最终刷新使用原生 BF16 投影和无损 slab。

| 方法 | 相对最快基线的五步几何平均加速 | 最慢格加速 | 仅复用步的总覆盖率 | 最差格复用覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 周期 3 | 2.201× | 1.907× | 42.605% | 39.355% |
| 周期 4 | 2.207× | 1.906× | 28.506% | 26.843% |

两法刷新步全部通过原生评分的严格选集检查。复用路径的 K-only 逐位等价。两法均有 0/18 格五步选集全等。周期 4 的末步是刷新，选集全等。周期 3 末步是复用，最差格覆盖 58.728%。128K/B16 连续槽五步最强基线为 2448.90 µs，周期 3 和 4 为 1004.03、978.43 µs。第一个完整周期为 528.73、513.28 µs，末尾片段为 494.85、460.97 µs。两段单独采样所得的中位数可与整窗不同。端到端任务质量由实际生成实验评估。

## 复现

在仓库根目录运行，输出名须为新路径。

```bash
scripts/indexer-round2/run_full.sh --int4 --output data/indexer-round2/results/full-rerun.jsonl
.venv/bin/python scripts/indexer-round2/summarize_full.py data/indexer-round2/results/full-rerun.jsonl
scripts/indexer-round2/run_full.sh --int4-direct --int4-ablate-2560 --output data/indexer-round2/results/int4-direct-rerun/full.jsonl
.venv/bin/python scripts/indexer-round2/summarize_direct.py data/indexer-round2/results/int4-direct-rerun/full.jsonl
scripts/indexer-round2/run_full.sh --int4-direct --int4-direct-policy scripts/indexer-round2/int4-direct-holdout-policy.json --layer 47 --step 4 --request-offset 0 --output data/indexer-round2/results/int4-direct-holdout-rerun/full.jsonl
.venv/bin/python scripts/indexer-round2/summarize_direct.py --policy scripts/indexer-round2/int4-direct-holdout-policy.json data/indexer-round2/results/int4-direct-holdout-rerun/full.jsonl
INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/verify_slab.py" scripts/indexer-round2/run_full.sh
INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/verify_direct.py" scripts/indexer-round2/run_full.sh
INDEXER_FULL_TARGET="$PWD/scripts/indexer-round2/periodic_full.py" scripts/indexer-round2/run_full.sh --output data/indexer-round2/results/periodic-rerun.jsonl
.venv/bin/python scripts/indexer-round2/summarize_periodic.py data/indexer-round2/results/periodic-rerun.jsonl
```

完整数据位于 `data/indexer-round2/results/full-frozen.{jsonl,csv,summary.json,metadata.json}` 和 `data/indexer-round2/results/periodic-native-frozen.{jsonl,csv,summary.json,metadata.json}`。旧脚本和上一轮数据保留。矩阵衡量单卡单层 Decode Indexer 的设备时间，端到端服务和任务质量使用独立实验。
