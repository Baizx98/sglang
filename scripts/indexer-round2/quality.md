# Keye 候选复用的生成质量与实际服务时间

## 配置

模型为 Keye-VL-2.0-30B-A3B。单请求、TP1/PP2，PP0 是 RTX A6000，PP1 是 L40S。使用 BF16 原生 DSA、Top2048 与完整 GPU KV，关闭 CUDA Graph、Radix Cache 和 overlap schedule。启动、模型加载和预热在请求计时窗口之外。按形状触发的首次编译可能进入计时。原生对照使用生产 BF16 打分与精确 PyTorch 选集开关。服务计时来自同一 HTTP 进程，微基准中的最快单卡选择器延迟另行记录。

预先从 330 请求留出的 RULER 子集选择 8K 和 32K 的单值检索、多值检索及 QA 各一个，共 6 个请求、3 种任务。同一输入按固定随机序运行每种方法。使用温度 0，最多生成 64 token，保留 EOS 自然终止。在助手前缀统一追加 4 个 Keye tokenizer token，表示空的已闭合思考区。这样可以在短生成预算测任务答案。NIAH 按目标值命中占比，QA 按任一参考答案命中计分。计分区分完整文本与可见最终答案。8K 和 32K 的 QA 样本共享同一问题，在不同长度材料中出现。

## 路径

- **原生 exact**。每个 Decode 步和每层用完整历史算原生 BF16 Top2048。
- **step**。同层每两步刷新，复用步把前一刷新步的 Top2048 与当前 token 合并，以真实当前层的 BF16 评分在候选中精排。仍投影当前 Q、K 和 gate。
- **layer**。每一步偶数层刷新，奇数层复用同 PP stage 的前一层 Top2048，加入当前 token 并按本层真实 BF16 评分精排。若前层结果缺失则当前层走原生索引。
- **skip4**。同层每四步刷新，三个复用步只投影并写入真实当前 Index K，并将上次刷新产生的逻辑 ID 直接交给当前主 Attention。新 token 会参加下一次完整刷新。不同请求或非连续长度使锚点失效。
- **INT4**。每层保留完整原生 BF16 K。首个 Decode 步按真实页表构造请求内连续的逻辑 K 摘要。之后在线只追加当前真实 K 的 INT4 码。与同轮微基准复用 INT4 编码、INT8 Q、INT32 点积及同一版 FlashInfer 选择。取粗分 Top4096，对候选读真实 BF16 K 精排，再映射逻辑 ID。另有 Top8192 可选。每物理池 slot 额外使用 34 B 摘要。首次摘要构建和后续追加均在真实生成路径内，BF16 精排使用设备原生 paged KV。
- **INT4 直接选集**。沿用原生 Q、K 和 gate 投影，按同样的 INT4 粗分直接选 Top2048 并向主 Attention 返回逻辑 ID。服务仍写入每步真实 BF16 K，维护原生数组与 INT4 摘要，便于原生对照和当前状态的重算验证。

quality_run.py 记录第一次 SSE token 到达时刻、终止时刻和每次累计输出步数。等长比较采用各组共同的最多 32 个输出 token，按完整刷新周期缩短。刷新耗时和复用耗时按观测到的单 token 区间数加权。计时包含框架调度、PP 通信、detokenize 和本地 HTTP。逐步耗时有 CPU 和网络噪声，单次样本可以揭示量级，无法分解纯 Indexer GPU kernel 开销。

服务分配 139,968 个可用 token slot，Index K 池每层分配 140,032 行。INT4 侧车 34 B/行，对 48 层合计约 218 MiB，属于原生 BF16 Index K 之外的新增分配。近似部署还需要查询编码、选择输出和 workspace。该内存数字由实际池容量和代码中的张量尺寸计算。

显式 --audit 诊断在指定层、当前真实近似轨迹上用同一份 Q/K/gate 重算完整 BF16 索引，比较集合和严格高分遗漏。审计请求增加同步和全历史计算，延迟另存。旧 330 请求的 exact 主轨迹代理独立给出。

## 正式任务分数

首次不修改 prompt 的 8K 多值检索候选基线达到 384 token 上限，思考尚未闭合。四个答案出现在思考文本，最终答案为空。随后 HTTP 客户端的 cache flush 返回 400，因为流水线仍在清理刚输出结束的请求。保留第一次的完整生成和日志。继续运行前对该状态增加短暂等待重试，并验证断点继续的输入内容与运行协议。

原生、skip4 和 INT4 Top4096 在同一新服务进程完成 6 个任务的 18 次生成。另一组原生、step 和 layer 完成相同任务的 18 次生成，前 4 组与后 2 组之间为释放 GPU 暂停过服务。两次原生对照的完整输出 token 在 6 个任务均相同。

| RULER 任务 | 原生 | skip4 | INT4 Top4096 | step | layer |
|---|---:|---:|---:|---:|---:|
| 8K 多值检索 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8K 单值检索 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8K QA | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32K 多值检索 | 0.25 | 0.00 | 0.50 | 0.25 | 0.25 |
| 32K 单值检索 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32K QA | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |

skip4 在全部 6 个请求失分。其中 2 个在输出 3 或 4 token 后停止，达不到第一个完整的 4 步 Decode 周期，剩余 4 个请求的等步数 Decode 前缀中位加速为 1.153×，后续的定长运行单列。INT4 的任务分均不低于同样的原生请求，但只有 3/6 的完整输出 token 相同。32K 多值检索多命中一个值，该样本不能支撑总体质量变化。step 和 layer 各有 4/6 的输出 token 相同。step 的刷新和复用区间按区间数加权为 171.51 和 178.82 ms，step 和 layer 的等步数 Decode 前缀加速中位数分别为 0.952× 和 0.959×。

INT4 的独立实现对拍使用 L40S，8K 和 32K 合成输入、连续与 128-token 物理块打散、首次构建与一步追加共 8 项。服务逻辑 sidecar 与微基准 slab 的粗分逐位相同，BF16 精排集合相同。

## 定长生成延迟

同一服务进程对 8K、32K 的单值检索各执行 2 次。每次固定返回 17 个 token，含 Prefill 产生的首 token 与 16 步 Decode。12 次运行均完整观测四个周期。以下为两次重复的中位数，Decode 配对加速对每次重复分别计算后取中位数。该表的强制输出长度不用于任务计分。

| 长度 | 路径 | TTFT（s） | Decode（ms/token） | 总耗时（s） | Decode 配对加速 |
|---|---|---:|---:|---:|---:|
| 8K | 原生 | 2.965 | 163.89 | 5.588 | 1.000× |
| 8K | skip4 | 2.938 | 143.94 | 5.241 | 1.139× |
| 8K | INT4 Top4096 | 2.963 | 168.40 | 5.658 | 0.973× |
| 32K | 原生 | 22.311 | 158.09 | 24.841 | 1.000× |
| 32K | skip4 | 22.317 | 126.34 | 24.338 | 1.253× |
| 32K | INT4 Top4096 | 22.260 | 144.64 | 24.574 | 1.096× |

skip4 的 8K 刷新和复用步分别为 167.09、136.22 ms，32K 分别为 145.56、119.93 ms。INT4 Top4096 在 32K 两次配对一快一慢，原生 Decode 自身从 174.21 波动到 141.97 ms/token。该服务实验的收益和质量只适用于混卡 PP2 单请求及上述固定样本。skip4 的速度提升伴随自然任务全部失分。

## 原生投影与 INT4 直接选集

重新加载服务并用 FlashInfer 0.7.4 直接选 INT4 粗分 Top2048。6 个任务均重新与同进程原生配对，原生输出和上轮 6 个原生输出逐 token 一致。INT4 直接选集在 6 个任务得分依次为 1、1、1、0.25、1、1，都与原生相同，输出 token 完全一致的请求有 3/6。

下表为同一新进程每格固定 17 输出 token、两次重复的中位数。计时含首次逻辑 INT4 摘要构建、逐步原生投影、当前 BF16 K 原生写入及 INT4 追加。

| 长度 | 路径 | TTFT（s） | Decode（ms/token） | 总耗时（s） | Decode 配对加速 |
|---|---|---:|---:|---:|---:|
| 8K | 原生 | 2.949 | 163.45 | 5.564 | 1.000× |
| 8K | INT4 直接 | 2.948 | 144.78 | 5.265 | 1.138× |
| 32K | 原生 | 22.224 | 161.29 | 24.805 | 1.000× |
| 32K | INT4 直接 | 22.218 | 140.49 | 24.466 | 1.146× |

单次 Decode 配对比值为 8K 的 1.041×/1.235× 与 32K 的 1.109×/1.182×。在线短审计每长度各 40 个层步，8K 的 Recall 均值/最低为 80.58%/62.65%，32K 为 73.96%/44.73%，严格高分遗漏分别为 15,881 与 21,310。选集偏差明显，六任务评分相同只覆盖本次输入与解码协议。

## 在线选择审计

对 8K 单值检索与 32K 多值检索分别执行 9 个输出 token 的插桩运行，按层 1、12、25、36、47 的适用步重算实际当前隐状态下的原生分数。刷新步沿用原生选择，skip4 统计每周期三个复用步。表内严格高分遗漏在每行独立计重。

| 长度 | 路径 | 层步数 | Recall 均值 | 最低 Recall | 严格高分遗漏 |
|---|---|---:|---:|---:|---:|
| 8K | skip4 | 30 | 71.01% | 47.31% | 17,794 |
| 32K | skip4 | 30 | 63.21% | 33.01% | 22,581 |
| 8K | INT4 Top4096 | 40 | 96.98% | 88.62% | 2,467 |
| 32K | INT4 Top4096 | 40 | 94.71% | 79.20% | 4,328 |
| 32K | INT4 Top8192 | 40 | 99.12% | 94.29% | 716 |
| 8K | step | 20 | 78.41% | 45.51% | 8,830 |
| 8K | layer | 24 | 69.87% | 41.89% | 14,788 |

skip4 的 60 个抽样复用层步中，K-only 写入的当前 Index K 与完整投影的当前 K 全部逐位相等。对量化的两种候选预算，真实轨迹审计显示最差样本比层级微基准里已测条件低，任务上使用 Top8192 的证据范围仍为该单次短诊断。

## 前层 h 的离线代理

使用归档的原生 exact 轨迹。候选生成拿前层 h 与目标层完整历史 K 的估计分，当前新位置预留一个候选。直接取预测 Top2048 时，原生集合的留出均值为 RULER 85.38%、LongBench 89.98%、LongBench-v2 81.90%、MRCR 83.24%、GraphWalks 84.54%、BFCL 91.36%。RULER 的 M3072/M4096 真评分精排分别为 95.03%/97.17%。

在同一 6 个服务请求的归档轨迹上，每个请求读取 47 层、64 步的真实候选顺序和真实 BF16 候选分。保护前 1024 个候选时，由低排名尾部选出剩余 1024 个。相对完整 M3072/M4096 精排，各长度与预算的均值损失为 0.004–0.146 个百分点。保护前 1536 个候选时损失为 0.85–2.82 个百分点。32K 子集直接取预测 Top2048 的格内最低 Recall 约 0.05%。归档 Attention 误差只覆盖 M3072/M4096 完整精排及前层直接复用，保护前缀和预测 Top2048 目前只有选择代理。

同一 exact 轨迹的四步锚点离线复用仅检查非刷新步。集合 Recall 的均值在 8K 子集为 75.25%，在 32K 子集为 64.85%。原生索引选择当前最新位置的比例分别为 100% 和 99.985%，直接复用锚点的比例为零。新位置缺失与整体选集退化同时存在，这组代理无法定位二者各自的生成影响。

## 融合归一化与 vLLM 选择的服务接入条件

微基准的融合归一化闭包捕获固定 hidden state 和位置。服务需要每步接收实际 hidden state 和位置，以每个 PP rank 的可复用 workspace 保存规范化结果，再调用原生 MRoPE 与 Hadamard。文本位置为一维时按原生逻辑扩展为三个轴。实时写入需要同步维护当前 Index K 与 INT4 侧车。使用融合投影 K 写 BF16 缓存时，少数 BF16 位差会随运行留在历史中，需要按照近似路径进行跨步质量检测。保持原生 BF16 K 的变体还需计算原生 K 归一化，并计入服务时间。

vLLM Persistent 微基准适配器只允许一个设备可见。PP2 适配需为每个进程选择自身 scorer 所在 CUDA 设备，加载对应架构的 kernel，使用当前 CUDA stream、真实当前长度、rank 内 workspace，输出逻辑 int32 Top2048。实验服务当前采用原生投影和 FlashInfer，以上接入条件已检查，融合投影和 vLLM 的组合未部署测量。

## 复现

在实验仓库根目录运行。用空闲的目标双卡启动本地端口，在另一个终端运行请求脚本。启动后先完成对照再结束服务。

```bash
mkdir -p data/indexer-round2/quality/online-shadow
INDEXER_QUALITY_FI074=1 \
INDEXER_ROUND2_QUALITY_AUDIT="$PWD/data/indexer-round2/quality/online-shadow" \
bash scripts/indexer-round2/quality_start.sh

.venv/bin/python scripts/indexer-round2/quality_run.py \
  --output data/indexer-round2/quality/reproduction \
  --phase quality --closed-think --max-tokens 64 --modes exact skip4 int4
.venv/bin/python scripts/indexer-round2/quality_summarize.py data/indexer-round2/quality/reproduction

.venv/bin/python scripts/indexer-round2/quality_run.py \
  --output data/indexer-round2/quality/reproduction-fixed --phase performance \
  --closed-think --max-tokens 17 --task niah_single_1 --repeat 2 \
  --modes exact skip4 int4 --skip-warmups
.venv/bin/python scripts/indexer-round2/quality_summarize.py data/indexer-round2/quality/reproduction-fixed
```

原始 SSE 时间戳、输出 token 和分数位于对应的 outputs.jsonl，无原文的逐样本结果在 paired.csv，模式汇总在 summary.json。冻结输入位于 data/indexer-round2/quality/requests.jsonl，每次运行的 prompt 文件 SHA-256 在对应 config.json。
