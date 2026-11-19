# Keye Decode 的无损压缩 Indexer K

Keye Indexer 每个物理 token 存放一个 64 维 BF16 K 向量。扫描显式页表后，Triton 内核把每个 K tile 还原到寄存器，再进入原有方向的 BF16 张量核乘法。QK 的 BF16 舍入、FP32 ReLU 与带符号门权累加沿用原评分公式。逻辑下标仍由页表和序列长度定义。显式解包消融在相同得分 tile 和选择器前恢复整个 BF16 缓冲。

## 布局与异常行

| 数组 | 每物理槽字节 | 用途 |
| --- | ---: | --- |
| `mantissa_sign` | 64 | 每个元素的符号位和 7 位尾数 |
| `exponent_pairs` | 32 | 每个元素 4 位，每字节两个编码 |
| `headers` | 4 | 低 8 位存底数，位 8–30 存异常行预留编号，最高位标记当前编码 |
| `exceptions` | 每预留行 128 | 仅对指数跨度超过 15 个非零档的槽保留原 BF16 行，包含未来 append 的预留空间 |

指数编码 0 对应 BF16 指数字段 0，保留原尾数与符号，覆盖正负零和次正规值。其余 15 种编码相对于这一行的最小非零指数。所有指数和尾数位在评分前恢复，异常位置从独立 BF16 行读入。缓存数据实际分配为 `100 × physical_slots + 128 × exception_capacity` 字节。公开的 `EncodedKeys.storage_bytes` 统计此值，Arena 另有 4 B 异常计数器，append 测试另有每请求 4 B 状态。

## 增量更新

`PackedKeyArena.from_cache` 可从已有物理缓存转换并预留物理槽。 `PackedKeyArena.write` 接收当前步 BF16 K 和物理写入位置。缺少预留容量时，先增大异常缓冲并复制已保留的异常行，后续在同一 CUDA 流继续写入。某个物理槽第一次写入异常行时，从 arena 分配唯一编号。该槽被退场请求释放、用常规值覆盖，再写入异常行时复用原编号。其他在用物理槽及其异常行保持有效。正在同一次写入的活动请求须有不同物理目标槽。用于通用接口的 `-1` 代表未启用的行。

CUDA Graph 保持固定地址。捕获前按图覆盖的不同写入行数预留容量，随后使用 `write_reserved` 或 `prepare_packed_full`。分配耗尽时，内核保留旧槽并返回逐行状态。调用方检查状态，并在重新预留后重试失败写入。超出预留覆盖期时应在图外扩大缓冲并重新捕获。正常运行可为后续一段 Decode horizon 一次预留最坏情况的异常行，固定输入计时只需预留一个批次。 `prepare_packed_full` 从最新页表和长度定位当前尾位置，直接将投影产出的 BF16 K 写入压缩缓存，评分直接使用本步新投影的 Q 和门权。参考路径为测量对照保留原始 BF16 K 缓冲。

## 复现

在仓库根目录使用相应 GPU UUID。验证脚本检查所有 BF16 位模式、异常缓存增长、多个 Decode 步、请求退场和重排、物理槽复用、预留耗尽重试、逐位评分以及不同长度与页表下的 CUDA Graph 重放。

```bash
CUDA_VISIBLE_DEVICES=GPU-79886749-8f57-63a5-8c8e-8a68a085565a \
  .venv/bin/python scripts/indexer-optimization/verify_packed_cache.py
```

消融模块交给同一个 benchmark，三条路径使用相同 QK tile 和 FlashInfer FP32 选择器。显式解包缓冲预分配，在线计时包含解包和 BF16 中间写回。

```bash
CUDA_VISIBLE_DEVICES=GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822 \
  .venv/bin/python scripts/indexer-optimization/bench.py \
  --source real --batch 4 16 --length 131072 --layer 24 --step 2 \
  --module scripts/indexer-optimization/packed_ablation.py \
  --select '^ablate_' --output data/indexer-optimization/results/packed-ablation.jsonl
```

已有 BF16 历史的转换 `P0` 单独用预热后同步墙钟统计，包含动态异常压实。已转换历史的 Decode 稳态收益应使用相同真实请求、同一 scorer tile 和同一 FP32 Top-K，分别测量投影、写入与完整 Indexer。如 `baseline_full_us - packed_full_us` 为正，转换盈亏平衡步数为 `ceil(P0_us / per_step_saved_us)`。该换算只覆盖指定形状的重复 Decode Indexer，不外推模型端到端速度。

## 持久缓冲验证与 A6000 留出数据

`results/heldout-new-append.jsonl` 的 4 行覆盖 B4/B16、层 24/47，共 16 个真实递增步。每步选集通过原生 Triton 和 PyTorch 两套评分核验，最终物理缓存逐位一致。此批真实增量没有产生新的异常行。强制连续异常、回收重用与异常缓冲耗尽重试由上方独立验证脚本覆盖。

`results/heldout-new-full128.jsonl` 共 8 行，对应层 36/47、step 2/4 和 B4/B16，FlashInfer 0.7.4，真实 128K 桶。完整区域包含相同投影、当前 K 写入、全历史评分和 FP32 Top-K。B4/B16 的压缩与 BF16 的实际 Indexer K 分配比约 78.6%，直接写入阶段中位数 2.13/2.39 µs。转换既有 BF16 缓冲的同步墙钟中位数分别约 423/960 µs。如果每点使用最快的有效基线及最快的有效压缩组合，128K 的单点转换盈亏平衡为 23–37/13–14 个 Decode 步。图内方法选择应引用冻结配置的主报告。后续分母只应涵盖相同 Selector 接口和相同物理布局，快速比较不同 Selector 时需标出选择器。

## 同评分器消融

A6000 上使用 FlashInfer 0.7.4、层 36、step 2、留出请求和连续物理槽。三条路径使用相同 128-token 评分 tile 和 FP32 选择器，计时从已有 BF16 Q、K 缓存及门权开始，到返回逻辑 Top-K 下标为止。压缩缓冲在计时前构建。显式解包的 BF16 输出缓冲也提前分配，计时包含完整解码写回和随后的 K 读取。所有路径在两个点均通过 Triton 和 PyTorch 的严格排名验证。

| 128K 桶 | 直接读取 BF16 / µs | 融合读压缩 K / µs | 显式解压后评分 / µs |
| --- | ---: | ---: | ---: |
| B4 | 141.653 | 125.355 | 316.672 |
| B16 | 491.776 | 422.315 | 1189.888 |

融合路径比相同评分器的 BF16 路径低 11.5%/14.1%。显式恢复整块 BF16 的路径为融合路径的 2.53×/2.82×。该消融覆盖当前实现的解包与中间写回的联合成本。原始记录位于 `data/indexer-optimization/results/ablation-packed.jsonl` 第 1/2 行，对应 B4/B16。
