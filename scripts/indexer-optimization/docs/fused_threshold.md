# Keye 点积后直接压实候选

`fused_threshold.py` 在 BF16 K 缓存上运行。它均匀划分逻辑历史并从 2048 个区间各取一个位置，按 Keye 公式计算样本分数。样本次序统计量用于猜测能留下约 3072 个候选的阈值。全表评分时，CTA 在点积、BF16 舍入和有符号门权累加后立即比较阈值，将够分 token 的 FP32 分数和逻辑下标写入候选缓冲。基准缓冲容纳 4096 个候选，计数器仍记录超过容量的候选数。

完整扫描得到的真实候选数在 2048–4096 时，每个未收集位置的分数都低于阈值。对完整候选集合做 FP32 Top-K 即得到精确解，遇到边界同分时允许任意满足排名的选集。候选过少或超容量的行走常规密集评分及 FP32 Top-K。CUDA Graph 保留密集回退工作区并捕获回退 launch，证书通过时评分 CTA 早退，密集分数缓冲不写入。短行直接返回逻辑下标并用 -1 填充。

L40S 功能用例包含不等长、逐 token 打散的物理槽。普通输入保留 2928 个候选，省略密集回退。32769 个同分 token 强制触发溢出回退。将高分设置在样本中的精确位置，候选数 1024，触发不足回退。普通输入和两种回退均验证 eager 与 CUDA Graph 重放下的精确选集、位置唯一和短行填充。FlashInfer 0.6.8.post1 与 0.7.4 的三组结果一致。

```bash
CUDA_VISIBLE_DEVICES=GPU-79886749-8f57-63a5-8c8e-8a68a085565a \
  .venv/bin/python scripts/indexer-optimization/verify_fused_threshold.py
```

同一 benchmark 可用 `--module scripts/indexer-optimization/fused_threshold.py --select '^fused_threshold_exact_flashinfer$'` 与原生分数及其他 selector 对照。GPU 计时包含样本读取与评分、取阈值、全表评分和候选写入、回退路由、FP32 选择和输出合并。所有在线数据均从当前查询计算。

## A6000 结果

使用 RTX A6000、FlashInfer 0.7.4、层 36、step 2 的真实 128K 桶留出请求。计时从已生成的 BF16 Q、历史 K 和门权开始，到返回逻辑 Top-K 下标为止。各方法使用相同输入，图中连续执行 12 次，预热后交错测量 11 轮。两个 batch 的 20 个请求行均通过候选证书，候选数 2293–3626，三次复验均严格匹配原生 Triton 和 PyTorch 的排名。

| 128K 桶 | 评分与候选压实 / µs | 原生评分 + FlashInfer / µs | 同点最快原生组合 / µs |
| --- | ---: | ---: | ---: |
| B4 | 165.973 | 140.629 | 130.048（vLLM persistent） |
| B16 | 480.000 | 488.363 | 482.645（vLLM Ampere CuTe） |

B4 较同点最快原生组合慢 27.6%。B16 单点快 0.55%，两者观察范围重叠。本轮覆盖的两个点未建立稳定收益。候选证书通过时仍对完整历史执行 QK，省去的是密集 FP32 分数的写入及选择阶段对它的读取，同时新增样本、候选压实、路由和输出合并。原始计时、候选数和测试结果位于 `data/indexer-optimization/results/fused-real.jsonl` 第 1/2 行，对应 B4/B16。

A6000 合成验证位于 `data/indexer-optimization/results/fused-ties.jsonl` 第 1/2 行。32768 个分数完全相同的行全部超容量并走密集回退。边界同分用例保留 4096 项且均通过证书。两组各复验 10 次，每次 4 行均符合原生 Triton 和 PyTorch 的严格排名。边界同分时可返回不同的等价选集。

## CPU 原始数据核对

以下命令从离线捕获重建固定区间 Oracle 与已观测物理槽邻接 CSV，并对照 SHA-256。Oracle 按原 manifest 顺序读取独立请求和返回有效的步，统计包含原精确 Top-K 至少一项的固定长度区间。邻接统计只对已观测逻辑槽累计物理映射，重复访问需一致，未知邻接槽保持未知。

```bash
stats_dir=$(mktemp -d data/indexer-optimization/repro-cpu.XXXXXX)
.venv/bin/python scripts/indexer-optimization/reproduce_trace_statistics.py \
  --output "$stats_dir"
```
