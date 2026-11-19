# Keye Decode Indexer 微实验

本目录在独立实验入口运行 Keye-VL-2.0-30B-A3B 的 Indexer，使用模型原有 16 个头、64 维 BF16、Top‑2048。主要实现按 token 无损打包历史 Index Key，在分页评分内核中直接恢复，然后调用现有 FP32 选择器。真实请求来自旧的全模型采集；加速数字来自 RTX A6000 单卡层回放。

| 入口 | 内容 |
|---|---|
| [FlashTopK 源码分析](docs/flashtopk_analysis.md) | SM90/100 原作的样本、筛选、恢复与硬件限制 |
| [假设和相关方法](docs/ideas.md) | 压缩、段摘要、数据布局及相关工作的区别 |
| [部署基线与工作负载](docs/baseline_results.md) | 合成分解、真实条件、物理布局反例 |
| [无损缓存实现与测试](docs/packed_cache.md) | 编码、异常行、增量写、同一 scorer 消融 |
| [段级筛选实验](docs/segments.md) | 粗筛、融合、覆盖与漏选 |
| [实验记录](docs/experiment_log.md) | 候选路线、开发数据、留出复验与结论 |

## 在实验服务器运行

从 `keye_sglang` 仓库根目录运行。使用 `.venv/bin/python`、`/Tan/model/Keye-VL-2.0-30B-A3B` 和 `data/horizonkv-motivation/2026-10-21_bs16-v01/fixtures`。实验产物保存在 `data/indexer-optimization/results/`，参考项目在 `data/indexer-optimization/references/`，部署代码未切换到实验实现。下例可复测一个真实点，使用新的输出文件名。

```bash
export CUDA_VISIBLE_DEVICES=GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822
export CUDA_HOME="$PWD/data/indexer-optimization/cuda130/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PWD/.venv/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib"
export CPATH="$CUDA_HOME/include/cccl:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/include"
export FLASHINFER_CUDA_ARCH_LIST=8.6

.venv/bin/python scripts/indexer-optimization/bench.py \
  --source real --batch 4 16 --length 131072 --layer 36 --step 2 \
  --request-offset 8 --module scripts/indexer-optimization/optimize.py \
  --module scripts/indexer-optimization/vllm_baseline.py \
  --select '^opt_qk_n64w4_vllm$' --align-width --full --direct-packed --packed-audit \
  --output data/indexer-optimization/results/reproduction-layer36.jsonl
```

部署版 FlashInfer 为 0.6.8.post1。实验目录下另有 0.7.4 与 CuTe DSL 4.8.0，置于单独解释器进程的搜索路径中；当前安装的 SGLang 使用原虚拟环境。在同一条命令前切换到该版本，并加 `--ampere-cute`，可增加 vLLM 在 Ampere 上的专用选择器。最新版进程的启动与全部参数记录在[可复现驱动](run_suite.sh) 以及每个 JSONL 旁的 `.metadata.json`。上游 vLLM 对照使用 `df763962fc61f8d0047e89aa39079efe0e70e678`；C++ 适配器读取该源码，保持设备端选择以及启动调度，CuTe 入口直接加载该提交的源文件。环境使用 CUDA Toolkit 13.0.88、CCCL 13.0.85、Torch 2.9.1+cu130、Triton 3.5.1。目标 UUID、功耗限制（200 W）、TP1 和布局都应随结果一起使用。

`--full` 计入当前 hidden state 的投影、归一化、位置变换、当前 Index Key 写入、打分和返回逻辑位置。`--direct-packed` 在压缩表示上直接写当前行。`--packed-audit` 测初次转换的同步墙钟和表示占用。`--append-steps` 在同一缓存上连续增长，并对照每步的参考选择。`--rotate` 轮换不同请求以检查 L2 影响。`--shuffle-block` 指定物理块大小并重新排布，保持逻辑分数。完整变量与其他选项见 [bench.py](bench.py)。

原始 JSONL 包含试次延迟、同分边界、逐行正确性、集合及顺序稳定性。复现表中汇总使用 `summarize_holdout.py`，应检查各组 JSONL 和退出状态后再生成。`data/` 中的原始权重输入、完整 trace、JIT 二进制与库均留在实验服务器。
