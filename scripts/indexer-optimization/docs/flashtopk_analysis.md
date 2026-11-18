# FlashTopK 对 Keye Decode 的适用性

2026-11-18 核查。实验对象为 Keye-VL-2.0-30B-A3B 单卡单层 Decode。线上形状取自模型配置与部署代码，H=16、D=64、一个 Index Key 头、k=2048。每个逻辑位置先查页表，计算 BF16 QK 点积，在每头点积转回 BF16 后，用 FP32 计算缩放、ReLU、带符号门权和头归约。下游消费逻辑位置的集合。独立记录同分替换、严格高分遗漏和输出顺序。GLM-4.7-Flash 的本地配置属于 Glm4MoeLite，权重索引无原生 Lightning Indexer 参数，按作者决定只验证 Keye。[Keye 模型实现](https://huggingface.co/Kwai-Keye/Keye-VL-2.0-30B-A3B/blob/main/modeling_keye_vl_2.py) [GLM 配置](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json)

## 原实现的执行顺序

| 阶段 | 源码审计结果 | 对 Decode 的含义 |
|---|---|---|
| 定阈值 | 在 KV 上均匀取共享样本，使用同一打分算术建立每行分桶阈值。样本量有上下限，初始阈值通常瞄准约 2k 个候选。 | 小 batch 时采样、规划和恢复的多次启动都须计入。不同请求的 KV 历史通常彼此独立。 |
| 扫描 | SM90 的 CTA 有 384 个线程，每个查询 tile 有 8 行。两个 math warpgroups 交错计算 64-key 块，专用 warp 用 TMA 搬入 Q、K、权重和量化尺度。寄存器内归约、warp ballot 与原子预留产生候选分数和下标。 | 阈值可以减少写回与最终选择量。全扫描仍需读取所有有效 Key 并计算所有头。A6000 缺少此路径使用的 WGMMA/TMA。 |
| 自适应剪枝 | 在 32 头中保留 14 头，将遗漏正贡献的投影残差用范数界定，负贡献可以合并给出界。采样估计同一查询组中至少一行保留 Key 的概率，根据 GPU 成本阈值选择全扫或掩码加精算。 | 只有跳过完整打分的 Key 才节省此部分计算。构建界本身仍读取 Key，精算还要再读幸存者。直接搬用 14/32 的拆分与 Keye 的 16 头不匹配。 |
| 精选、恢复 | 候选缓冲中对 FP32 有序位做 radix 选择，边界同分取较小位置，再生成规范顺序。若候选少于所需或溢出，GPU 重算受影响行；大量坏行按调用恢复。 | 同卡比较需遵循同一种消费语义，并把任何溢出或短缺恢复计入计时。规范排序比只返回无序集合多做工作。 |

源码固定于 `090902abad86e5bb3324519124a680122fff91a1`。实现、测试和官方基准的入口分别见 [调度与缓冲区](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/flashtopk/topk.py)、[SM90 扫描](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/flashtopk/kernels/scan_sm90.py)、[上界规划](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/flashtopk/plan.py)、[异常行恢复](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/flashtopk/recovery.py)、[FP64 边界测试](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/tests/test_bounds.py) 与 [基准定义](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/reproduce/methods.py)。

## 同卡对照的选择

| 实现 | 对照资格与证据 |
|---|---|
| SGLang Keye SM80 | 本地部署首要对照。页表、带符号权重、BF16 中间舍入及 FP32 分数按部署实现给定。对同一打分器更换选择器，可定位选择阶段的收益。 |
| FlashInfer 与 vLLM | 发布的 FP32 ragged Top‑k、persistent Top‑k 都是值得实测的下游选择器。按同一输入逐点取实测最短组合。记录库版本、实际调度、输出合同和工作区。 |
| FlashTopK | 公开 fused 路径要求 SM90/SM100、32×128 和连续窗口。用户的 A6000 为 SM86。原版留作算法与伪代码层面的参照。[支持矩阵](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/docs/hardware.md) |
| DeepSelect | `bfa4507d` 的默认 CUDA 构建目标为 SM100a/103a，FP32 选择实现的公共加载路径调用 SM90 TMA。代码树中有 SM80 通用辅助头文件，不能据此判断该算子可用于 A6000。[构建配置](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/setup.py) [设备端加载](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/csrc/cuda_kernels/common_parts.cuh) |
| LiteTopK / HPC-Ops | LiteTopK 指定的框架集成面向 SM100 的量化 Prefill。HPC-Ops 公布的构建目标自 SM90 起。两者可用于比较阈值设计。[LiteTopK 实现说明](https://github.com/Heisenberg-Yin/LiteTopK/blob/75d0c385d587f41a6659e2651b23af23b5ebe4f1/README.md) [HPC-Ops 要求](https://github.com/Tencent/hpc-ops/blob/96be0af753c4ff0f6e62f5129f45293748c5dee8/README.md) |

## Decode 的访存账

满长独立请求每层每步有 B×L 个 FP32 分数，物化后各写读一次约 8BL 字节。每个 BF16 Key 占 128 字节，单次全扫的有效 Key 载荷为 128BL 字节。即使完全消除分数写读，节省的这部分逻辑字节也只有 Key 载荷的 6.25%。该估算假设一次分数读取，实际选择可能多遍扫描，页表、缓存、写分配与访存事务另计。B=16、L=131072 时，Key 载荷为 256 MiB，分数张量为 8 MiB。这里的数值由张量形状推算。小 batch 是否受选择阶段并行度限制由实测分解判断。

Prefill 中同一段 K 可以服务多个查询行，每行具有不同的因果终点，整段分数数量随长度二次增长。普通自回归 Decode 每个请求每步只有一个查询；不同请求的 K 只能在确有共享前缀时复用。公开 Prefill 的加速比不能外推到当前每步一次查询的工作负载。Keye 的模型输入可以由离线 Prefill 准备，GPU 性能计时从本步已经得到的 Q、Key Cache、head weights 和页表开始。[HyScale](https://arxiv.org/abs/2610.32057) [DoSA](https://arxiv.org/abs/2610.13990) [LiteTopK](https://arxiv.org/abs/2607.11976)
