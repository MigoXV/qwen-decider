# 推理验证与性能记录

测试日期：2026-10-02。默认使用 **BF16**，这是用户结合已有精度测试作出的选择。以下保留原始数值门槛和测量结果：超过概率误差门槛不等于模型完全不可用。INT8 停止后续测试，保留导出和已有结果。

## 环境与范围

推理环境固定为 Python 3.10.18、PyTorch 2.9.1、vLLM 0.14.1、Transformers 4.57.6。初始环境中的包没有升级、降级或删除，仅新增本项目；见 [环境核对](results/environment-check.json)。没有修改 site-packages。

三种候选均使用本机独占 GPU 0：NVIDIA A100-PCIE-40GB，驱动 595.71.05。单卡、输入上限 2048 token、最多 32 个引擎序列、HTTP 容量 16、GPU 内存比例 0.45、前缀缓存关闭。vLLM 编译和 FULL_AND_PIECEWISE CUDA Graph 已启用。各模型分别启动、完成分长度预热、测试、退出；没有多个候选争用同一 GPU。

性能测量为客户端 HTTP 决策请求的完整响应耗时，包含服务调度和传输，不是纯 CUDA 内核耗时，也不是会议处理端到端耗时。使用 8 条固定合成样本，每模式三轮；每请求 1／8 题，并发 1／2／4／8。八题模式重复同一道题并使用不同 ID，表示调度压力，不代表八道独立业务题。短测试不能替代长时间稳定性验证。

## 三份标准 Hugging Face 权重

均位于本项目 `model-bin/` 下，包含原生 `Qwen3_5ForCausalLM` 配置、根目录 tokenizer 和 safetensors，没有训练 wrapper 或外部 adapter 依赖。

| 目录 | 权重文件字节数 | 保存形式 |
|---|---:|---|
| decider-v3-bf16 | 3,763,692,048 | 320 个 BF16 张量 |
| decider-v3-fp16 | 3,763,691,728 | 320 个 FP16 张量 |
| decider-v3-int8 | 2,771,202,040 | 96 个 INT8 权重、96 个 FP32 scale、224 个 BF16 张量 |

INT8 是 compressed-tensors W8A8：MLP、全注意力线性层量化；循环分支、embedding、归一化保留 BF16。实际 vLLM 加载日志确认启用 CutlassScaledMMLinearKernel。模型加载占用约 BF16／FP16 3.54 GiB、INT8 2.61 GiB，区别于包括缓存预分配在内的整个进程显存。

BF16、FP16 在**原有训练环境**的 Transformers 5.18 中通过原生 AutoModelForCausalLM 加载，没有缺失、多余或形状不匹配的权重；未修改推理环境版本。INT8 通过标准 compressed-tensors 配置验证和实际 vLLM 加载；未宣称在缺少对应 compressed-tensors 依赖的训练环境完成 Transformers INT8 加载。详见 [导出检查](results/exports.json)、[原生 HF 加载](results/hf-reload.json)。

## 数值核对

参考为原始训练实现的 FP32 输出，两份合成 holdout 共 160 条。参考答案未进入模型输入。预设门槛为选择一致率 ≥99%、最大概率绝对误差 ≤0.02。

| 精度 | 选择一致率 | 最大概率绝对误差 | 原定数值门槛 |
|---|---:|---:|---|
| BF16 | 160/160，100% | 0.140552 | 概率误差超限；用户选择先使用 |
| FP16 | 160/160，100% | 0.007513 | 通过 |
| INT8 | 156/160，97.5% | 0.539135 | 未通过；停止后续测试 |

BF16 并发 8 核对同样 100% 选择一致。另以同精度原生 PyTorch BF16 作参考，vLLM 选择仍全部一致，但最大概率差为 0.110919，因此不能把全部差异归因于 FP32 到 BF16 转换。尚未逐层定位概率差异；概率没有校准，`calibrated=false`。

## 已完成的速度测试

下表为三轮各自 P50／P95／吞吐的中位数，吞吐单位为题/秒。每轮仅 8 个请求，P95 与 P99 均落在最大样本附近，不能作为稳定尾延迟估计。所有测量请求错误数为 0。

| 精度 | 每请求题数 | 并发 | P50 ms | P95 ms | 题/秒 |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 1 | 45.73 | 55.13 | 20.85 |
| FP16 | 1 | 1 | 43.24 | 45.57 | 23.15 |
| INT8 | 1 | 1 | 44.40 | 45.99 | 22.39 |
| BF16 | 1 | 8 | 158.47 | 163.59 | 47.32 |
| FP16 | 1 | 8 | 126.06 | 130.99 | 58.12 |
| INT8 | 1 | 8 | 127.23 | 133.15 | 57.28 |
| BF16 | 8 | 1 | 123.53 | 193.35 | 55.12 |
| FP16 | 8 | 1 | 117.57 | 155.74 | 63.02 |
| INT8 | 8 | 1 | 113.56 | 134.77 | 68.47 |
| BF16 | 8 | 8 | 414.74 | 619.46 | 102.29 |
| FP16 | 8 | 8 | 303.54 | 512.52 | 124.11 |
| INT8 | 8 | 8 | 286.25 | 430.79 | 147.63 |

远端 4B spark25 服务单题单并发 P50 为 200.16 ms，对应本机 BF16 45.73 ms，当前部署约快 4.38 倍。远端就绪接口确认 vLLM 0.14.1、INT8 compressed-tensors、BF16 激活、编译启用；其硬件未独立核验，网络、提示词组织、输出方式、上下文配置均不同。这是现有两种部署完成同一决策任务的对照，不能解释为相同硬件上的纯模型架构加速比。远端通过 Responses 生成选项 JSON，本项目只执行一个受限答案位置。

## 工程检查与限制

- 14 项单元／行为测试通过，覆盖提示词、选项解析、边界、超时、背压、导出、投影排列和不完整权重拒绝。
- 真实 HTTP 验证覆盖 2–10 选项、多问题隔离、概率和、超长输入 413、客户端取消后恢复，见 [接口检查](results/live-contract.json)。
- 强制终止 INT8 Engine Core 后，待处理请求返回 503；ready 返回 503、live 保持 200，关闭后无残留子进程且 GPU 0 显存归零，见 [故障检查](results/fault-core.json)。该测试在停止 INT8 后续测试的决定之前完成。
- 正常 SIGTERM 关闭已核对 GPU 释放。API 父进程 SIGKILL 回收机制已实现，但未完成该场景的实测。
- 没有会议端到端测速、长时间稳态验收、GPU 内核分段 profile 或生产数据评估。未改动 MeetNote 的模型配置。

## 复现与原始记录

按 README 的独立模型路径启动服务，然后执行：

```bash
poetry run python scripts/check_exports.py
poetry run python -m unittest discover -s tests -v
poetry run python scripts/validate.py --url http://127.0.0.1:10003 --output outputs/parity-bf16.json
poetry run python scripts/benchmark.py --url http://127.0.0.1:10003 --output outputs/benchmark-bf16.json
```

`validate.py` 保留原始门槛，所以 BF16 会返回非零并保存实际结果；这不修改用户选择的默认精度。无需继续运行 INT8。

[完整矩阵汇总](results/summary.json)；逐请求原始记录已压缩归档为 `results/benchmark-{bf16,fp16,int8}.json.gz`，数值记录为 `results/parity-{bf16,fp16,int8}.json.gz`，可用 Python gzip 或 `gzip -dc` 读取。每份记录保留服务指纹、版本、运行设置、每轮耗时和失败情况。BF16／FP16 测量发生在最后一批加载检查加固之前，INT8 包含该修复；这些改动未修改浮点 forward 数学。项目采用分批 Git 提交，未将权重提交 Git。
