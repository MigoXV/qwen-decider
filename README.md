# qwen-decider

基于 **固定 vLLM 0.14.1** 的 Qwen3.5 有限选项决策推理服务。输入上下文、问题和选项，返回所选选项与条件概率；兼容 MeetNote 的 `/v1/decisions`。业务 Prompt 由调用方提供。

项目内部注册 Qwen3.5 配置和模型，复用原有 vLLM 调度、FlashAttention、Gated DeltaNet、编译和 CUDA Graph。不会升级依赖或改写 site-packages。首版为单卡、纯文本、2–10 选项，不开启前缀缓存。

## 环境和安装

使用已建立的 Poetry Python 3.10 环境。PyTorch 2.9.1、vLLM 0.14.1、Transformers 4.57.6 固定；`requirements-runtime.txt` 保存初始完整环境。不要直接执行可能替换依赖的升级、sync 或无约束安装。

```bash
# 仅安装本项目及插件入口，不解析或更改运行依赖。
poetry run pip install --no-deps -e .
poetry run python -m unittest discover -s tests -v
```

## 标准 Hugging Face 权重导出

导出完整原生 `Qwen3_5ForCausalLM` 仓库，包括 config、根目录 tokenizer 和 safetensors；没有训练包装层、外部 adapter 依赖或远程代码。原始源权重不覆盖，输出目录必须不存在。

```bash
poetry run qwen-decider convert /path/to/decider-domain-v3-hf model-bin/decider-v3-bf16 --precision bf16
poetry run qwen-decider convert /path/to/decider-domain-v3-hf model-bin/decider-v3-fp16 --precision fp16
poetry run qwen-decider quantize-int8 model-bin/decider-v3-bf16 model-bin/decider-v3-int8
poetry run python scripts/check_exports.py
```

BF16、FP16 的实际浮点权重分别保存为对应 dtype。INT8 使用标准 compressed-tensors W8A8：MLP 和 full-attention 线性层采用逐输出通道对称 INT8 权重、逐 token 动态 INT8 激活；Gated DeltaNet、embedding 与归一化保留 BF16，scale 使用 FP32。当前约 9.94 亿参数量化、8.88 亿参数保留浮点；不是所有张量都为 INT8。无需校准数据，此转换可能改变判断与置信度，必须查看实测报告。

旧 Transformers 4.57.6 本身不认识 Qwen3.5，项目插件负责本环境的 vLLM 加载；标准 HF 仓库可供原生支持 Qwen3.5 的 Transformers 使用。INT8 还需相应 compressed-tensors 支持。服务不下载模型。

## 启动

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 \
poetry run qwen-decider serve --model-path model-bin/decider-v3-bf16 --precision bf16

# 其余两种启动方式使用对应独立权重目录：
poetry run qwen-decider serve --model-path model-bin/decider-v3-fp16 --precision fp16
poetry run qwen-decider serve --model-path model-bin/decider-v3-int8 --precision int8
```

默认精度为 BF16，由用户结合精度测试选择使用。现有 160 条参考样本决策全部一致，但概率误差超过原定 0.02 门槛；保留这一测量结论，不将其视为不可用的结论。INT8 已停止后续测试，现有权重和结果保留，详见验证报告。默认地址 `127.0.0.1:10003`，最多 16 个并行 HTTP 请求、1 个引擎序列，每请求最多 8 题，每题完整输入不超过 2048 token，body 上限 256 KiB，业务期限 30 秒。CLI 参数支持 `DECIDER_*` 环境变量，见 `.env.example`。CLI 不自动读取 `.env`；VS Code 配置会读取。

启动先检查版本、加载模型、编译和按不同长度预热，再进入 ready；首次编译可能耗时数分钟。`--enforce-eager` 仅用于 vLLM 正确性诊断，不作为已加速性能结果。GPU 由 `CUDA_VISIBLE_DEVICES` 指定，服务不改写该变量。

## 接口

```json
{
  "state": "主持人说：建议小李负责验收，分工尚未确认。",
  "questions": {
    "owner": {
      "type": "choice",
      "instructions": "原文是否支持小李已被指定为验收负责人？",
      "criteria": {"supported": "支持", "unsupported": "不支持", "uncertain": "无法判断"}
    }
  }
}
```

发送到 `POST /v1/decisions`。`answers` 按问题 ID 返回 `type`、`choice`、`probabilities`、`confidence`；附带模型 fingerprint、精度、后端版本和 usage。模型指纹包含权重、配置、运行代码和依赖版本。

保持训练用 state-first 分段编码，保留字典与选项顺序。一次 vLLM 请求只执行一个答案位置，限制合法字母并取得所有候选分数。输出不是自由生成的理由或 JSON；JSON 由程序组装。业务 `output_tokens=0` 表示没有生成业务文本，`engine_output_tokens` 单独记录实际答案位置数量。概率是合法选项间归一化的原始分数，`calibrated=false`，不得理解为已校准正确率。

- `GET /v1/info`：模型、引擎、精度、编译模式和容量。
- `GET /health/live`、`GET /health/ready`：存活与引擎就绪。
- `GET /metrics`：请求结果、HTTP inflight 和耗时指标。
- 413：body 或 token 超限；422：输入不合法；503：过载或引擎故障；504：deadline。

整份请求预校验后才提交模型。取消、断连、失败和关闭会 abort 相关引擎请求；请求不持久化，不记录原文日志。主进程关闭会回收 vLLM 子进程，Linux 子进程还绑定父进程死亡信号。

## 验证与性能

```bash
poetry run python scripts/validate.py --url http://127.0.0.1:10003 --output outputs/parity-bf16.json
poetry run python scripts/benchmark.py --url http://127.0.0.1:10003 --output outputs/benchmark-bf16.json
```

脚本默认读取本机相邻训练工程的合成测试集，可通过参数换路径。参考标签只用于离线核对，不进入模型输入。速度对照使用现有 spark25 Responses 服务，支持 `--qwen-only`；只测决策任务，不跑会议端到端。

每个模式包含每请求 1／8 题、并发 1／2／4／8、三轮。8 题模式重复同一题作为共享上下文的调度压力测试，不代表八道独立业务题。原始记录在 `outputs/`，性能摘要和限制见 `docs/VALIDATION.md`。远端 4B 的硬件、量化和网络差异必须随结果披露，不把部署差异归因于参数量。

导出数据和权重不提交 Git；不含输入原文的压缩测试记录保存在 `docs/results/`。代码分阶段提交在 `feature/qwen-decider`。

## 8 GB 显存约束

默认保持 BF16、vLLM 编译和 CUDA Graph，不以 eager 代替加速。单实例上限按十进制 8,000,000,000 字节配置：引擎最多同时运行 1 个序列，每轮最多 2048 token，KV 缓存固定为 128 MiB，不再按显卡容量扩大缓存。`gpu_memory_utilization` 默认 0.12；固定 KV 缓存启用时，该比例不决定缓存容量。

本地 Worker 子类通过 PyTorch 分配器限制最多 5 GB，预留 3 GB 给 CUDA context 和非 PyTorch 开销；加载、预热以及每次执行前后检查 NVML 实际进程显存，超过 8 GB 就报错，不发布成功结果。此检查是进程分配限额与运行时检查，不是驱动层显存分区；采样不能证明任意瞬时峰值。`/v1/info` 返回预算、缓存大小、实际使用量及已观测峰值。输入上限仍为 2048 token，超长输入拒绝，不截断；HTTP 容量仍为 16，GPU 同时执行的序列最多为 1。

不修改 PyTorch/vLLM 安装包或版本。修改显存配置后应重新验证最长输入、并发请求、精度与速度。
