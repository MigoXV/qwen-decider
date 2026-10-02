# 上游代码

本项目使用 Apache-2.0 许可，LICENSE 保留完整条文。

- `models/config.py` 摘自 vLLM v0.17.1 的 `vllm/transformers_utils/configs/qwen3_5.py`，仅保留文本配置，增加 dense 模型标记。
- `models/model.py` 按 vLLM v0.14.1 `qwen3_next.py` 与 v0.17.1 `qwen3_5.py` 改编；复用已安装的算子，调整 QKVZ/BA 排列、dense 层构造、权重合并、文本 RoPE 与绑定输出层。只支持单卡，无额外 vLLM 副本。
- state-first 输入格式兼容 Mapika/decider-2b 与相邻训练项目，模型权重另有其来源说明。

上游：https://github.com/vllm-project/vllm/tree/v0.17.1
版权：Copyright contributors to the vLLM project；Copyright 2025 The Qwen Team and The HuggingFace Inc. team。
