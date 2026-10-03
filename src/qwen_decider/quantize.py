"""Offline W8A8: symmetric per-output-channel weights, dynamic per-token activations.

MLP and full-attention linear layers only; recurrent branch, embeddings and
normalization remain floating point. This conversion needs no calibration data.
"""

import json
import shutil
from pathlib import Path

from .convert import fingerprint


def quantize(source: Path, destination: Path):
    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_text":
        raise ValueError("expected native Qwen3.5 text weights")
    if not list(source.glob("*.safetensors")):
        raise ValueError("missing weights")
    if config.get("quantization_config"):
        raise ValueError("source already quantized")
    if destination.exists():
        raise ValueError("destination must not exist")
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    counts = {"int8_parameters": 0, "floating_parameters": 0}
    try:
        for p in source.iterdir():
            if (
                p.is_file()
                and p.suffix != ".safetensors"
                and p.name != "model.safetensors.index.json"
            ):
                shutil.copy2(p, staging / p.name)
        weight_map = {}
        total = 0
        for shard in sorted(source.glob("*.safetensors")):
            result = {}
            for name, tensor in load_file(str(shard)).items():
                if (
                    tensor.ndim == 2
                    and name.endswith(".weight")
                    and (".mlp." in name or ".self_attn." in name)
                ):
                    scale = tensor.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
                    result[name] = (tensor.float() / scale).round().clamp(-127, 127).to(torch.int8)
                    result[name.removesuffix(".weight") + ".weight_scale"] = scale
                    counts["int8_parameters"] += tensor.numel()
                else:
                    result[name] = tensor
                    counts["floating_parameters"] += tensor.numel()
            save_file(result, str(staging / shard.name), metadata={"format": "pt"})
            weight_map.update({k: shard.name for k in result})
            total += sum(t.numel() * t.element_size() for t in result.values())
        if len(set(weight_map.values())) > 1:
            (staging / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
            )
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "format": "int-quantized",
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
            "ignore": ["lm_head", "re:.*linear_attn.*"],
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2))
        (staging / "quantization.json").write_text(
            json.dumps(
                {
                    "source_fingerprint": fingerprint(source),
                    "method": "W8A8 per-channel/per-token, excludes recurrent branch",
                    **counts,
                },
                indent=2,
            )
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
