"""Validate complete native HF directories and actual tensor dtypes."""

import json
from collections import Counter
from pathlib import Path

from compressed_tensors.quantization import QuantizationConfig
from safetensors import safe_open
from transformers import AutoTokenizer

root = Path("model-bin")
reports = []
for precision in ["bf16", "fp16", "int8"]:
    path = root / f"decider-v3-{precision}"
    cfg = json.loads((path / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"]
    assert cfg["model_type"] == "qwen3_5_text"
    assert "auto_map" not in cfg
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    types = Counter()
    names = set()
    size = 0
    for shard in path.glob("*.safetensors"):
        size += shard.stat().st_size
        with safe_open(shard, framework="pt") as tensors:
            for key in tensors.keys():
                assert not key.startswith("lm.")
                assert key not in names
                names.add(key)
                types[tensors.get_slice(key).get_dtype()] += 1
    if precision == "int8":
        QuantizationConfig.model_validate(cfg["quantization_config"])
        assert types["I8"] > 0 and types["BF16"] > 0
        for name in names:
            if name.endswith(".weight_scale"):
                assert name.removesuffix(".weight_scale") + ".weight" in names
    else:
        assert set(types) == ({"BF16"} if precision == "bf16" else {"F16"})
    reports.append(
        {
            "precision": precision,
            "architecture": cfg["architectures"][0],
            "tensors": len(names),
            "tensor_dtypes": dict(types),
            "weight_bytes": size,
            "tokenizer_size": len(tokenizer),
        }
    )
Path("outputs/exports.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
