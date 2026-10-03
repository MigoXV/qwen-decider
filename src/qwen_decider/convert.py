"""Lossless conversion of the local training wrapper; no training imports."""

import hashlib
import json
import shutil
from pathlib import Path


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path, precision="bf16"):
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    from safetensors.torch import load_file, save_file

    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "meetnote_qwen_decision":
        raise ValueError("expected a complete meetnote_qwen_decision wrapper")
    native = cfg["backbone_config"]
    if native["model_type"] != "qwen3_5_text" or native.get("num_experts", 0):
        raise ValueError("only dense Qwen3.5 text is supported")
    if destination.exists():
        raise ValueError("destination must not exist")
    if not (source / "tokenizer/tokenizer.json").is_file():
        raise ValueError("missing tokenizer")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError("missing safetensors")
    native = dict(
        native,
        architectures=["Qwen3_5ForCausalLM"],
        dtype=str(dtype).removeprefix("torch."),
        torch_dtype=str(dtype).removeprefix("torch."),
    )
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        weight_map, size = {}, 0
        for shard in shards:
            data = load_file(str(shard))
            if any(not k.startswith("lm.") for k in data):
                raise ValueError("unexpected wrapper keys")
            data = {k[3:]: v.to(dtype) if v.is_floating_point() else v for k, v in data.items()}
            if set(data) & set(weight_map):
                raise ValueError("duplicate tensor keys")
            save_file(data, str(staging / shard.name), metadata={"format": "pt"})
            weight_map.update({k: shard.name for k in data})
            size += sum(t.numel() * t.element_size() for t in data.values())
        (staging / "config.json").write_text(json.dumps(native, indent=2))
        if len(shards) > 1:
            (staging / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map})
            )
        for p in (source / "tokenizer").iterdir():
            if p.is_file():
                shutil.copy2(p, staging / p.name)
        (staging / "decider_config.json").write_text(
            json.dumps(
                {
                    "max_length": cfg["max_length"],
                    "letter_token_ids": cfg["letter_token_ids"],
                    "temperature": 1.0,
                    "calibrated": False,
                }
            )
        )
        (staging / "source.json").write_text(
            json.dumps(
                {
                    "source": str(source.resolve()),
                    "fingerprint": fingerprint(source),
                    "conversion": "lm. prefix removal; explicit floating point cast",
                    "precision": precision,
                },
                indent=2,
            )
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
