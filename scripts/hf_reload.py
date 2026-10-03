"""Run with the existing training environment (Transformers with native Qwen3.5).
The inference environment remains untouched. Checks standard HF loading only.
"""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path("/workspace/opusi/qwen-decider")
report = []
for precision, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
    path = root / "model-bin" / f"decider-v3-{precision}"
    model, info = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=dtype, output_loading_info=True
    )
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    info = {k: list(v) if isinstance(v, set) else v for k, v in info.items()}
    if any(info.values()):
        raise ValueError(info)
    report.append(
        {
            "precision": precision,
            "model_class": type(model).__name__,
            "loading_info": info,
            "dtype": str(next(model.parameters()).dtype),
            "vocab_size": len(tok),
        }
    )
    del model
    gc.collect()
(root / "outputs/hf-reload.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
