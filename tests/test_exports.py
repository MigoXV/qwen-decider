import json
import tempfile
import unittest
from pathlib import Path

import torch
from qwen_decider.convert import convert
from qwen_decider.quantize import quantize
from safetensors.torch import load_file, save_file


class ExportTests(unittest.TestCase):
    def test_native_precision_and_int8_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer").mkdir()
            (source / "tokenizer/tokenizer.json").write_text("{}")
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "meetnote_qwen_decision",
                        "backbone_config": {
                            "model_type": "qwen3_5_text",
                            "architectures": ["Qwen3_5ForCausalLM"],
                        },
                        "max_length": 2048,
                        "letter_token_ids": list(range(32, 42)),
                    }
                )
            )
            original = torch.tensor([[0.0, 1.0, -2.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
            save_file(
                {
                    "lm.model.language_model.layers.0.mlp.up_proj.weight": original,
                    "lm.model.language_model.embed_tokens.weight": original.clone(),
                },
                str(source / "model.safetensors"),
            )
            for precision, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
                out = root / precision
                convert(source, out, precision)
                config = json.loads((out / "config.json").read_text())
                self.assertEqual(config["architectures"], ["Qwen3_5ForCausalLM"])
                self.assertEqual(config["model_type"], "qwen3_5_text")
                for tensor in load_file(str(out / "model.safetensors")).values():
                    self.assertEqual(tensor.dtype, dtype)
                    torch.testing.assert_close(tensor, original.to(dtype), rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    convert(source, out, precision)
            quantize(root / "bf16", root / "int8")
            tensors = load_file(str(root / "int8/model.safetensors"))
            prefix = "model.language_model.layers.0.mlp.up_proj"
            weight = tensors[prefix + ".weight"]
            scale = tensors[prefix + ".weight_scale"]
            self.assertEqual(weight.dtype, torch.int8)
            self.assertTrue(torch.isfinite(scale).all())
            torch.testing.assert_close(weight.float() * scale, original, rtol=0, atol=0.008)
            self.assertEqual(
                tensors["model.language_model.embed_tokens.weight"].dtype, torch.bfloat16
            )
