"""Run in the pre-existing training environment; generates same-dtype HF reference."""

import argparse
import json
from pathlib import Path

import torch
from meetnote_qwen_decision_train.models.processing import checked_sequence, letter_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--precision", choices=["bf16", "fp16"], required=True)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()
torch.set_num_threads(4)
root = Path("/workspace/opusi/qwen-decider")
path = root / "model-bin" / f"decider-v3-{args.precision}"
dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
model = (
    AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    )
    .to(args.device)
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
letters = letter_ids(tokenizer)
source = Path("/workspace/opus/meetnote-decision-train/data-bin")
rows = []
with torch.inference_mode():
    for file in [
        "hard-holdout-v1/hard-holdout-v1.jsonl",
        "final-holdout-v2-meetnote-options/test.jsonl",
    ]:
        for line in (source / file).read_text().splitlines():
            r = json.loads(line)
            q = json.loads(r["question"])
            state = json.loads(r["state"])
            ids, slot = checked_sequence(tokenizer, state, q, {"max_len": 2048})
            tokens = torch.tensor([ids], device=args.device)
            hidden = model.model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
            ).last_hidden_state[0, -1]
            weight = model.get_output_embeddings().weight[letters[: len(q["criteria"])]]
            logits = torch.nn.functional.linear(hidden, weight).float()
            probs = logits.softmax(-1).cpu().tolist()
            keys = list(q["criteria"])
            rows.append(
                {
                    "id": r["id"],
                    "dataset": file,
                    "choice": keys[max(range(len(keys)), key=probs.__getitem__)],
                    "probabilities": dict(zip(keys, probs)),
                }
            )
(root / f"outputs/reference-{args.precision}.json").write_text(
    json.dumps({"precision": args.precision, "rows": rows}, ensure_ascii=False, indent=2)
)
print("saved", len(rows), "reference predictions")
