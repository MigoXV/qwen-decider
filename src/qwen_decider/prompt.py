"""Training-compatible state-first encoding; no chat template."""

import json
import math

from .schema import Question

LETTERS = "ABCDEFGHIJ"


class InputTooLong(ValueError):
    pass


def letter_ids(tokenizer):
    ids = []
    prefix = tokenizer.encode("\nAnswer: (", add_special_tokens=False)
    for letter in LETTERS:
        item = tokenizer.encode(letter, add_special_tokens=False)
        if len(item) != 1 or item[0] in tokenizer.all_special_ids:
            raise ValueError("answer letters must be single ordinary tokens")
        if tokenizer.encode("\nAnswer: (" + letter, add_special_tokens=False) != prefix + item:
            raise ValueError("answer letter merges with prefix")
        ids.extend(item)
    if len(set(ids)) != 10:
        raise ValueError("duplicate answer token IDs")
    return ids


def prepare(tokenizer, state, question: Question, limit=2048):
    state = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
    )
    options = [key if not desc else f"{key}: {desc}" for key, desc in question.criteria.items()]
    suffix = (
        "\n\nQuestion: "
        + question.instructions
        + "\nOptions:"
        + "".join(f"\n({LETTERS[i]}) {v}" for i, v in enumerate(options))
        + "\nAnswer: ("
    )
    ids = tokenizer.encode("Context:\n" + state, add_special_tokens=False) + tokenizer.encode(
        suffix, add_special_tokens=False
    )
    if len(ids) > limit:
        raise InputTooLong("complete question exceeds input token limit")
    if set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("reserved control token in input")
    return ids


def decode(keys, token_ids, logprobs):
    if any(t not in logprobs for t in token_ids):
        raise RuntimeError("missing option logprob")
    values = [logprobs[t].logprob for t in token_ids]
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("nonfinite option logprob")
    peak = max(values)
    probs = [math.exp(v - peak) for v in values]
    total = sum(probs)
    probs = [v / total for v in probs]
    winner = max(range(len(keys)), key=probs.__getitem__)
    return {
        "type": "choice",
        "choice": keys[winner],
        "probabilities": dict(zip(keys, probs)),
        "confidence": probs[winner],
    }
