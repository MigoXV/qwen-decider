from qwen_decider.models import register


def main():
    register()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t = AutoTokenizer.from_pretrained("model-bin/decider-v3-bf16", local_files_only=True)
    llm = LLM(
        model="model-bin/decider-v3-bf16",
        dtype="bfloat16",
        max_model_len=2049,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_num_seqs=8,
        gpu_memory_utilization=0.45,
        logprobs_mode="processed_logprobs",
    )
    ids = t.encode("Context:\n小李负责验收。", add_special_tokens=False) + t.encode(
        "\n\nQuestion: 谁负责验收？\nOptions:\n(A) 小李\n(B) 小王\nAnswer: (",
        add_special_tokens=False,
    )
    print(
        llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(max_tokens=1, temperature=0, allowed_token_ids=[32, 33], logprobs=2),
        )
    )


if __name__ == "__main__":
    main()
