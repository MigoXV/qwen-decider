"""vLLM owns batching, device execution and subprocess lifecycle."""

import asyncio
import json
from importlib.metadata import version

from .convert import fingerprint
from .models import register
from .prompt import decode, letter_ids, prepare
from .schema import Question


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.core = None
        self.info = {}

    async def start(self):
        register()
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        path = self.settings.model_path.resolve()
        native = json.loads((path / "config.json").read_text())
        if native.get("model_type") != "qwen3_5_text" or native.get("num_experts", 0):
            raise ValueError("expected a dense native Qwen3.5 text checkpoint")
        cfg = json.loads((path / "decider_config.json").read_text())
        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(path), local_files_only=True
        )
        self.letters = letter_ids(self.tokenizer)
        if self.letters != cfg["letter_token_ids"]:
            raise ValueError("tokenizer identity mismatch")
        self.limit = min(cfg["max_length"], self.settings.max_input_tokens)
        quantized = json.loads((path / "config.json").read_text()).get("quantization_config")
        if bool(quantized) != (self.settings.precision == "int8"):
            raise ValueError("precision does not match checkpoint quantization")
        dtype = "float16" if self.settings.precision == "fp16" else "bfloat16"
        args = AsyncEngineArgs(
            model=str(path),
            tokenizer=str(path),
            dtype=dtype,
            max_model_len=self.limit + 1,
            max_num_seqs=self.settings.max_num_seqs,
            max_num_batched_tokens=self.settings.max_num_batched_tokens,
            kv_cache_memory_bytes=self.settings.kv_cache_memory_bytes,
            worker_cls="qwen_decider.worker.MemoryBoundWorker",
            additional_config={"decider_memory_limit_bytes": self.settings.gpu_memory_limit_bytes},
            enable_prefix_caching=False,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            enforce_eager=self.settings.enforce_eager,
            logprobs_mode="processed_logprobs",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        self.core = AsyncLLM.from_engine_args(args)
        try:
            q = Question(instructions="请选择一致的选项", criteria={"yes": "是", "no": "否"})
            import logging

            logger = logging.getLogger(__name__)
            base_ids = self.prepare("是", q)
            for length in sorted({min(n, self.limit) for n in (64, 128, 256, 512, 1024, 2048)}):
                logger.info("Warmup input_tokens=%d", length)
                ids = (base_ids * (length // len(base_ids) + 1))[:length]
                await asyncio.wait_for(self.predict(ids, q, f"warmup-{length}"), 300)

            memory = await self.core.collective_rpc("check_memory", timeout=2)
            compilation = self.core.vllm_config.compilation_config
            identity = {
                "weights": await asyncio.to_thread(fingerprint, path),
                "runtime": {k: version(k) for k in ["vllm", "torch", "transformers"]},
                "settings": self.settings.model_dump(mode="json"),
                "code": await asyncio.to_thread(
                    fingerprint, __import__("pathlib").Path(__file__).parent
                ),
            }
            import hashlib

            self.info = {
                "memory": memory,
                "model": str(path),
                "fingerprint": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
                "backend": "vllm",
                "dtype": dtype,
                "precision": self.settings.precision,
                "device": "cuda:0",
                "calibrated": False,
                "versions": identity["runtime"],
                "compilation": str(compilation.mode),
                "cudagraph": str(compilation.cudagraph_mode),
                "enforce_eager": self.settings.enforce_eager,
                "max_input_tokens": self.limit,
                "max_options": 10,
                "gpu_memory_limit_bytes": self.settings.gpu_memory_limit_bytes,
                "allocator_limit_bytes": self.settings.gpu_memory_limit_bytes - 3_000_000_000,
                "kv_cache_memory_bytes": self.settings.kv_cache_memory_bytes,
                "max_num_seqs": self.settings.max_num_seqs,
                "max_num_batched_tokens": self.settings.max_num_batched_tokens,
                "prefix_caching": False,
            }
        except BaseException:
            await self.close()
            raise

    def prepare(self, state, q):
        return prepare(self.tokenizer, state, q, self.limit)

    async def predict(self, ids, q, request_id):
        from vllm import SamplingParams

        tokens = self.letters[: len(q.criteria)]
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            top_p=1,
            top_k=-1,
            allowed_token_ids=tokens,
            logprobs=len(tokens),
            detokenize=False,
        )
        final = None
        try:
            async for result in self.core.generate({"prompt_token_ids": ids}, params, request_id):
                final = result
            if final is None or not final.finished or len(final.outputs[0].token_ids) != 1:
                raise RuntimeError("incomplete decision output")
            return decode(list(q.criteria), tokens, final.outputs[0].logprobs[0])
        finally:
            if self.core is not None:
                await self.core.abort(request_id)

    async def health(self):
        if self.core is None:
            return False
        try:
            await asyncio.wait_for(self.core.check_health(), 2)
            self.info["memory"] = await self.core.collective_rpc("check_memory", timeout=2)
            return True
        except Exception:
            return False

    async def close(self):
        if self.core is not None:
            self.core.shutdown()
            self.core = None
