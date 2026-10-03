import asyncio
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from pydantic import ValidationError
from qwen_decider.api import create_app
from qwen_decider.prompt import InputTooLong, decode, prepare
from qwen_decider.schema import Question
from qwen_decider.settings import Settings


class Tokenizer:
    all_special_ids = [999999]

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class CoreTests(unittest.TestCase):
    def test_format_preserves_object_and_option_order(self):
        q = Question(instructions="判断", criteria={"b": "乙", "a": "甲"})
        ids = prepare(Tokenizer(), {"乙": 2, "甲": 1}, q)
        text = bytes(ids).decode()
        self.assertEqual(
            text,
            'Context:\n{"乙": 2, "甲": 1}\n\nQuestion: 判断\nOptions:\n(A) b: 乙\n(B) a: 甲\nAnswer: (',
        )

    def test_probabilities_are_option_conditional(self):
        result = decode(
            ["b", "a"],
            [32, 33],
            {32: SimpleNamespace(logprob=-1001), 33: SimpleNamespace(logprob=-1000)},
        )
        self.assertEqual(result["choice"], "a")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
        self.assertAlmostEqual(result["confidence"], 1 / (1 + math.exp(-1)))

    def test_missing_or_nonfinite_scores_fail(self):
        for scores in [
            {32: SimpleNamespace(logprob=0)},
            {32: SimpleNamespace(logprob=0), 33: SimpleNamespace(logprob=float("-inf"))},
        ]:
            with self.assertRaises(RuntimeError):
                decode(["a", "b"], [32, 33], scores)

    def test_limit_never_truncates(self):
        q = Question(instructions="x", criteria={"a": "", "b": ""})
        with self.assertRaises(InputTooLong):
            prepare(Tokenizer(), "abc", q, 2)

    def test_schema_rejects_bad_options(self):
        for criteria in [{"a": "x"}, {"": "x", "b": "y"}, {str(i): "" for i in range(11)}]:
            with self.assertRaises(ValidationError):
                Question(instructions="x", criteria=criteria)


class FakeEngine:
    info = {"fingerprint": "test", "calibrated": False}

    def __init__(self):
        self.calls = 0
        self.cancelled = 0
        self.delay = 0
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True

    async def health(self):
        return True

    def prepare(self, state, q):
        if q.instructions == "long":
            raise InputTooLong("too long")
        return [1, 2]

    async def predict(self, ids, q, rid):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return {
            "type": "choice",
            "choice": next(iter(q.criteria)),
            "probabilities": dict.fromkeys(q.criteria, 0.5),
            "confidence": 0.5,
        }


PAYLOAD = {
    "state": "test",
    "questions": {
        "a": {"type": "choice", "instructions": "test", "criteria": {"yes": "yes", "no": "no"}}
    },
}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.app = create_app(
            Settings(model_path=Path("."), max_requests=1, request_timeout=0.1), self.engine
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.assertTrue(self.engine.closed)

    async def test_response_and_usage(self):
        r = await self.client.post("/v1/decisions", json=PAYLOAD)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json()["usage"], {"input_tokens": 2, "output_tokens": 0, "engine_output_tokens": 1}
        )
        self.assertEqual((await self.client.get("/v1/info")).json()["inflight"], 0)

    async def test_preflight_is_atomic(self):
        p = {
            "state": "x",
            "questions": {
                **PAYLOAD["questions"],
                "bad": {"instructions": "long", "criteria": {"a": "", "b": ""}},
            },
        }
        r = await self.client.post("/v1/decisions", json=p)
        self.assertEqual(r.status_code, 413)
        self.assertEqual(self.engine.calls, 0)

    async def test_timeout_aborts_and_releases(self):
        self.engine.delay = 1
        r = await self.client.post("/v1/decisions", json=PAYLOAD)
        self.assertEqual(r.status_code, 504)
        self.assertEqual(self.engine.cancelled, 1)
        self.engine.delay = 0
        self.assertEqual((await self.client.post("/v1/decisions", json=PAYLOAD)).status_code, 200)

    async def test_backpressure(self):
        self.engine.delay = 0.08
        first = asyncio.create_task(self.client.post("/v1/decisions", json=PAYLOAD))
        await asyncio.sleep(0.02)
        self.assertEqual((await self.client.post("/v1/decisions", json=PAYLOAD)).status_code, 503)
        self.assertEqual((await first).status_code, 200)

    async def test_body_limit(self):
        r = await self.client.post("/v1/decisions", content=b"x" * 262145)
        self.assertEqual(r.status_code, 413)


class ModelLayoutTests(unittest.TestCase):
    def test_loader_rejects_incomplete_packed_weights(self):
        from types import SimpleNamespace

        import torch
        from qwen_decider.models.model import QwenDeciderForCausalLM

        param = torch.nn.Parameter(torch.zeros(3, 2), requires_grad=False)

        def load(target, value, shard):
            target[{"q": 0, "k": 1, "v": 2}[shard]].copy_(value)

        param.weight_loader = load
        key = "model.layers.3.self_attn.qkv_proj.weight"
        runtime_key = "model.layers.3.self_attn.attn.k_scale"
        model = SimpleNamespace(
            named_parameters=lambda: [(key, param), (runtime_key, torch.tensor(-1.0))]
        )
        weights = [
            (f"model.language_model.layers.3.self_attn.{s}_proj.weight", torch.ones(2))
            for s in ["q", "k", "v"]
        ]
        with self.assertRaisesRegex(ValueError, "incomplete packed"):
            QwenDeciderForCausalLM.load_weights(model, weights[:1])
        self.assertEqual(QwenDeciderForCausalLM.load_weights(model, weights), {key})
        torch.testing.assert_close(param, torch.ones(3, 2))

    def test_qwen35_projection_order(self):
        import torch
        from qwen_decider.models.model import GatedDeltaNet

        layer = GatedDeltaNet.__new__(GatedDeltaNet)
        torch.nn.Module.__init__(layer)
        layer.num_k_heads = 2
        layer.num_v_heads = 4
        layer.head_k_dim = 3
        layer.head_v_dim = 2
        layer.key_dim = 6
        layer.value_dim = 8
        qkvz = torch.arange(56, dtype=torch.float32).reshape(2, 28)
        ba = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        q, k, v, z, b, a = layer.fix_query_key_value_ordering(qkvz, ba)
        torch.testing.assert_close(q.flatten(1), qkvz[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(k.flatten(1), qkvz[:, 6:12], rtol=0, atol=0)
        torch.testing.assert_close(v.flatten(1), qkvz[:, 12:20], rtol=0, atol=0)
        torch.testing.assert_close(z.flatten(1), qkvz[:, 20:], rtol=0, atol=0)
        torch.testing.assert_close(b, ba[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(a, ba[:, 4:], rtol=0, atol=0)

    def test_config_text_rope_and_dense(self):
        from qwen_decider.models.config import Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
            tie_word_embeddings=True,
        )
        self.assertTrue(config.tie_word_embeddings)
        self.assertEqual(config.num_experts, 0)
        self.assertNotIn("mrope_section", config.rope_parameters)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.25)
