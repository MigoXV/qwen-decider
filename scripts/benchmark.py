"""Decision-only service benchmark. Repeated synthetic cases, no result cache."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

import httpx
from openai import AsyncOpenAI


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(p * len(values)) - 1)] if values else None


async def main(args):
    rows = [json.loads(s) for s in Path(args.data).read_text().splitlines()]
    selected = []
    for task in sorted({r["task"] for r in rows}):
        selected.extend([r for r in rows if r["task"] == task][:2])
    outputs = []
    async with (
        httpx.AsyncClient(base_url=args.url, timeout=90) as client,
        AsyncOpenAI(base_url=args.baseline, api_key="local", max_retries=0, timeout=90) as base,
    ):
        info = (await client.get("/v1/info")).json()
        if not info.get("ready") or info.get("backend") != "vllm" or info.get("enforce_eager"):
            raise RuntimeError("benchmark requires ready accelerated vLLM service")

        async def run(backend, row, questions):
            q = json.loads(row["question"])
            state = json.loads(row["state"])
            # Multiple independent questions share the same state; no duplicate-ID collapsing.
            qs = {f"q{i}": q for i in range(questions)}
            start = perf_counter()
            error = None
            usage = None
            answers = None
            try:
                if backend == "qwen":
                    res = await client.post("/v1/decisions", json={"state": state, "questions": qs})
                    res.raise_for_status()
                    data = res.json()
                    answers = data["answers"]
                    usage = data["usage"]
                else:
                    props = {key: {"type": "string", "enum": list(q["criteria"])} for key in qs}
                    schema = {
                        "type": "object",
                        "properties": props,
                        "required": list(props),
                        "additionalProperties": False,
                    }
                    res = await base.responses.create(
                        model=args.model,
                        instructions="按各题 instructions 判断，仅返回所选选项键，严格遵守 JSON Schema。",
                        input=json.dumps({"state": state, "questions": qs}, ensure_ascii=False),
                        temperature=0,
                        max_output_tokens=256,
                        store=False,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "decisions",
                                "strict": True,
                                "schema": schema,
                            }
                        },
                        extra_body={"enable_thinking": False},
                    )
                    parsed = json.loads(res.output_text)
                    if set(parsed) != set(qs) or any(
                        v not in q["criteria"] for v in parsed.values()
                    ):
                        raise ValueError("invalid baseline output")
                    answers = {k: {"choice": v} for k, v in parsed.items()}
                    usage = res.usage.model_dump() if res.usage else None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
            return {
                "case": row["id"],
                "task": row["task"],
                "seconds": perf_counter() - start,
                "error": error,
                "usage": usage,
                "choices": {k: v["choice"] for k, v in answers.items()} if answers else None,
            }

        backends = ["qwen"] if args.qwen_only else ["qwen", "spark4b"]
        for backend in backends:
            for row in selected[:4]:
                await run(backend, row, 1)
        for questions in [1, 8]:
            for concurrency in [1, 2, 4, 8]:
                for repeat in range(3):
                    for backend in backends if repeat % 2 == 0 else list(reversed(backends)):
                        sem = asyncio.Semaphore(concurrency)

                        async def limited(row):
                            async with sem:
                                return await run(backend, row, questions)

                        start = perf_counter()
                        records = await asyncio.gather(*(limited(r) for r in selected))
                        elapsed = perf_counter() - start
                        valid = [r["seconds"] for r in records if r["error"] is None]
                        report = {
                            "backend": backend,
                            "questions_per_request": questions,
                            "concurrency": concurrency,
                            "round": repeat,
                            "requests": len(records),
                            "errors": len(records) - len(valid),
                            "elapsed_seconds": elapsed,
                            "questions_per_second": len(valid) * questions / elapsed,
                            "p50_seconds": percentile(valid, 0.5),
                            "p95_seconds": percentile(valid, 0.95),
                            "p99_seconds": percentile(valid, 0.99),
                            "max_seconds": max(valid) if valid else None,
                            "rows": records,
                        }
                        outputs.append(report)
                        print(
                            json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True
                        )
                        Path(args.output).write_text(
                            json.dumps(
                                {
                                    "candidate": info,
                                    "baseline": args.baseline,
                                    "baseline_hardware": "not independently verified",
                                    "records": outputs,
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
    if any(r["errors"] for r in outputs):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:10013")
    p.add_argument("--baseline", default="http://192.168.0.222:10001/v1")
    p.add_argument("--model", default="spark25")
    p.add_argument(
        "--data",
        default="/workspace/opus/meetnote-decision-train/data-bin/hard-holdout-v1/hard-holdout-v1.jsonl",
    )
    p.add_argument("--output", default="outputs/benchmark.json")
    p.add_argument("--qwen-only", action="store_true")
    asyncio.run(main(p.parse_args()))
