"""Compare vLLM results against frozen FP32 reference; no gold in inference input."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx


async def main(args):
    data_root = Path(args.data_root)
    refs = Path(args.reference_root)
    pairs = [
        ("hard-holdout-v1/hard-holdout-v1.jsonl", "v3-hard.json"),
        ("final-holdout-v2-meetnote-options/test.jsonl", "v3-meetnote-options-online.json"),
    ]
    reports = []
    async with httpx.AsyncClient(base_url=args.url, timeout=90) as client:
        info = (await client.get("/v1/info")).json()
        for source, reference in pairs:
            source = data_root / source
            ref = json.loads((refs / reference).read_text())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if ref["dataset_sha256"] != digest:
                raise ValueError("reference/data fingerprint mismatch")
            expected = {r["id"]: r for r in ref["rows"]}
            sem = asyncio.Semaphore(args.concurrency)

            async def run(row):
                state = json.loads(row["state"])
                question = json.loads(row["question"])
                async with sem:
                    response = await client.post(
                        "/v1/decisions", json={"state": state, "questions": {"answer": question}}
                    )
                    response.raise_for_status()
                    answer = response.json()["answers"]["answer"]
                old = expected[row["id"]]
                return {
                    "id": row["id"],
                    "dataset": str(source),
                    "choice": answer["choice"],
                    "reference": old["prediction"],
                    "same_choice": answer["choice"] == old["prediction"],
                    "probability_error": max(
                        abs(answer["probabilities"][k] - v) for k, v in old["probabilities"].items()
                    ),
                    "probabilities": answer["probabilities"],
                }

            reports.extend(
                await asyncio.gather(
                    *(run(json.loads(line)) for line in source.read_text().splitlines())
                )
            )
    agreement = sum(r["same_choice"] for r in reports) / len(reports)
    maximum = max(r["probability_error"] for r in reports)
    report = {
        "model": info,
        "rows": reports,
        "agreement": agreement,
        "max_probability_error": maximum,
        "passed": agreement >= 0.99 and maximum <= 0.02,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:10013")
    p.add_argument("--data-root", default="/workspace/opus/meetnote-decision-train/data-bin")
    p.add_argument(
        "--reference-root",
        default="/workspace/opus/meetnote-qwen-decision-train/outputs/evaluations",
    )
    p.add_argument("--output", default="outputs/parity.json")
    p.add_argument("--concurrency", type=int, default=1)
    asyncio.run(main(p.parse_args()))
