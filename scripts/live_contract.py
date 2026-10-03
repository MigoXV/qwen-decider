"""Exercise real service limits, multi-question isolation and cancellation recovery."""

import asyncio
import json
from pathlib import Path

import httpx


async def main():
    results = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:10013", timeout=90) as c:
        for n in range(2, 11):
            q = {
                "type": "choice",
                "instructions": "请选择与上下文相符的选项",
                "criteria": {f"k{i}": str(i) for i in range(n)},
            }
            r = await c.post(
                "/v1/decisions", json={"state": "0", "questions": {"one": q, "two": q}}
            )
            r.raise_for_status()
            answers = r.json()["answers"]
            assert answers["one"]["choice"] == answers["two"]["choice"]
            for a in answers.values():
                assert set(a["probabilities"]) == set(q["criteria"])
                assert abs(sum(a["probabilities"].values()) - 1) < 1e-6
            results.append({"options": n, "status": r.status_code})
        q = {"instructions": "test", "criteria": {"yes": "yes", "no": "no"}}
        r = await c.post(
            "/v1/decisions", json={"state": "中文内容。" * 4000, "questions": {"answer": q}}
        )
        assert r.status_code == 413
        results.append({"overlong_status": r.status_code})
        try:
            await c.post(
                "/v1/decisions", json={"state": "x", "questions": {"answer": q}}, timeout=0.001
            )
        except httpx.TimeoutException:
            pass
        await asyncio.sleep(0.2)
        assert (await c.get("/health/ready")).status_code == 200
        assert (await c.get("/v1/info")).json()["inflight"] == 0
        results.append({"cancel_recovered": True})
    Path("outputs/live-contract.json").write_text(json.dumps(results, indent=2))
    print("live contract passed")


if __name__ == "__main__":
    asyncio.run(main())
