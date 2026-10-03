"""Destructive probe of an explicitly selected disposable local test instance."""

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path
from time import monotonic

import httpx
import psutil


async def main(args):
    parent = psutil.Process(args.api_pid)
    assert "qwen-decider" in " ".join(parent.cmdline())
    core = psutil.Process(args.core_pid)
    assert core.ppid() == parent.pid
    children = parent.children(recursive=True)
    async with httpx.AsyncClient(base_url=args.url, timeout=10) as client:
        assert (await client.get("/health/ready")).status_code == 200
        q = {"instructions": "判断是否支持", "criteria": {"yes": "是", "no": "否"}}
        task = asyncio.create_task(
            client.post(
                "/v1/decisions",
                json={
                    "state": "讨论事项，暂未确认。" * 100,
                    "questions": {str(i): q for i in range(8)},
                },
            )
        )
        await asyncio.sleep(0.02)
        started = monotonic()
        os.kill(args.core_pid if args.mode == "core" else args.api_pid, signal.SIGKILL)
        try:
            response = await task
            request_result = response.status_code
            assert response.status_code in (200, 503)
        except httpx.TransportError as exc:
            request_result = type(exc).__name__
        elapsed = monotonic() - started
        assert elapsed < 10
        result = {
            "mode": args.mode,
            "request_result": request_result,
            "completion_seconds": elapsed,
        }
        if args.mode == "core":
            await asyncio.sleep(0.2)
            result["ready_status"] = (await client.get("/health/ready")).status_code
            result["live_status"] = (await client.get("/health/live")).status_code
            assert result["ready_status"] == 503 and result["live_status"] == 200
            os.kill(args.api_pid, signal.SIGTERM)
        deadline = monotonic() + 15

        def alive(p):
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE

        while any(alive(p) for p in [parent, *children]) and monotonic() < deadline:
            await asyncio.sleep(0.1)
        result["remaining_pids"] = [p.pid for p in [parent, *children] if alive(p)]
        Path(args.output).write_text(json.dumps(result, indent=2))
        assert not result["remaining_pids"], result
        print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api-pid", type=int, required=True)
    p.add_argument("--core-pid", type=int, required=True)
    p.add_argument("--mode", choices=["core", "parent"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:10013")
    p.add_argument("--output", required=True)
    asyncio.run(main(p.parse_args()))
