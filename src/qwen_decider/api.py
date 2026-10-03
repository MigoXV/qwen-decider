"""Bounded, stateless HTTP API. No conversation payload persistence."""

import asyncio
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .engine import Engine
from .prompt import InputTooLong
from .schema import DecisionRequest

logger = logging.getLogger(__name__)


class BodyLimit:
    def __init__(self, app, limit=262144):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0
        messages = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                await Response("request body exceeds limit", status_code=413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)


def create_app(settings, engine=None):
    engine = engine or Engine(settings)
    active = set()
    ready = False
    registry = CollectorRegistry()
    requests = Counter("decider_requests", "Completed requests", ["outcome"], registry=registry)
    inflight = Gauge("decider_inflight", "Admitted HTTP requests", registry=registry)
    latency = Histogram("decider_request_seconds", "Request wall time", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready
        try:
            await engine.start()
            ready = True
            yield
        finally:
            ready = False
            for task in tuple(active):
                task.cancel()
            if active:
                await asyncio.gather(*tuple(active), return_exceptions=True)
            await engine.close()

    app = FastAPI(title="qwen-decider", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    @app.get("/health/live")
    async def live():
        return {"alive": True}

    @app.get("/health/ready")
    async def health():
        if not ready or not await engine.health():
            raise HTTPException(503, "engine unavailable")
        return {"ready": True}

    @app.get("/v1/info")
    async def info():
        return {
            **engine.info,
            "ready": ready and await engine.health(),
            "inflight": len(active),
            "capacity": settings.max_requests,
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/decisions")
    async def decide(payload: DecisionRequest, request: Request):
        if not ready or not await engine.health():
            raise HTTPException(503, "engine unavailable")
        # No await between admission check and reservation.
        if len(active) >= settings.max_requests:
            requests.labels("overloaded").inc()
            raise HTTPException(503, "request capacity exhausted")
        owner = asyncio.current_task()
        active.add(owner)
        inflight.inc()
        started = perf_counter()
        work = watch = None
        outcome = "error"
        try:
            # Bounded request preflight: validate every question before submitting any.
            prepared = [
                (key, engine.prepare(payload.state, q), q) for key, q in payload.questions.items()
            ]
            rid = uuid4().hex

            async def infer():
                tasks = [
                    asyncio.create_task(engine.predict(ids, q, f"{rid}-{i}"))
                    for i, (_, ids, q) in enumerate(prepared)
                ]
                try:
                    return await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            async def disconnected():
                while not await request.is_disconnected():
                    await asyncio.sleep(0.05)

            work = asyncio.create_task(infer())
            watch = asyncio.create_task(disconnected())
            remaining = settings.request_timeout - (perf_counter() - started)
            done, _ = await asyncio.wait(
                [work, watch], timeout=max(0, remaining), return_when=asyncio.FIRST_COMPLETED
            )
            if work in done:
                answers = work.result()
            elif watch in done:
                outcome = "disconnected"
                raise HTTPException(499, "client disconnected")
            else:
                outcome = "timeout"
                raise HTTPException(504, "decision deadline exceeded")
            outcome = "success"
            return {
                "answers": {key: answer for (key, _, _), answer in zip(prepared, answers)},
                "usage": {
                    "input_tokens": sum(len(ids) for _, ids, _ in prepared),
                    "output_tokens": 0,
                    "engine_output_tokens": len(prepared),
                },
                "model": engine.info,
                "timings": {"request_ms": round((perf_counter() - started) * 1000, 3)},
            }
        except InputTooLong as exc:
            outcome = "too_long"
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            outcome = "invalid"
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            logger.error("decision failed category=%s", type(exc).__name__)
            raise HTTPException(503, "decision engine failed") from exc
        finally:
            for task in (work, watch):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (work, watch) if t is not None), return_exceptions=True
            )
            active.discard(owner)
            inflight.dec()
            requests.labels(outcome).inc()
            latency.observe(perf_counter() - started)

    return app
