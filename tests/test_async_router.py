"""Unit tests for `AsyncRequestRouter` (vllm_rollout_spmd.py).

The router wraps a vLLM `LLMEngine` with an asyncio facade. We don't import
real vLLM here — a `FakeEngine` exposes only the surface the router uses
(`add_request`, `step`, `has_unfinished_requests`) and a fake `RequestOutput`
mimics the field shape the router reads. Keeps these tests CPU-only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

try:
    import pytest
except ModuleNotFoundError:  # allow `python -m unittest`-style runs
    pytest = None  # type: ignore[assignment]

from verl.workers.rollout.vllm_rollout_spmd import AsyncRequestRouter


# ── fakes ─────────────────────────────────────────────────────────────


@dataclass
class _FakeOut:
    token_ids: list[int]
    logprobs: Optional[list] = None
    finish_reason: str = "stop"


@dataclass
class _FakeRequestOutput:
    request_id: str
    finished: bool = True
    outputs: list[_FakeOut] = field(default_factory=list)


class FakeEngine:
    """Fake `LLMEngine` enough for the router. Each `step()` finalizes one
    pending request (FIFO order). Latency between submission and completion is
    therefore ``num_step_calls * sleep_per_step``."""

    def __init__(self, completion_tokens=(1, 2, 3)):
        self._queue: list[str] = []
        self.completion_tokens = list(completion_tokens)
        self.add_request_calls = 0
        self.step_calls = 0

    def add_request(self, request_id, prompt, params, lora_request=None):
        self.add_request_calls += 1
        self._queue.append(request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self._queue)

    def step(self) -> list[_FakeRequestOutput]:
        self.step_calls += 1
        if not self._queue:
            return []
        rid = self._queue.pop(0)
        return [
            _FakeRequestOutput(
                request_id=rid,
                finished=True,
                outputs=[_FakeOut(token_ids=list(self.completion_tokens))],
            )
        ]


class StuckEngine(FakeEngine):
    """Never finishes a request — `wait_idle` should block."""

    def step(self):
        self.step_calls += 1
        # Touch the queue but leave it intact: pretend the engine is busy
        # decoding without ever hitting EOS.
        return []


class RaisingEngine(FakeEngine):
    """Raises during `step()` — router should fail pending futures."""

    def step(self):
        raise RuntimeError("engine kaboom")


# ── tests ─────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def test_generate_one_returns_tokens():
    engine = FakeEngine(completion_tokens=(7, 8, 9))
    router = AsyncRequestRouter(engine)

    async def go():
        result = await router.generate_one(
            request_id="req-0",
            prompt_token_ids=[1, 2, 3],
            sampling_params=object(),  # router doesn't introspect this
        )
        await router.stop()
        return result

    out = _run(go())
    assert out.token_ids == [7, 8, 9]
    assert out.finish_reason == "stop"
    assert out.logprobs is None
    assert engine.add_request_calls == 1
    assert engine.step_calls >= 1


def test_wait_idle_blocks_until_complete():
    engine = FakeEngine()
    router = AsyncRequestRouter(engine)

    async def go():
        # Fire 3 concurrent requests; wait_idle should not return until all
        # three resolve.
        tasks = [
            asyncio.create_task(
                router.generate_one(
                    request_id=f"r{i}",
                    prompt_token_ids=[i],
                    sampling_params=object(),
                )
            )
            for i in range(3)
        ]
        # Let the step loop finish them.
        await asyncio.gather(*tasks)
        await router.wait_idle()
        # After wait_idle, no pending futures remain and queue is empty.
        assert not engine.has_unfinished_requests()
        assert not router._pending
        await router.stop()

    _run(go())


def test_wait_idle_does_not_return_while_busy():
    engine = StuckEngine()
    router = AsyncRequestRouter(engine, sleep_quantum=0.001)

    async def go():
        gen_task = asyncio.create_task(
            router.generate_one(
                request_id="stuck",
                prompt_token_ids=[1],
                sampling_params=object(),
            )
        )
        # Give the step loop a moment to start polling.
        await asyncio.sleep(0.05)
        # wait_idle should time out — i.e. not return — because the engine
        # never finishes. We verify by racing it against a small timer.
        idle_done = False

        async def _check_idle():
            nonlocal idle_done
            await router.wait_idle(poll_interval=0.001)
            idle_done = True

        idle_task = asyncio.create_task(_check_idle())
        await asyncio.sleep(0.1)
        assert not idle_done, "wait_idle returned while engine still busy"

        # Cancel everything so the test exits cleanly.
        idle_task.cancel()
        gen_task.cancel()
        for t in (idle_task, gen_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await router.stop()

    _run(go())


def test_engine_exception_propagates_to_pending_futures():
    engine = RaisingEngine()
    router = AsyncRequestRouter(engine)

    async def go():
        with pytest_raises(RuntimeError):
            await router.generate_one(
                request_id="boom",
                prompt_token_ids=[1],
                sampling_params=object(),
            )
        # Drain the step-loop task so its exception isn't reported as
        # "Task exception was never retrieved" at GC time.
        if router._step_task is not None:
            try:
                await router._step_task
            except Exception:
                pass

    _run(go())


# ── tiny shim so the file works whether pytest is installed or not ────


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, tb):
        if etype is None:
            raise AssertionError(f"expected {self.exc_type.__name__}, got nothing")
        return issubclass(etype, self.exc_type)


def pytest_raises(exc_type):
    if pytest is not None:
        return pytest.raises(exc_type)
    return _Raises(exc_type)


if __name__ == "__main__":
    test_generate_one_returns_tokens()
    test_wait_idle_blocks_until_complete()
    test_wait_idle_does_not_return_while_busy()
    test_engine_exception_propagates_to_pending_futures()
    print("OK")
