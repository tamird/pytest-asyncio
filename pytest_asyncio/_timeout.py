"""Cooperative delivery of pytest-timeout's signal failures."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import sys
import threading
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass
from types import CoroutineType
from typing import Any, TypeVar

import pytest


class _RunnerState(threading.local):
    invocation: _Delivery | None = None


@dataclass
class _Delivery:
    loop: asyncio.AbstractEventLoop
    closing: bool = False
    exception: BaseException | None = None
    timeout: asyncio.Timeout | None = None

    def interrupt(self, state: _RunnerState) -> None:
        if state.invocation is not self:
            return
        if self.closing:
            # A completed shutdown phase can consume stop(). Keep stopping
            # until Runner.close() returns; never stop a reusable invocation.
            self.loop.stop()
            self.loop.call_soon(self.interrupt, state)
        elif self.timeout is not None:
            self.timeout.reschedule(self.loop.time())


_RUNNER_STATE = pytest.StashKey[_RunnerState]()
_T = TypeVar("_T")


def configure(config: pytest.Config) -> None:
    if sys.version_info < (3, 11):
        return
    if not config.hook.pytest_timeout_expired.has_spec():
        return
    config.stash[_RUNNER_STATE] = _RunnerState()


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_expired(item: pytest.Item, exception: BaseException) -> bool | None:
    state = item.config.stash.get(_RUNNER_STATE, None)
    if state is None or state.invocation is None:
        return None
    invocation = state.invocation
    if invocation.exception is None:
        invocation.exception = exception
        # Raising here can interrupt asyncio before it schedules a task's next
        # step. Return to the interrupted code and cancel at a safe loop turn.
        # Late callbacks check ownership instead of relying on Handle.cancel():
        # SIGINT can interrupt scheduling before the handle is returned.
        if not invocation.loop.is_closed():
            invocation.loop.call_soon_threadsafe(invocation.interrupt, state)
    return True


@contextlib.contextmanager
def _deliver(config: pytest.Config, invocation: _Delivery) -> Iterator[None]:
    __tracebackhide__ = True
    state = config.stash[_RUNNER_STATE]
    previous = state.invocation
    state.invocation = invocation
    try:
        try:
            yield
        finally:
            # Once the runner returns, a new signal can fail synchronously.
            # Stop claiming it before deciding which outcome to propagate.
            state.invocation = previous
    except (KeyboardInterrupt, SystemExit, pytest.exit.Exception):
        raise
    except asyncio.CancelledError as exc:
        if invocation.exception is None:
            raise
        # asyncio.Timeout converts only its own cancellation to TimeoutError.
        # Preserve cancellation requested by another caller.
        raise exc from invocation.exception
    except BaseException as exc:
        if invocation.exception is None or exc is invocation.exception:
            raise
        raise invocation.exception from exc
    if invocation.exception is not None:
        raise invocation.exception


def run(
    runner: asyncio.Runner,
    coro: Coroutine[Any, Any, _T],
    *,
    context: contextvars.Context,
    config: pytest.Config,
) -> _T:
    __tracebackhide__ = True
    if _RUNNER_STATE not in config.stash or not isinstance(coro, CoroutineType):
        return runner.run(coro, context=context)

    invocation = _Delivery(runner.get_loop())

    async def invoke() -> _T:
        __tracebackhide__ = True
        if invocation.exception is not None:
            raise invocation.exception
        try:
            async with asyncio.timeout(None) as timeout:
                invocation.timeout = timeout
                return await coro
        finally:
            # The signal may have queued delivery just as the coroutine exits.
            # Do not reschedule a timeout whose context has already exited.
            invocation.timeout = None

    wrapped = invoke()
    try:
        with _deliver(config, invocation):
            return runner.run(wrapped, context=context)
    except BaseException:

        def close_unstarted(_task: asyncio.Future[Any] | None = None) -> None:
            # Never run suspended user cleanup outside its task.
            for coroutine in (wrapped, coro):
                if inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
                    coroutine.close()

        # A task can own the wrapper without having started it.
        for task in asyncio.all_tasks(invocation.loop):
            if task.get_coro() is wrapped:
                task.add_done_callback(close_unstarted)
                break
        else:
            close_unstarted()
        raise


def close(runner: asyncio.Runner, *, config: pytest.Config) -> None:
    __tracebackhide__ = True
    if _RUNNER_STATE not in config.stash:
        runner.close()
        return
    with _deliver(config, _Delivery(runner.get_loop(), closing=True)):
        runner.close()
