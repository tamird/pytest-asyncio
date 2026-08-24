from __future__ import annotations

import signal
import sys
from textwrap import dedent

import pytest
from pytest import Pytester

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="requires the signal timeout method"
)


@pytest.fixture
def timeout_plugin(request: pytest.FixtureRequest):
    if not request.config.hook.pytest_timeout_expired.has_spec():
        pytest.skip("requires pytest-timeout's signal-expiry hook")


@pytest.fixture
def cooperative_timeout(timeout_plugin: None):
    if sys.version_info < (3, 11):
        pytest.skip("cooperative timeouts require Python 3.11 or newer")


def test_runner_timeout_delivery(pytester: Pytester, cooperative_timeout: None):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import pytest

        failures = []

        @pytest.hookimpl(wrapper=True)
        def pytest_timeout_expired(item, exception):
            failures.append(exception)
            return (yield)
        """))
    pytester.makepyfile(dedent("""\
        import asyncio
        import contextvars
        import inspect
        import signal
        import sys
        from collections.abc import Coroutine
        import pytest
        from conftest import failures
        from pytest_asyncio._timeout import run

        pytestmark = pytest.mark.timeout(10, method="signal", func_only=True)

        @pytest.mark.parametrize("startup", [
            "timeout", "error", "interrupt_before_start", "interrupt_during_timeout",
            "interrupt_after_start", "custom_coroutine",
        ])
        def test_startup(startup, request, monkeypatch):
            entered = []
            wrappers = []
            tasks = []
            failure = ValueError("task creation failed")

            async def body():
                entered.append(True)
                if startup == "interrupt_after_start":
                    try:
                        await asyncio.Future()
                    finally:
                        raise failure

            async def later():
                return 42

            with asyncio.Runner() as runner:
                loop = runner.get_loop()

                def task_factory(loop, coro, **kwargs):
                    loop.set_task_factory(None)
                    wrappers.append(coro)
                    if startup == "error":
                        raise failure
                    if startup == "timeout":
                        signal.raise_signal(signal.SIGALRM)
                    task = loop.create_task(coro, **kwargs)
                    tasks.append(task)
                    if startup == "custom_coroutine":
                        raise failure
                    return task

                if startup in ("interrupt_before_start", "interrupt_during_timeout"):
                    target, name = (
                        (loop, "run_until_complete")
                        if startup == "interrupt_before_start"
                        else (asyncio, "timeout")
                    )
                    original = getattr(target, name)

                    def interrupt(*args, **kwargs):
                        monkeypatch.setattr(target, name, original)
                        signal.raise_signal(signal.SIGINT)
                        signal.raise_signal(signal.SIGINT)
                        return original(*args, **kwargs)

                    monkeypatch.setattr(target, name, interrupt)

                loop.set_task_factory(task_factory)
                coro = body()
                argument = coro
                if startup == "custom_coroutine":
                    class CustomCoroutine(Coroutine):
                        def send(self, value):
                            return coro.send(value)

                        def throw(self, *args):
                            return coro.throw(*args)

                        def __await__(self):
                            return coro.__await__()

                    argument = CustomCoroutine()

                original_trace = sys.gettrace()
                if startup == "interrupt_after_start":
                    def trace(frame, event, arg):
                        if (
                            wrappers and frame.f_code is wrappers[0].cr_code
                            and event == "return" and arg is not None
                        ):
                            sys.settrace(None)
                            signal.raise_signal(signal.SIGINT)
                            signal.raise_signal(signal.SIGINT)
                        return trace

                    sys.settrace(trace)

                expected = {
                    "timeout": pytest.fail.Exception, "error": ValueError,
                    "custom_coroutine": ValueError,
                }.get(startup, KeyboardInterrupt)
                try:
                    with pytest.raises(expected) as caught:
                        run(
                            runner, argument, context=contextvars.copy_context(),
                            config=request.config,
                        )
                finally:
                    sys.settrace(original_trace)
                if startup in ("timeout", "error", "custom_coroutine"):
                    assert caught.value is (
                        failures[-1] if startup == "timeout" else failure
                    )
                if startup == "interrupt_after_start":
                    with pytest.raises(ValueError) as cleanup:
                        coro.close()
                    assert cleanup.value is failure
                if startup != "custom_coroutine":
                    assert run(
                        runner, later(), context=contextvars.copy_context(),
                        config=request.config,
                    ) == 42
            assert bool(entered) is (startup == "interrupt_after_start")
            assert all(
                inspect.getcoroutinestate(c) == inspect.CORO_CLOSED
                for c in [coro, *wrappers] if inspect.iscoroutine(c)
            )
            if startup in ("interrupt_before_start", "custom_coroutine"):
                assert tasks[0].cancelled()
            elif startup in ("interrupt_during_timeout", "interrupt_after_start"):
                assert tasks[0].exception() is caught.value

        @pytest.mark.parametrize(
            "cleanup",
            ["return", "xfail", "error", "interrupt", "exit", "system_exit", "cancel"],
        )
        def test_cleanup(cleanup, request):
            async def body():
                loop = asyncio.get_running_loop()
                loop.call_soon(signal.raise_signal, signal.SIGALRM)
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if cleanup == "return":
                        return
                    if cleanup == "xfail":
                        pytest.xfail("cleanup xfail")
                    if cleanup == "error":
                        raise ValueError("cleanup error")
                    if cleanup == "exit":
                        pytest.exit("requested exit", returncode=4)
                    if cleanup == "system_exit":
                        raise SystemExit(7)
                    if cleanup == "cancel":
                        asyncio.current_task().cancel()
                    else:
                        signal.raise_signal(signal.SIGINT)
                    await asyncio.sleep(0)

            expected = {
                "interrupt": KeyboardInterrupt,
                "exit": pytest.exit.Exception,
                "system_exit": SystemExit,
                "cancel": asyncio.CancelledError,
            }.get(cleanup, pytest.fail.Exception)
            with asyncio.Runner() as runner:
                with pytest.raises(expected) as caught:
                    run(
                        runner, body(), context=contextvars.copy_context(),
                        config=request.config,
                    )
            if cleanup == "cancel":
                assert caught.value.__cause__ is failures[-1]
            elif cleanup == "system_exit":
                assert caught.value.code == 7
            elif cleanup == "exit":
                assert caught.value.returncode == 4
                assert str(caught.value) == "requested exit"
            elif cleanup != "interrupt":
                assert caught.value is failures[-1]
        """))
    result = pytester.runpytest_subprocess(timeout=10)
    result.assert_outcomes(passed=13)
    assert "was never awaited" not in result.stdout.str() + result.stderr.str()


def test_signal_timeout_preserves_shared_loop(
    pytester: Pytester, cooperative_timeout: None
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import signal
        import time
        import pytest

        cleaned = []

        async def application_wait():
            await asyncio.Future()

        @pytest.mark.parametrize(
            "trigger",
            [
                pytest.param("timer", marks=pytest.mark.timeout(
                    0.1, method="signal", func_only=True,
                )),
                pytest.param("reschedule", marks=pytest.mark.timeout(
                    10, method="signal", func_only=True,
                )),
            ],
        )
        @pytest.mark.asyncio(loop_scope="module")
        async def test_timeout(trigger):
            loop = asyncio.get_running_loop()
            original = loop.call_soon
            task = asyncio.current_task()

            def reschedule(callback, *args, context=None):
                if getattr(callback, "__self__", None) is task:
                    loop.call_soon = original
                    signal.raise_signal(signal.SIGALRM)
                return original(callback, *args, context=context)

            if trigger == "reschedule":
                loop.call_soon = reschedule
            else:
                loop.call_soon(time.sleep, 0.2)
            try:
                await asyncio.sleep(0)
                await application_wait()
            finally:
                loop.call_soon = original
                cleaned.append(trigger)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later():
            assert cleaned == ["timer", "reschedule"]

        @pytest.mark.timeout(10, method="signal", func_only=True)
        def test_synchronous():
            with pytest.raises(pytest.fail.Exception, match="Timeout"):
                signal.raise_signal(signal.SIGALRM)
        """))
    result = pytester.runpytest_subprocess("--tb=short", timeout=10)
    result.assert_outcomes(failed=2, passed=2)
    result.stdout.fnmatch_lines(["E *Failed: Timeout*from pytest-timeout.*"] * 2)
    result.stdout.fnmatch_lines(["*in application_wait*", "*CancelledError*"])
    assert "_timeout.py" not in result.stdout.str()


def test_timeout_during_async_cleanup(pytester: Pytester, cooperative_timeout: None):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import signal
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import pytest
        import pytest_asyncio

        executor = None
        release = threading.Event()
        finished = threading.Event()

        def worker():
            release.wait(5)
            finished.set()

        async def timeout():
            asyncio.get_running_loop().call_soon(signal.raise_signal, signal.SIGALRM)
            await asyncio.Future()

        async def background():
            try:
                await asyncio.Future()
            finally:
                await timeout()

        @pytest_asyncio.fixture
        async def coroutine(phase):
            if phase == "coroutine_setup":
                await timeout()

        @pytest_asyncio.fixture
        async def fixture(phase):
            if phase == "generator_setup":
                await timeout()
            yield
            if phase == "teardown":
                await timeout()

        @pytest.mark.parametrize(
            "phase",
            [
                "coroutine_setup", "generator_setup", "teardown",
                "shutdown", "shutdown_boundary",
            ],
        )
        @pytest.mark.timeout(10, method="signal")
        @pytest.mark.asyncio
        async def test_timeout(phase, coroutine, fixture):
            global executor
            if phase == "shutdown":
                asyncio.create_task(background())
                await asyncio.sleep(0)
            if phase == "shutdown_boundary":
                loop = asyncio.get_running_loop()
                executor = ThreadPoolExecutor()
                loop.set_default_executor(executor)
                loop.run_in_executor(None, worker)
                original = loop.shutdown_asyncgens

                async def shutdown_asyncgens():
                    await original()
                    signal.raise_signal(signal.SIGALRM)

                loop.shutdown_asyncgens = shutdown_asyncgens

        def test_later():
            if executor is not None:
                try:
                    assert not finished.is_set()
                finally:
                    release.set()
                    executor.shutdown(wait=True)
        """))
    result = pytester.runpytest_subprocess(timeout=10)
    result.assert_outcomes(errors=5, passed=4)
    result.stdout.fnmatch_lines(["E *Failed: Timeout*from pytest-timeout.*"] * 5)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="legacy Python 3.10 behavior")
def test_signal_timeout_on_python310(pytester: Pytester, timeout_plugin: None):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import signal
        import pytest

        @pytest.mark.timeout(10, method="signal", func_only=True)
        @pytest.mark.asyncio
        async def test_signal():
            with pytest.raises(pytest.fail.Exception, match="Timeout"):
                signal.raise_signal(signal.SIGALRM)
        """))
    result = pytester.runpytest_subprocess(timeout=10)
    result.assert_outcomes(passed=1)
