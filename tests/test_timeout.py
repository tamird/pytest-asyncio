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


@pytest.mark.parametrize("startup", ["timeout", "error"])
def test_timeout_before_task_start(
    pytester: Pytester, startup: str, cooperative_timeout: None
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import pytest

        failures = []

        @pytest.hookimpl(wrapper=True)
        def pytest_timeout_expired(item, exception):
            failures.append(exception)
            return (yield)
        """))
    pytester.makepyfile(dedent(f"""\
        import contextvars
        import inspect
        import signal
        from asyncio import Runner
        import pytest
        from conftest import failures
        from pytest_asyncio._timeout import run

        @pytest.mark.timeout(10, method="signal", func_only=True)
        def test_startup(request):
            startup = {startup!r}
            entered = []
            wrappers = []
            failure = ValueError("task creation failed")

            async def body():
                entered.append(True)

            async def later():
                return 42

            with Runner() as runner:
                loop = runner.get_loop()

                def task_factory(loop, coro, **kwargs):
                    loop.set_task_factory(None)
                    wrappers.append(coro)
                    if startup == "error":
                        raise failure
                    signal.raise_signal(signal.SIGALRM)
                    return loop.create_task(coro, **kwargs)

                loop.set_task_factory(task_factory)
                coro = body()
                expected = ValueError if startup == "error" else pytest.fail.Exception
                with pytest.raises(expected) as caught:
                    run(
                        runner, coro, context=contextvars.copy_context(),
                        config=request.config,
                    )
                assert caught.value is (failure if startup == "error" else failures[0])
                assert not entered
                assert len(wrappers) == 1 and wrappers[0] is not coro
                assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                assert inspect.getcoroutinestate(wrappers[0]) == inspect.CORO_CLOSED
                assert run(
                    runner, later(), context=contextvars.copy_context(),
                    config=request.config,
                ) == 42
        """))
    result = pytester.runpytest_subprocess(timeout=10)
    result.assert_outcomes(passed=1)
    assert "was never awaited" not in result.stdout.str() + result.stderr.str()


@pytest.mark.parametrize("trigger", ["timer", "reschedule"])
def test_signal_timeout_preserves_shared_loop(
    pytester: Pytester, trigger: str, cooperative_timeout: None
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import signal
        import time
        import pytest

        cleaned = []

        async def application_wait():
            await asyncio.Future()

        @pytest.mark.timeout(
            {0.1 if trigger == "timer" else 10}, method="signal", func_only=True
        )
        @pytest.mark.asyncio(loop_scope="module")
        async def test_timeout():
            loop = asyncio.get_running_loop()
            original = loop.call_soon
            task = asyncio.current_task()

            def reschedule(callback, *args, context=None):
                if getattr(callback, "__self__", None) is task:
                    loop.call_soon = original
                    signal.raise_signal(signal.SIGALRM)
                return original(callback, *args, context=context)

            if {trigger!r} == "reschedule":
                loop.call_soon = reschedule
            else:
                loop.call_soon(time.sleep, 0.2)
            try:
                await asyncio.sleep(0)
                await application_wait()
            finally:
                loop.call_soon = original
                cleaned.append(True)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later():
            assert cleaned == [True]

        @pytest.mark.timeout(10, method="signal", func_only=True)
        def test_synchronous():
            with pytest.raises(pytest.fail.Exception, match="Timeout"):
                signal.raise_signal(signal.SIGALRM)
        """))
    result = pytester.runpytest_subprocess("--tb=short", timeout=10)
    result.assert_outcomes(failed=1, passed=2)
    result.stdout.fnmatch_lines(["*Failed: Timeout*from pytest-timeout.*"])
    if trigger == "timer":
        result.stdout.fnmatch_lines(["*in application_wait*", "*CancelledError*"])
    assert "_timeout.py" not in result.stdout.str()


@pytest.mark.parametrize(
    "cleanup",
    ["return", "xfail", "error", "interrupt", "exit", "system_exit", "cancel"],
)
def test_timeout_preserves_process_control(
    pytester: Pytester, cleanup: str, cooperative_timeout: None
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(f"""\
        import asyncio
        import pytest

        failures = []

        @pytest.hookimpl(wrapper=True)
        def pytest_timeout_expired(item, exception):
            failures.append(exception)
            return (yield)

        def pytest_runtest_makereport(item, call):
            if call.when == "call":
                error = call.excinfo.value
                if {cleanup!r} == "cancel":
                    assert isinstance(error, asyncio.CancelledError)
                    assert error.__cause__ is failures[0]
                elif {cleanup!r} == "system_exit":
                    assert isinstance(error, SystemExit) and error.code == 7
                else:
                    assert error is failures[0]
        """))
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import signal
        import pytest

        @pytest.mark.timeout(10, method="signal", func_only=True)
        @pytest.mark.asyncio
        async def test_timeout():
            loop = asyncio.get_running_loop()
            loop.call_soon(signal.raise_signal, signal.SIGALRM)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if {cleanup!r} == "return":
                    return
                if {cleanup!r} == "xfail":
                    pytest.xfail("cleanup xfail")
                if {cleanup!r} == "error":
                    raise ValueError("cleanup error")
                if {cleanup!r} == "exit":
                    pytest.exit("requested exit", returncode=4)
                if {cleanup!r} == "system_exit":
                    raise SystemExit(7)
                if {cleanup!r} == "cancel":
                    asyncio.current_task().cancel()
                else:
                    signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(0)
        """))
    result = pytester.runpytest_subprocess(timeout=10)
    if cleanup == "interrupt":
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])
    elif cleanup == "exit":
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stdout.fnmatch_lines(["*Exit: requested exit*"])
    elif cleanup == "system_exit":
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*SystemExit: 7*"])
    elif cleanup == "cancel":
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*Failed: Timeout*", "*CancelledError*"])
    else:
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*Failed: Timeout*from pytest-timeout.*"])


@pytest.mark.parametrize(
    "phase",
    ["coroutine_setup", "generator_setup", "teardown", "shutdown", "shutdown_boundary"],
)
def test_timeout_during_async_cleanup(
    pytester: Pytester, phase: str, cooperative_timeout: None
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
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
        async def coroutine():
            if {phase!r} == "coroutine_setup":
                await timeout()

        @pytest_asyncio.fixture
        async def fixture():
            if {phase!r} == "generator_setup":
                await timeout()
            yield
            if {phase!r} == "teardown":
                await timeout()

        @pytest.mark.timeout(10, method="signal")
        @pytest.mark.asyncio
        async def test_timeout(coroutine, fixture):
            global executor
            if {phase!r} == "shutdown":
                asyncio.create_task(background())
                await asyncio.sleep(0)
            if {phase!r} == "shutdown_boundary":
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
    result.assert_outcomes(errors=1, passed=1 if phase.endswith("setup") else 2)
    result.stdout.fnmatch_lines(["*Failed: Timeout*from pytest-timeout.*"])


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
