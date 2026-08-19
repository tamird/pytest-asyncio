from __future__ import annotations

from textwrap import dedent

import pytest
from pytest import Pytester


def test_task_is_cancelled_when_abandoned_by_test(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest

        @pytest.mark.asyncio
        async def test_create_task():
            async def coroutine():
                try:
                    while True:
                        await asyncio.sleep(0)
                finally:
                    raise RuntimeError("The task should be cancelled at this point.")

            asyncio.create_task(coroutine())
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("suspend", [False, True])
@pytest.mark.parametrize("outcome", ["fail", "xfail"])
def test_pytest_failure_in_callback_fails_test(
    pytester: Pytester, suspend: bool, outcome: str
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_callback():
            asyncio.get_running_loop().call_soon(pytest.{outcome}, "callback failure")
            if {suspend!r}:
                await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later():
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-rx")
    result.assert_outcomes(
        failed=int(outcome == "fail"), xfailed=int(outcome == "xfail"), passed=1
    )
    result.stdout.fnmatch_lines(["*callback failure*"])


@pytest.mark.parametrize(
    "phase", ["queued", "coroutine", "generator_setup", "teardown"]
)
def test_pytest_failure_in_fixture_callback(pytester: Pytester, phase: str):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import pytest
        import pytest_asyncio

        async def fail():
            asyncio.get_running_loop().call_soon(pytest.fail, "callback failure")
            await asyncio.sleep(0)

        @pytest_asyncio.fixture
        def queued():
            if {phase!r} == "queued":
                asyncio.get_event_loop().call_soon(pytest.fail, "callback failure")

        @pytest_asyncio.fixture
        async def coroutine():
            if {phase!r} == "coroutine":
                await fail()

        @pytest_asyncio.fixture
        async def generator():
            if {phase!r} == "generator_setup":
                await fail()
            yield
            if {phase!r} == "teardown":
                await fail()

        @pytest.mark.asyncio
        async def test_fixture(queued, coroutine, generator):
            pass

        @pytest.mark.asyncio
        async def test_later():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1, passed=2 if phase == "teardown" else 1)
    result.stdout.fnmatch_lines(["*Failed: callback failure*"])


def test_shared_fixture_can_delegate_to_saved_exception_handler(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def handled():
            loop = asyncio.get_running_loop()
            previous = loop.get_exception_handler()
            errors = []
            def handler(loop, details):
                errors.append(details["exception"])
                previous(loop, details)
            loop.set_exception_handler(handler)
            yield errors
            loop.set_exception_handler(previous)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_ordinary_error(handled):
            def callback():
                raise ValueError("ordinary error")
            asyncio.get_running_loop().call_soon(callback)
            await asyncio.sleep(0)
            assert isinstance(handled[0], ValueError)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_explicit_failure(handled):
            asyncio.get_running_loop().call_soon(pytest.fail, "callback failure")
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later(handled):
            assert len(handled) == 2
            assert isinstance(handled[1], pytest.fail.Exception)
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(failed=1, passed=2)
    result.stdout.fnmatch_lines(["*Failed: callback failure*"])


def test_callback_failure_does_not_hide_keyboard_interrupt(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest

        @pytest.mark.asyncio
        async def test_interrupt():
            asyncio.get_running_loop().call_soon(pytest.fail, "callback failure")
            try:
                await asyncio.sleep(0)
            finally:
                raise KeyboardInterrupt("interrupted cleanup")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*KeyboardInterrupt: interrupted cleanup*"])
