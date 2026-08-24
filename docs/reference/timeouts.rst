========
Timeouts
========

On Python 3.11 and newer, pytest-asyncio automatically delivers pytest-timeout
signal failures by cancelling the active asynchronous test or fixture when
pytest-timeout provides the ``pytest_timeout_expired`` hook. Cooperative cleanup
can then run before pytest reports the original timeout. No configuration is
needed. Python 3.10 and older pytest-timeout versions retain their existing
signal behavior. Runner invocations with custom coroutine objects retain
synchronous signal delivery; cooperative execution requires native coroutine
objects created by ``async def`` functions.
pytest-timeout still controls the configured duration, covered test phases,
debugger detection, and timeout diagnostics.

Cancellation waits for the event loop and coroutine to cooperate. A raised
signal exception can interrupt CPU-bound Python code, but cooperative
cancellation cannot interrupt code that never yields. It also cannot stop
an indefinitely blocking callback or a task that refuses cancellation.
Use pytest-timeout's ``thread`` method or an independent process watchdog when
the process must be terminated; these stop the entire process without normal
test teardown. A timeout during final event-loop shutdown stops that shutdown
and closes the loop; remaining resource cleanup may be incomplete.

The integration does not take over runners managed by other async plugins or
synchronous tests that call ``asyncio.run()`` themselves. It wraps asynchronous
tests and fixtures in a timeout context, so ``asyncio.current_task().get_coro()``
returns the wrapper coroutine rather than the original test coroutine.
