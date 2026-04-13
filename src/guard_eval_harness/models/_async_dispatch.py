"""Shared async dispatch helpers for API model adapters.

``httpx`` is an optional dependency (``.[api]`` extra) and is imported
lazily inside :func:`run_async_batch` so that base installs can still
import this module during registry discovery.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Awaitable, Callable, Sequence, TypeVar

if TYPE_CHECKING:
    import httpx

T = TypeVar("T")
U = TypeVar("U")


async def _run_all(
    client: "httpx.AsyncClient",
    factory: Callable[["httpx.AsyncClient", T], Awaitable[U]],
    items: Sequence[T],
    *,
    concurrency: int,
) -> list[U | BaseException]:
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(item: T) -> U:
        async with sem:
            return await factory(client, item)

    tasks = [asyncio.create_task(_guarded(item)) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


def run_async_batch(
    factory: Callable[["httpx.AsyncClient", T], Awaitable[U]],
    items: Sequence[T],
    *,
    concurrency: int,
    timeout: float,
) -> list[U | BaseException]:
    """Fan out ``items`` concurrently over a shared ``AsyncClient``.

    Returns one entry per item in input order.  Failed coroutines are
    returned as exception instances (not raised) so adapters can decide
    whether to drop or re-raise per sample.

    Safe to call from synchronous code or from inside an already-running
    event loop (e.g. notebooks, async test harnesses): in the latter
    case the work is executed on a private loop in a worker thread so
    the caller's loop is not disturbed.
    """
    if not items:
        return []
    import httpx  # Imported lazily; optional dependency.

    concurrency = max(1, concurrency)

    async def _main() -> list[U | BaseException]:
        limits = httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        )
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
        ) as client:
            return await _run_all(
                client,
                factory,
                items,
                concurrency=concurrency,
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_main())

    # A loop is already running in this thread — execute on a private
    # loop in a worker thread so we don't interfere with the caller.
    def _thread_target() -> list[U | BaseException]:
        return asyncio.run(_main())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_thread_target).result()
