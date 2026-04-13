"""Shared async dispatch helpers for API model adapters."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Awaitable, Callable, Sequence, TypeVar

import httpx

T = TypeVar("T")
U = TypeVar("U")


async def _run_all(
    client: httpx.AsyncClient,
    factory: Callable[[httpx.AsyncClient, T], Awaitable[U]],
    items: Sequence[T],
) -> list[U | BaseException]:
    # Concurrency is gated by the ``httpx.Limits`` on ``client``:
    # ``client.post()`` awaits a connection slot from the pool before
    # sending, so only ``max_connections`` requests are in flight at
    # once. Crucially, the connection slot is released as soon as the
    # response body is read (which happens inside the non-streaming
    # ``post`` call), *before* any retry/backoff ``asyncio.sleep``
    # runs. That means a task waiting in backoff is not holding a
    # permit, and queued tasks can start immediately — the property
    # an explicit ``asyncio.Semaphore`` wrapped around the whole
    # factory call would break.
    tasks = [
        asyncio.create_task(factory(client, item)) for item in items
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def run_async_batch(
    factory: Callable[[httpx.AsyncClient, T], Awaitable[U]],
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
            return await _run_all(client, factory, items)

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
