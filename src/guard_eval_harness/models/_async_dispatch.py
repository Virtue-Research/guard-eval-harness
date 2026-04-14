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
    *,
    concurrency: int,
) -> list[U | BaseException]:
    # Two layers of gating, both needed:
    #
    # 1. ``httpx.Limits(max_connections=concurrency)`` on the shared
    #    ``AsyncClient`` bounds the number of in-flight TCP
    #    connections, so only ``concurrency`` HTTP requests hit the
    #    server at once.
    #
    # 2. The ``sem`` semaphore below bounds the whole factory call —
    #    including the sync pre-``await`` work such as payload
    #    building, message templating, and (for multimodal samples)
    #    base64-encoding local image bytes. Without this, every task
    #    runs its factory until the first ``await`` immediately, so
    #    large batches build every payload in memory up-front before
    #    a single request leaves the client. That regresses the old
    #    ``ThreadPoolExecutor(max_workers=concurrency)`` behavior,
    #    which bounded the entire per-sample pipeline (build →
    #    POST → parse), not just the network leg.
    #
    # A task that enters backoff inside the factory keeps holding its
    # permit until its retry loop completes — matching the thread-pool
    # semantics where a retrying worker held its slot during
    # ``time.sleep``.
    sem = asyncio.Semaphore(concurrency)

    async def _gated(item: T) -> U:
        async with sem:
            return await factory(client, item)

    tasks = [asyncio.create_task(_gated(item)) for item in items]
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
        # ``pool=None`` disables the pool-acquire timeout so queued
        # requests wait indefinitely for a connection slot instead of
        # raising ``httpx.PoolTimeout`` when a batch is large relative
        # to ``concurrency`` or endpoint latency is high. Connect /
        # read / write still use the caller's ``timeout``, so a slow
        # *send* still fails fast — only the wait-in-line for a slot
        # is uncapped. This mirrors the old threadpool behavior where
        # queued jobs never timed out before starting.
        client_timeout = httpx.Timeout(timeout, pool=None)
        async with httpx.AsyncClient(
            timeout=client_timeout,
            limits=limits,
        ) as client:
            return await _run_all(
                client, factory, items, concurrency=concurrency
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
