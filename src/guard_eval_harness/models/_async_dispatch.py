"""Shared async dispatch helpers for API model adapters."""

from __future__ import annotations

import asyncio
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
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(item: T) -> U:
        async with sem:
            return await factory(client, item)

    tasks = [asyncio.create_task(_guarded(item)) for item in items]
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
            return await _run_all(
                client,
                factory,
                items,
                concurrency=concurrency,
            )

    return asyncio.run(_main())
