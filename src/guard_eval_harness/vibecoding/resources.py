"""Resource budget computation from host limits + EnvSpec estimates.

The runner derives a :class:`ResourceBudget` once and passes it to each oracle
adapter, which translates it into upstream worker flags (e.g. ``--max_workers``)
so GEH and the upstream harness do not double-schedule heavy Docker workloads.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from guard_eval_harness.vibecoding.schema import EnvSpec, ResourceBudget


def _host_cpu_cores() -> int:
    """Best-effort CPU core count (>= 1)."""
    return max(1, os.cpu_count() or 1)


def _host_disk_gb(path: str | Path) -> float:
    """Free disk in GiB at ``path`` (falls back to its first existing parent)."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return 0.0
    return usage.free / (1024.0**3)


def compute_resource_budget(
    env_spec: EnvSpec,
    *,
    max_workers_override: int | None = None,
    cache_dir: str | Path | None = None,
) -> ResourceBudget:
    """Derive an evaluation budget clamped to host limits + per-worker needs.

    The number of workers is bounded by:

    - the adapter's declared ``parallelism.max_workers`` (and the default),
    - how many per-worker CPU slices fit on the host,
    - how many per-worker disk slices fit in free disk,
    - an explicit ``max_workers_override`` when provided.

    The returned ``cpu_cores`` / ``disk_gb`` always reflect actual host limits
    so a budget never claims resources the host does not have.
    """
    parallelism = env_spec.parallelism
    estimate = env_spec.resource_estimate

    host_cores = _host_cpu_cores()
    free_disk_gb = _host_disk_gb(cache_dir or Path.cwd())

    cpu_per_worker = max(1, estimate.cpu_per_worker)
    cpu_capacity = max(1, host_cores // cpu_per_worker)

    disk_per_worker = estimate.disk_gb_per_worker
    if disk_per_worker > 0:
        disk_capacity = max(0, int(free_disk_gb // disk_per_worker))
    else:
        disk_capacity = parallelism.max_workers

    workers = parallelism.default_workers
    workers = min(workers, parallelism.max_workers)
    workers = min(workers, cpu_capacity)
    if disk_capacity > 0:
        workers = min(workers, disk_capacity)
    if max_workers_override is not None:
        workers = min(workers, max(1, max_workers_override))
    workers = max(1, workers)

    memory_gb = estimate.memory_gb_per_worker * workers

    docker_containers = workers if env_spec.requires_docker else 0

    return ResourceBudget(
        max_workers=workers,
        cpu_cores=host_cores,
        memory_gb=memory_gb,
        disk_gb=free_disk_gb,
        docker_containers=docker_containers,
    )


__all__ = ["compute_resource_budget"]
