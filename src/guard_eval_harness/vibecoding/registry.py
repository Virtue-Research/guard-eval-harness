"""Registries for vibecoding task sources, oracles, and agents.

Mirrors :mod:`guard_eval_harness.registry.core`: three ``Registry[T]``
instances plus an idempotent ``ensure_vibe_registrations()`` that auto-imports
the builtin submodules and discovers entry points.
"""

from __future__ import annotations

from typing import Any

from guard_eval_harness.plugins.discovery import import_submodules
from guard_eval_harness.registry.core import Registry

task_source_registry: Registry[Any] = Registry("vibe_task_source")
oracle_registry: Registry[Any] = Registry("vibe_oracle")
agent_registry: Registry[Any] = Registry("vibe_agent")

_vibe_registered = False


def ensure_vibe_registrations() -> None:
    """Import builtin vibecoding modules + entry points once.

    Idempotent: safe to call repeatedly. The builtin submodule packages may be
    empty during early build stages; ``import_submodules`` tolerates that and
    skips ``base`` / private modules.
    """
    global _vibe_registered
    if _vibe_registered:
        return
    import_submodules("guard_eval_harness.vibecoding.oracles")
    import_submodules("guard_eval_harness.vibecoding.sources")
    import_submodules("guard_eval_harness.vibecoding.agents")
    oracle_registry.discover_entry_points(
        "guard_eval_harness.vibecoding.oracles"
    )
    task_source_registry.discover_entry_points(
        "guard_eval_harness.vibecoding.sources"
    )
    agent_registry.discover_entry_points(
        "guard_eval_harness.vibecoding.agents"
    )
    _vibe_registered = True
