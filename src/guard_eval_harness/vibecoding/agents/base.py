"""Base contract for agent drivers in the VibeCoding Safety Bench.

An :class:`AgentDriver` turns a :class:`VibeTask` into an
:class:`AgentArtifact` plus run metadata. Drivers are *dataset-agnostic*:
they read only the normalized task fields (``instructions``, ``task_type``,
``repo``) and never branch on the upstream dataset (SusVibes / A.S.E /
SecRepoBench / ...). A driver may run a live model, contact an external
service, or simply replay preloaded predictions.

The runner wires ``live_agent`` mode separately; this module only defines the
ABC, the result bundle, and a tiny registry helper over ``agent_registry``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import Field

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import GenerationSpec
from guard_eval_harness.vibecoding.registry import (
    agent_registry,
    ensure_vibe_registrations,
)
from guard_eval_harness.vibecoding.schema import VibeModel, VibeTask


class AgentResult(VibeModel):
    """An :class:`AgentArtifact` plus optional per-generation metadata.

    The artifact is the load-bearing payload handed to oracles; everything
    else (token counts, cost, the resolved model name, log paths) is optional
    bookkeeping the runner may surface in provenance/reports.
    """

    artifact: AgentArtifact
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    log_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentDriver(ABC):
    """Produces an :class:`AgentResult` for a single task.

    Subclasses set a class-level ``name`` and implement :meth:`generate`.
    They must remain dataset-agnostic: read the normalized ``VibeTask`` only,
    never the source dataset's bespoke format.
    """

    name: str

    # Optional active cache dir, set by the runner before generation so a
    # driver can resolve upstream checkouts / clones under ``--cache-dir``
    # rather than a hard-coded default (mirrors the oracle's ``run_cache_dir``).
    run_cache_dir: str | Path | None = None

    @abstractmethod
    def generate(
        self,
        task: VibeTask,
        *,
        workdir: str | Path | None = None,
        model: str | None = None,
        gen_spec: GenerationSpec | None = None,
    ) -> AgentResult:
        """Generate a candidate artifact for ``task``.

        ``workdir`` is an optional checkout the driver may inspect (e.g. to
        snapshot repo files for context). ``model`` optionally overrides the
        driver's default model. ``gen_spec`` is the oracle's
        :class:`GenerationSpec` (artifact kind + optional prompt/parse); a
        model-backed driver threads it into the shared engine, while a replay
        driver (BYO) ignores it. Drivers stay dataset-agnostic: they apply the
        spec the oracle produced rather than branching on the dataset.
        Implementations should be robust to empty or unusable model output and
        return an artifact the oracle can record as an empty/null result rather
        than raising.
        """
        raise NotImplementedError


def get_agent_driver(name: str) -> AgentDriver:
    """Resolve and instantiate a registered agent driver by alias.

    Ensures builtin agent modules are imported first so registration side
    effects have run, then materializes the registered class.
    """
    ensure_vibe_registrations()
    driver_cls = agent_registry.get(name)
    return driver_cls()


__all__ = ["AgentResult", "AgentDriver", "get_agent_driver"]
