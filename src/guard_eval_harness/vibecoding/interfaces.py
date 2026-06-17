"""Adapter interfaces and value models for the vibecoding subsystem.

Defines the seams the runner orchestrates: task loading, artifact validation
/ conversion, and out-of-process oracle execution. The runner injects an
:class:`EnvProvider` into adapters so adapters never spawn processes
themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.schema import (
    ResourceBudget,
    VibeModel,
    VibeTask,
)


class UnsupportedArtifactError(Exception):
    """Raised when an adapter cannot evaluate an artifact/task shape."""


class StagedOracleInput(VibeModel):
    """Upstream-format input written by an adapter's ``stage``."""

    adapter_name: str = Field(min_length=1)
    inputs_dir: str
    task_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OracleRunConfig(VibeModel):
    """Per-run knobs passed into an adapter's ``evaluate``."""

    run_id: str = Field(min_length=1)
    run_dir: str
    trial_index: int = Field(default=0, ge=0)
    random_seed: int | None = None
    no_cache: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class RawOracleResult(VibeModel):
    """Raw upstream output located on disk after ``evaluate``."""

    adapter_name: str = Field(min_length=1)
    outputs_dir: str
    logs_dir: str | None = None
    exit_code: int | None = None
    task_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class GenerationSpec:
    """How a live agent should frame generation for one oracle/task.

    Produced by :meth:`OracleAdapter.generation_spec` and consumed by the
    shared generation engine (:func:`agents._engine.generate_with`). It is the
    generation-side mirror of the scoring-side ``stage``/``evaluate`` seam: the
    oracle declares what artifact *kind* its upstream expects, and optionally
    how to frame the prompt and parse the model's raw text into that kind.

    The defaults reproduce the legacy task-typed behavior -- a unified-diff
    ``patch`` (or a ``completion`` for repo-completion tasks), built with the
    engine's generic prompt and fenced-block parser -- so an oracle that does
    not override generation behaves exactly as before.

    - ``artifact_kind``: the :class:`AgentArtifact` kind to emit
      (``patch`` / ``completion`` / ``full_file`` / ``repo_dir``). Must be one
      the oracle's ``artifact_kinds`` accepts.
    - ``prompt``: ``(task, repo_snapshot) -> (system, user)``. ``None`` uses the
      engine's task-typed prompt.
    - ``parse``: ``(task, model, raw_text) -> AgentArtifact | None``. Returns
      ``None`` for empty/garbled output (the engine emits an empty artifact).
      ``None`` uses the engine's fenced-block extraction + kind wrapping.
    """

    artifact_kind: str
    prompt: Callable[[VibeTask, str], tuple[str, str]] | None = None
    parse: Callable[[VibeTask, str, str], AgentArtifact | None] | None = None


class TaskSource(ABC):
    """Loads upstream metadata into normalized ``VibeTask`` records."""

    name: str

    @abstractmethod
    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[Any]:
        """Return normalized ``VibeTask`` records for this source.

        ``cache_dir`` lets the runner point a source's upstream-checkout
        resolution at the same ``.geh`` cache used for evaluation (e.g. for
        ``geh vibe eval --cache-dir``); ``None`` uses the default.
        """
        raise NotImplementedError


class ArtifactAdapter(ABC):
    """Validates and converts artifact shapes for one oracle."""

    @abstractmethod
    def validate(self, artifact: AgentArtifact) -> None:
        """Raise ``UnsupportedArtifactError`` for incompatible shapes."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, artifact: AgentArtifact) -> AgentArtifact:
        """Convert to the kind this oracle accepts (or raise)."""
        raise NotImplementedError


@runtime_checkable
class EnvProvider(Protocol):
    """Runs upstream commands out of process for an adapter."""

    def evaluate(
        self,
        env: Any,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
    ) -> RawOracleResult:
        """Execute the upstream evaluator and return raw output paths."""
        ...


__all__ = [
    "UnsupportedArtifactError",
    "StagedOracleInput",
    "OracleRunConfig",
    "RawOracleResult",
    "GenerationSpec",
    "TaskSource",
    "ArtifactAdapter",
    "EnvProvider",
]
