"""Base class for oracle adapters.

An oracle adapter is a thin wrapper around one upstream secure-coding
benchmark. It declares its capabilities/parallelism statically and implements
``stage`` / ``evaluate`` / ``parse``. It never spawns processes itself: the
runner injects an :class:`EnvProvider` into ``evaluate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import (
    EnvProvider,
    GenerationSpec,
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    VibeTask,
)

Granularity = Literal["per_task", "batch"]


class OracleAdapter(ABC):
    """Stages artifacts, calls upstream evaluation, parses outputs."""

    name: ClassVar[str]
    env: ClassVar[EnvSpec]
    artifact_kinds: ClassVar[set[str]]
    task_types: ClassVar[set[str]]
    granularity: ClassVar[Granularity]
    capabilities: ClassVar[OracleCapabilities]
    parallelism: ClassVar[OracleParallelism]
    parser_version: ClassVar[str]

    @abstractmethod
    def stage(
        self,
        tasks: list[VibeTask],
        artifacts: list[AgentArtifact],
        run_dir: Path,
    ) -> StagedOracleInput:
        """Write upstream-compatible input files for ``tasks``."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
        env_provider: EnvProvider,
    ) -> RawOracleResult:
        """Run the upstream evaluator via the injected env provider."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: RawOracleResult) -> list[VibeTaskResult]:
        """Normalize raw upstream output into result rows."""
        raise NotImplementedError

    # --- generation-side seam (mirror of stage/evaluate) --------------

    def generation_spec(
        self, task: VibeTask, cache_dir: str | None = None
    ) -> GenerationSpec:
        """How a live agent should frame generation for ``task``.

        The generation-side mirror of :meth:`stage`: the oracle declares what
        artifact kind its upstream scores (and optionally a tailored prompt /
        parser) so a dataset-agnostic driver emits exactly what this oracle can
        evaluate. ``cache_dir`` is the run's ``.geh`` cache (honored by oracles
        whose prompt/base resolution reads the acquired checkout/venv; ignored
        by the default). The default reproduces the legacy task-typed behavior
        -- a unified-diff ``patch``, or a ``completion`` for repo-completion
        tasks -- so an oracle that scores those kinds behaves as before.

        An oracle that scores only a richer kind (``full_file`` for the
        project-scaffold benchmarks, ``repo_dir`` for A.S.E) cannot be served by
        this default: those kinds need an oracle-specific prompt + parser (the
        engine cannot wrap a bare model body into a file map / worktree), so the
        default raises here rather than emit a ``patch`` every candidate would
        score as ``unsupported``. Such oracles override this method (see
        BaxBench / SecureVibeBench); until one does, ``geh vibe run`` on it fails
        loudly with guidance instead of silently producing zero scoreable rows.
        """
        kind = "completion" if task.task_type == "repo_completion" else "patch"
        if kind not in self.artifact_kinds:
            raise ValueError(
                f"live-agent generation is not supported for the {self.name!r} "
                f"oracle: it scores {sorted(self.artifact_kinds)} but the "
                f"default contract emits {kind!r}. Score externally-generated "
                f"predictions with `geh vibe eval --dataset {self.name} "
                f"--predictions <file>`, or add a GenerationSpec override "
                f"(prompt + parse) for this oracle."
            )
        return GenerationSpec(artifact_kind=kind)

    def prepare_acquisition(self, resolved: Any) -> None:
        """One-time post-acquire fixups after ``ensure_ready`` (no-op default).

        Run by ``geh vibe acquire`` once the upstream checkout + venv are in
        place (``resolved`` is the :class:`ResolvedEnv`). Oracles use it to
        materialize descriptors the task source globs (BaxBench
        ``scenario.json``)
        or to keep a recoverable copy of shared ground-truth files the scorer
        truncates (SecRepoBench ``assets/ids.txt``). Idempotent.
        """
        return None
