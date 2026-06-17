"""Core task/environment schema for the VibeCoding Safety Bench.

Every model here subclasses the local :class:`VibeModel`, which is a thin
``extra="forbid"`` wrapper over the shared :class:`HarnessModel`. These are
the foundational contracts that every other vibecoding module imports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from guard_eval_harness.schemas.core import HarnessModel

# --- Literal enums (verbatim value sets from the architecture spec) ---

TaskType = Literal[
    "repo_patch",
    "repo_completion",
    "project_scaffold",
    "repo_dir",
    "compositional_chain",
    "security_engineering",
]

ParallelismModel = Literal[
    "per_task_external",
    "batch_internal",
    "service_internal",
    "serial",
]

LicensePolicy = Literal[
    "vendor_allowed",
    "external_only",
    "unknown",
]

EnvKind = Literal["venv", "conda", "container", "inline"]


class VibeModel(HarnessModel):
    """Common base for vibecoding contracts (forbids unknown keys)."""

    model_config = ConfigDict(extra="forbid")


class RepoSpec(VibeModel):
    """Where a task's repository lives and at which commit."""

    url: str | None = None
    base_commit: str | None = None
    workdir: str = "."


class TaskLabels(VibeModel):
    """Security labels attached to a task."""

    cwe: list[str] = Field(default_factory=list)
    cve: list[str] = Field(default_factory=list)


class TaskEnvironmentRef(VibeModel):
    """Pointer from a task to the oracle/environment that scores it."""

    oracle: str = Field(min_length=1)
    requires_docker: bool = False


class ResourceEstimate(VibeModel):
    """Per-worker resource estimate declared by an environment."""

    cpu_per_worker: int = Field(default=1, ge=1)
    memory_gb_per_worker: float = Field(default=1.0, ge=0.0)
    disk_gb_per_worker: float = Field(default=1.0, ge=0.0)
    gpu_required: bool = False


class OracleParallelism(VibeModel):
    """How an oracle parallelizes evaluation across tasks."""

    model: ParallelismModel = "serial"
    default_workers: int = Field(default=1, ge=1)
    max_workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_worker_bounds(self) -> "OracleParallelism":
        if self.max_workers < self.default_workers:
            raise ValueError(
                "max_workers must be >= default_workers "
                f"(got max_workers={self.max_workers}, "
                f"default_workers={self.default_workers})"
            )
        return self


class OracleCapabilities(VibeModel):
    """What an oracle can actually measure.

    Used both as a static adapter declaration and as a per-result record so
    leaderboards only average rows with compatible capability sets.
    """

    runs_functional_tests: bool = False
    detects_target_vuln: bool = False
    detects_new_vuln: bool = False
    dynamic_pov: bool = False
    static_analysis: bool = False
    fuzzing: bool = False
    llm_judge: bool = False
    deterministic: bool = True


class EnvSpec(VibeModel):
    """How an upstream benchmark environment is acquired and invoked."""

    name: str = Field(min_length=1)
    kind: EnvKind = "venv"
    upstream_url: str | None = None
    upstream_ref: str | None = None
    root: str | None = None
    workdir: str | None = None
    python: str | None = None
    install: list[str] = Field(default_factory=list)
    requires_docker: bool = False
    requires_network_for_eval: bool = False
    disk_gb_estimate: float = Field(default=0.0, ge=0.0)
    resource_estimate: ResourceEstimate = Field(
        default_factory=ResourceEstimate
    )
    parallelism: OracleParallelism = Field(default_factory=OracleParallelism)
    license_policy: LicensePolicy = "unknown"
    env: dict[str, str] = Field(default_factory=dict)


class ResourceBudget(VibeModel):
    """Evaluation budget the runner derives from host limits + estimates."""

    max_workers: int = Field(default=1, ge=1)
    cpu_cores: int = Field(default=1, ge=1)
    memory_gb: float = Field(default=1.0, ge=0.0)
    disk_gb: float = Field(default=1.0, ge=0.0)
    docker_containers: int = Field(default=1, ge=0)


class VibeTask(VibeModel):
    """Normalized benchmark task metadata."""

    id: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    task_type: TaskType = "repo_patch"
    instructions: str = ""
    repo: RepoSpec = Field(default_factory=RepoSpec)
    labels: TaskLabels = Field(default_factory=TaskLabels)
    environment: TaskEnvironmentRef | None = None

    @model_validator(mode="after")
    def _check_non_empty_ids(self) -> "VibeTask":
        if not self.id.strip():
            raise ValueError("VibeTask.id must be non-empty")
        if not self.source_dataset.strip():
            raise ValueError("VibeTask.source_dataset must be non-empty")
        return self
