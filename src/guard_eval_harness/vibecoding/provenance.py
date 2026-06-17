"""Provenance model + builders for reproducible vibecoding results.

Every :class:`VibeTaskResult` should carry enough information to audit or
reproduce it: the GEH commit/version, adapter/parser versions, oracle
capabilities, upstream url/ref/command/workdir, environment + docker
fingerprints, the resource budget + worker count, trial coordinates, the
anti-cheat policy, redacted env vars, and the artifact/task sha256.

This module owns the richer ``Provenance`` record. The thinner
``ProvenanceBlock`` lives in ``results`` (the on-result field); this builder
also returns a ``ProvenanceBlock`` view so callers can attach it directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field

from guard_eval_harness.execution.artifacts import (
    sanitize_payload_for_artifacts,
)
from guard_eval_harness.vibecoding.results import ProvenanceBlock
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    ResourceBudget,
    VibeModel,
)


def git_commit(repo_root: str | Path) -> str | None:
    """Return ``git rev-parse HEAD`` for ``repo_root`` or ``None``.

    Never raises: a missing repo, missing git binary, or any subprocess
    error simply yields ``None`` so provenance capture is best-effort.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


class Provenance(VibeModel):
    """Full reproduction/audit record for a single result.

    A superset of the on-result :class:`ProvenanceBlock`; carries the
    oracle capabilities and trial coordinates that the block omits.
    """

    geh_commit: str | None = None
    geh_version: str | None = None
    adapter_name: str | None = None
    adapter_version: str | None = None
    parser_version: str | None = None
    oracle_capabilities: OracleCapabilities = Field(
        default_factory=OracleCapabilities
    )
    upstream_url: str | None = None
    upstream_ref: str | None = None
    upstream_command: list[str] = Field(default_factory=list)
    upstream_workdir: str | None = None
    env_fingerprint: dict[str, Any] = Field(default_factory=dict)
    docker_digests: dict[str, str] = Field(default_factory=dict)
    resource_budget: dict[str, Any] = Field(default_factory=dict)
    worker_count: int | None = None
    trial_index: int = Field(default=0, ge=0)
    trial_count: int = Field(default=1, ge=1)
    random_seed: int | None = None
    anti_cheat_policy_id: str | None = None
    anti_cheat_enforced: bool = False
    redacted_env: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: str | None = None
    task_sha256: str | None = None

    def to_block(self) -> ProvenanceBlock:
        """Project onto the slimmer on-result :class:`ProvenanceBlock`."""
        return ProvenanceBlock(
            geh_commit=self.geh_commit,
            geh_version=self.geh_version,
            adapter_name=self.adapter_name,
            parser_version=self.parser_version,
            upstream_url=self.upstream_url,
            upstream_ref=self.upstream_ref,
            upstream_command=list(self.upstream_command),
            upstream_workdir=self.upstream_workdir,
            env_fingerprint=dict(self.env_fingerprint),
            docker_digests=dict(self.docker_digests),
            resource_budget=dict(self.resource_budget),
            worker_count=self.worker_count,
            anti_cheat_policy_id=self.anti_cheat_policy_id,
            redacted_env=dict(self.redacted_env),
            artifact_sha256=self.artifact_sha256,
            task_sha256=self.task_sha256,
        )


def build_provenance(
    *,
    repo_root: str | Path | None = None,
    geh_version: str | None = None,
    adapter_name: str | None = None,
    adapter_version: str | None = None,
    parser_version: str | None = None,
    oracle_capabilities: OracleCapabilities | None = None,
    upstream_url: str | None = None,
    upstream_ref: str | None = None,
    upstream_command: list[str] | None = None,
    upstream_workdir: str | None = None,
    env_fingerprint: dict[str, Any] | None = None,
    docker_digests: dict[str, str] | None = None,
    resource_budget: ResourceBudget | dict[str, Any] | None = None,
    worker_count: int | None = None,
    trial_index: int = 0,
    trial_count: int = 1,
    random_seed: int | None = None,
    anti_cheat_policy_id: str | None = None,
    anti_cheat_enforced: bool = False,
    env: dict[str, str] | None = None,
    artifact_sha256: str | None = None,
    task_sha256: str | None = None,
) -> Provenance:
    """Assemble a :class:`Provenance` record.

    ``repo_root`` (when given) is used to capture the GEH commit. ``env`` is
    redacted before being recorded so secrets never reach artifacts.
    """
    if isinstance(resource_budget, ResourceBudget):
        budget_dump: dict[str, Any] = resource_budget.model_dump(mode="json")
    else:
        budget_dump = dict(resource_budget or {})

    redacted_env: dict[str, str] = {}
    if env:
        redacted_env = sanitize_payload_for_artifacts(dict(env))

    geh_commit = git_commit(repo_root) if repo_root is not None else None

    return Provenance(
        geh_commit=geh_commit,
        geh_version=geh_version,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        parser_version=parser_version,
        oracle_capabilities=oracle_capabilities or OracleCapabilities(),
        upstream_url=upstream_url,
        upstream_ref=upstream_ref,
        upstream_command=list(upstream_command or []),
        upstream_workdir=upstream_workdir,
        env_fingerprint=dict(env_fingerprint or {}),
        docker_digests=dict(docker_digests or {}),
        resource_budget=budget_dump,
        worker_count=worker_count,
        trial_index=trial_index,
        trial_count=trial_count,
        random_seed=random_seed,
        anti_cheat_policy_id=anti_cheat_policy_id,
        anti_cheat_enforced=anti_cheat_enforced,
        redacted_env=redacted_env,
        artifact_sha256=artifact_sha256,
        task_sha256=task_sha256,
    )
