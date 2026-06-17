"""Run-store layout, artifact materialization, and sha256 helpers.

The vibecoding run dir is laid out as::

    runs/vibecoding/<run_id>/
      run_config.yaml
      manifest.json
      tasks.jsonl
      artifacts/<safe_task_id>/{agent.patch,agent-files/,artifact.json}
      upstream/<adapter>/{inputs,outputs,logs}
      results.jsonl
      summary.json
      report.md

The classification ``ensure_run_layout`` is the wrong shape, so we rebuild
the layout here. All writes reuse the public helpers in
``execution/artifacts`` (atomic writes, deterministic JSON, redaction,
hashing); none of that is reimplemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from guard_eval_harness.execution.artifacts import (
    atomic_text_writer,
    dump_json,
    dump_jsonl,
    dump_model,
    sanitize_payload_for_artifacts,
    sha256_payload,
)
from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.safe_path import safe_relpath
from guard_eval_harness.vibecoding.schema import VibeModel, VibeTask


def safe_task_id(task_id: str) -> str:
    """Map a task id to a filesystem-safe directory name.

    Task ids look like ``susvibes/instance-1``; the slash would create an
    unexpected nested directory, so we flatten it to ``__``.
    """
    return task_id.replace("/", "__")


class ArtifactRefs(VibeModel):
    """Filesystem pointers to a materialized agent artifact."""

    artifact_dir: str
    artifact_json: str
    patch_path: str | None = None
    files_dir: str | None = None
    file_paths: dict[str, str] = Field(default_factory=dict)
    completion_path: str | None = None
    artifact_sha256: str


def ensure_vibe_run_layout(run_dir: str | Path) -> Path:
    """Create the stable vibecoding run directory tree.

    Creates the run root plus the ``artifacts/`` and ``upstream/`` subtrees.
    Returns the run root as a :class:`Path`.
    """
    root = Path(run_dir)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "upstream").mkdir(parents=True, exist_ok=True)
    return root


def task_artifact_dir(run_dir: str | Path, task_id: str) -> Path:
    """Directory holding one task's materialized agent artifact."""
    return Path(run_dir) / "artifacts" / safe_task_id(task_id)


def upstream_dir(run_dir: str | Path, adapter_name: str) -> Path:
    """Directory holding one adapter's upstream inputs/outputs/logs."""
    return Path(run_dir) / "upstream" / adapter_name


def write_run_config(run_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write ``run_config.yaml`` with sensitive values redacted.

    We keep dependencies stdlib-only, so the YAML body is JSON (which is
    valid YAML) rather than pulling in PyYAML.
    """
    sanitized = sanitize_payload_for_artifacts(config)
    path = Path(run_dir) / "run_config.yaml"
    with atomic_text_writer(path) as handle:
        handle.write(json.dumps(sanitized, indent=2, sort_keys=True))
        handle.write("\n")
    return path


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` with sensitive values redacted."""
    sanitized = sanitize_payload_for_artifacts(manifest)
    path = Path(run_dir) / "manifest.json"
    dump_json(path, sanitized)
    return path


def write_tasks(
    run_dir: str | Path, tasks: Iterable[VibeTask]
) -> Path:
    """Write ``tasks.jsonl`` (one task per line)."""
    path = Path(run_dir) / "tasks.jsonl"
    dump_jsonl(path, (t.model_dump(mode="json") for t in tasks))
    return path


def write_results(
    run_dir: str | Path, results: Iterable[VibeTaskResult]
) -> Path:
    """Write ``results.jsonl`` (one result per line)."""
    path = Path(run_dir) / "results.jsonl"
    dump_jsonl(path, (r.model_dump(mode="json") for r in results))
    return path


def append_result(
    run_dir: str | Path, result: VibeTaskResult
) -> Path:
    """Append one result row to ``results.jsonl``.

    Used for incremental writing while a run is in progress. Reads the
    existing rows and rewrites atomically so a crash leaves a valid file.
    """
    path = Path(run_dir) / "results.jsonl"
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.append(result.model_dump(mode="json"))
    dump_jsonl(path, rows)
    return path


def _candidate_dir(artifact: AgentArtifact) -> str:
    """Per-candidate subdir so duplicate task/model artifacts never clash.

    Distinct candidate artifacts for the same task get distinct dirs (keyed
    by model + content hash); re-submitting the same artifact is idempotent.
    """
    model = safe_task_id(str(artifact.model or "none"))
    return f"{model}-{compute_artifact_sha256(artifact)[:12]}"


def write_agent_artifact(
    run_dir: str | Path, artifact: AgentArtifact
) -> ArtifactRefs:
    """Materialize an :class:`AgentArtifact` under ``artifacts/<task>/``.

    Writes ``artifact.json`` always; a ``patch`` payload goes to
    ``agent.patch`` and ``files`` go under ``agent-files/``. Returns the
    filesystem pointers plus the content sha256.

    The artifact lives under ``artifacts/<safe_task_id>/<candidate>/`` so
    multiple candidates for the same task (pass@k / trials / model
    comparisons) never overwrite each other; ``<candidate>`` is keyed by
    model + content hash.
    """
    art_dir = (
        task_artifact_dir(run_dir, artifact.task_id)
        / _candidate_dir(artifact)
    )
    art_dir.mkdir(parents=True, exist_ok=True)

    patch_path: str | None = None
    files_dir: str | None = None
    file_paths: dict[str, str] = {}
    completion_path: str | None = None

    if artifact.patch is not None:
        target = art_dir / "agent.patch"
        with atomic_text_writer(target) as handle:
            handle.write(artifact.patch)
        patch_path = str(target)

    if artifact.files:
        files_root = (art_dir / "agent-files").resolve()
        files_root.mkdir(parents=True, exist_ok=True)
        files_dir = str(files_root)
        for rel_path, content in artifact.files.items():
            # Prediction artifacts are external/BYO input: confine every file
            # key to agent-files/ (rejects ``..``, absolute, symlink escapes).
            target = safe_relpath(files_root, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with atomic_text_writer(target) as handle:
                handle.write(content)
            file_paths[rel_path] = str(target)

    if artifact.completion is not None:
        target = art_dir / "completion.txt"
        with atomic_text_writer(target) as handle:
            handle.write(artifact.completion)
        completion_path = str(target)

    artifact_json = art_dir / "artifact.json"
    dump_model(artifact_json, artifact)

    return ArtifactRefs(
        artifact_dir=str(art_dir),
        artifact_json=str(artifact_json),
        patch_path=patch_path,
        files_dir=files_dir,
        file_paths=file_paths,
        completion_path=completion_path,
        artifact_sha256=compute_artifact_sha256(artifact),
    )


def compute_artifact_sha256(artifact: AgentArtifact) -> str:
    """Canonical content fingerprint of an artifact.

    Hashes only the load-bearing payload (kind, patch, sorted files,
    completion, base_commit) and deliberately excludes ``metadata`` and
    ``worktree`` so the digest depends on what the oracle actually sees,
    not on bookkeeping fields. The result is therefore stable across
    metadata changes.
    """
    canonical = {
        "kind": artifact.kind,
        "patch": artifact.patch,
        "files": {k: artifact.files[k] for k in sorted(artifact.files)},
        "completion": artifact.completion,
        "base_commit": artifact.base_commit,
    }
    return sha256_payload(canonical)


def compute_task_sha256(task: VibeTask) -> str:
    """Canonical fingerprint of a task's metadata."""
    return sha256_payload(task.model_dump(mode="json"))
