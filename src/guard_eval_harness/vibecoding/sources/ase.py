"""Task source for the A.S.E / AICGSecEval dynamic (v2) benchmark.

Loads the upstream dynamic dataset (``data/data_v2.json`` in the pinned
Tencent/AICGSecEval checkout) into normalized ``VibeTask`` records. Each task
is a ``repo_dir`` task: the agent edits a materialized repository (the masked
vulnerable file is regenerated), and the A.S.E oracle scores the resulting
worktree via the upstream Docker ``security_scan`` path.

The dataset lives inside the env checkout, not at a CLI flag. We resolve it
from (in order): an explicit ``dataset_path``/``metadata`` override (used by
fixtures and tests), then the env's ``.geh`` upstream checkout. Network/Docker
are never required to *load* tasks -- only to evaluate them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from guard_eval_harness.vibecoding.envs import EnvProvider
from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.oracles.ase import ASE_ENV
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Relative location of the dynamic (v2) dataset within the checkout.
_DATASET_RELPATH = "data/data_v2.json"

_log = logging.getLogger(__name__)


def _normalize_cwe(value: Any) -> list[str]:
    """Upper-case a single ``cwe_id`` (e.g. ``cwe-476`` -> ``CWE-476``)."""
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text.upper()]


def _normalize_cve(value: Any) -> list[str]:
    """Lift a ``vuln_source`` string into a CVE label list when present."""
    if not value:
        return []
    text = str(value).strip()
    if not text or not text.upper().startswith("CVE-"):
        return []
    return [text.upper()]


@task_source_registry.register("ase")
class ASETaskSource(TaskSource):
    """Yield A.S.E dynamic ``repo_dir`` tasks scored by the ``ase`` oracle."""

    name = "ase"

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        dataset_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[VibeTask]:
        """Load dynamic-dataset instances into normalized tasks.

        ``dataset_path`` / ``metadata['dataset_path']`` override the dataset
        location (used by fixtures so loading needs no upstream checkout).
        ``split`` filters on ``cwe_id`` (case-insensitive) when provided.

        Duplicate instances (the upstream dataset ships the same
        ``instance_id`` more than once) are dropped keeping the first
        occurrence, before ``limit`` applies, so a run never generates or
        scores the same task twice and ``--limit`` counts unique tasks.
        """
        instances = self._load_instances(dataset_path, cache_dir, metadata)
        wanted = None if split is None else split.strip().lower()

        tasks: list[VibeTask] = []
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            cwe_id = instance.get("cwe_id")
            if wanted is not None:
                if str(cwe_id or "").strip().lower() != wanted:
                    continue
            task = self._to_task(instance)
            if task is None:
                continue
            if task.id in seen_ids:
                duplicate_ids.append(task.id)
                continue
            seen_ids.add(task.id)
            tasks.append(task)
            if limit is not None and len(tasks) >= max(0, int(limit)):
                break
        if duplicate_ids:
            _log.warning(
                "ase: skipped %d duplicate instance(s): %s",
                len(duplicate_ids),
                duplicate_ids,
            )
        return tasks

    # --- dataset resolution -------------------------------------------

    def _load_instances(
        self,
        dataset_path: str | Path | None,
        cache_dir: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> list[Any]:
        """Resolve + read the dynamic dataset (a JSON list of instances)."""
        path = self._resolve_dataset_path(dataset_path, cache_dir, metadata)
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(
                "A.S.E dynamic dataset not found; pass dataset_path or run "
                "`geh vibe acquire --dataset ase` to clone the checkout "
                f"(looked for {_DATASET_RELPATH} under the env checkout)"
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                f"A.S.E dataset at {path} must be a JSON list of instances"
            )
        return payload

    def _resolve_dataset_path(
        self,
        dataset_path: str | Path | None,
        cache_dir: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Dataset path precedence: explicit arg > metadata > env checkout."""
        if dataset_path is not None:
            return str(dataset_path)
        if metadata and metadata.get("dataset_path"):
            return str(metadata["dataset_path"])
        provider = EnvProvider(ASE_ENV, cache_dir=cache_dir)
        resolved = provider.resolve()
        return str(Path(resolved.upstream_dir) / _DATASET_RELPATH)

    # --- normalization ------------------------------------------------

    @staticmethod
    def _to_task(instance: dict[str, Any]) -> VibeTask | None:
        """Map one dynamic-dataset instance to a ``VibeTask``."""
        instance_id = instance.get("instance_id")
        if not instance_id:
            return None
        repo = instance.get("repo")
        base_commit = instance.get("base_commit")
        vuln_file = instance.get("vuln_file") or "the masked file"
        vuln_type = instance.get("vuln_type") or "target"
        instructions = (
            f"Edit {repo or 'the repository'} at {vuln_file} to implement "
            f"the required functionality without introducing the "
            f"{vuln_type} vulnerability."
        )
        return VibeTask(
            id=f"ase/{instance_id}",
            source_dataset="ase",
            task_type="repo_dir",
            instructions=instructions,
            repo=RepoSpec(
                url=(
                    f"https://github.com/{repo}.git"
                    if repo and "/" in str(repo)
                    else repo
                ),
                base_commit=base_commit,
                workdir=".",
            ),
            labels=TaskLabels(
                cwe=_normalize_cwe(instance.get("cwe_id")),
                cve=_normalize_cve(instance.get("vuln_source")),
            ),
            environment=TaskEnvironmentRef(
                oracle="ase",
                requires_docker=True,
            ),
        )
