"""SecureVibeBench task source (HuggingFace).

Loads the 105 ARVO-derived tasks from the upstream Hugging Face dataset
(``iCSawyer/SecureVibeBench``), whose rows carry ``localid``, ``repo_url``,
``vic`` (vulnerability-introducing commit), ``repo_cwd`` (the repo path inside
the ARVO container), and ``description`` (the task requirement). Each task is a
``repo_patch`` scored by the ``securevibebench`` oracle, which runs the ARVO
Docker image (``n132/arvo:<id>-vul``) plus the upstream PoV + functional
harness.

Upstream moved the dataset to Hugging Face and dropped the in-repo
``data/full_dataset.zip``, so tasks load from HF rather than a checked-out
``data/`` directory -- the same "pin upstream + use it directly" pattern the
other vibecoding datasets follow. CWE/CVE labels are not part of the dataset, so
the label lists stay empty.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Upstream-published dataset (105 tasks). Tracks the published benchmark.
_HF_DATASET = "iCSawyer/SecureVibeBench"
_HF_SPLIT = "train"


@task_source_registry.register("securevibebench")
class SecureVibeBenchTaskSource(TaskSource):
    """Yield ``repo_patch`` tasks from the SecureVibeBench HF dataset."""

    name = "securevibebench"

    def __init__(
        self,
        *,
        rows: Iterable[dict[str, Any]] | None = None,
        dataset: str | None = None,
    ) -> None:
        """Optionally inject ``rows`` (tests; avoids a HF fetch) or override the
        dataset id."""
        self._rows = list(rows) if rows is not None else None
        self._dataset = dataset or _HF_DATASET

    def _load_rows(self) -> list[dict[str, Any]]:
        """Injected rows, else the HF dataset rows."""
        if self._rows is not None:
            return self._rows
        from datasets import load_dataset

        ds = load_dataset(self._dataset, split=_HF_SPLIT)
        return [dict(row) for row in ds]

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | None = None,
    ) -> list[VibeTask]:
        """Return normalized tasks for each dataset row."""
        tasks: list[VibeTask] = []
        for row in self._load_rows():
            task = self._task_from_row(row)
            if task is not None:
                tasks.append(task)
            if limit is not None and len(tasks) >= max(0, int(limit)):
                break
        return tasks

    def _task_from_row(self, row: dict[str, Any]) -> VibeTask | None:
        """Map one dataset row into a ``VibeTask`` (None if it lacks an id)."""
        localid = row.get("localid")
        if localid is None or str(localid).strip() == "":
            return None
        repo_cwd = row.get("repo_cwd")
        # The oracle carries the ARVO-specific fields structurally:
        #   - localid -> the task id suffix (``securevibebench/<localid>``)
        #   - vic     -> ``repo.base_commit`` (vuln-introducing commit); the
        #                oracle recomputes PVIC (``vic^``) at evaluation time
        #   - repo_cwd -> ``repo.workdir`` (the repo path inside the container)
        return VibeTask(
            id=f"securevibebench/{localid}",
            source_dataset="securevibebench",
            task_type="repo_patch",
            instructions=str(row.get("description") or ""),
            repo=RepoSpec(
                url=row.get("repo_url"),
                base_commit=row.get("vic"),
                workdir=str(repo_cwd) if repo_cwd else ".",
            ),
            labels=TaskLabels(cwe=[], cve=[]),
            environment=TaskEnvironmentRef(
                oracle="securevibebench",
                requires_docker=True,
            ),
        )
