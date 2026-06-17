"""SusVibes task source.

Loads the upstream SusVibes dataset (``datasets/default/susvibes_dataset.jsonl``
in the pinned checkout) into normalized :class:`VibeTask` records. Each row is a
SWE-bench-style repo-patch task: the agent receives an issue-style
``problem_statement`` in a repository and must produce a patch that fixes the
functional requirement *and* the known target vulnerability (CWE/CVE).

The dataset path is not a CLI flag in upstream; GEH points at the checked-out
``datasets/default/susvibes_dataset.jsonl``. Tests pass a ``dataset_path``
override at a mini fixture jsonl so loading needs neither Docker nor the full
7MB dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from guard_eval_harness.vibecoding.envs import EnvProvider
from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Relative location of the dataset within an upstream SusVibes checkout.
_DATASET_RELPATH = "datasets/default/susvibes_dataset.jsonl"


def _github_url(project: str | None) -> str | None:
    """Build a GitHub clone URL from an upstream ``project`` slug."""
    if not project:
        return None
    project = project.strip().strip("/")
    if not project:
        return None
    return f"https://github.com/{project}"


@task_source_registry.register("susvibes")
class SusVibesTaskSource(TaskSource):
    """Load SusVibes repo-patch tasks into normalized ``VibeTask`` records.

    The default dataset location is resolved from the ``susvibes`` env's
    ``.geh`` checkout (``<upstream>/datasets/default/susvibes_dataset.jsonl``).
    Pass ``dataset_path`` to point at a mini fixture jsonl in tests.
    """

    name = "susvibes"

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        self._dataset_path = (
            Path(dataset_path) if dataset_path is not None else None
        )

    def _resolve_dataset_path(
        self, cache_dir: str | Path | None = None
    ) -> Path:
        """Resolve the dataset jsonl: explicit override or env checkout."""
        if self._dataset_path is not None:
            return self._dataset_path
        # Mirror the oracle's EnvSpec so both resolve the same checkout.
        from guard_eval_harness.vibecoding.oracles.susvibes import (
            SusVibesOracle,
        )

        provider = EnvProvider(SusVibesOracle.env, cache_dir=cache_dir)
        resolved = provider.resolve()
        return Path(resolved.upstream_dir) / _DATASET_RELPATH

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[VibeTask]:
        """Return up to ``limit`` SusVibes tasks (``split`` is advisory)."""
        path = self._resolve_dataset_path(cache_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"SusVibes dataset not found at {path}; run "
                "`geh vibe acquire --dataset susvibes` to clone the "
                "upstream checkout, or pass dataset_path=<mini.jsonl>."
            )
        tasks: list[VibeTask] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                tasks.append(self._to_task(record))
                if limit is not None and len(tasks) >= int(limit):
                    break
        return tasks

    def _to_task(self, record: dict) -> VibeTask:
        """Map one upstream dataset record to a normalized ``VibeTask``."""
        instance_id = record["instance_id"]
        cwe_ids = list(record.get("cwe_ids") or [])
        cve_id = record.get("cve_id")
        cves = [cve_id] if cve_id else []
        # Upstream extras (image_name / expected_failures / language) are
        # informational here: the upstream evaluator re-reads them from the
        # checked-out dataset + ``components.json`` keyed by instance_id, so
        # GEH does not need to forward them through the predictions file. The
        # normalized ``VibeTask`` is ``extra="forbid"`` and carries only the
        # fields its consumers (oracle stage + metrics) actually use.
        return VibeTask(
            id=f"susvibes/{instance_id}",
            source_dataset="susvibes",
            task_type="repo_patch",
            instructions=record.get("problem_statement", ""),
            repo=RepoSpec(
                url=_github_url(record.get("project")),
                base_commit=record.get("base_commit"),
                workdir=".",
            ),
            labels=TaskLabels(cwe=cwe_ids, cve=cves),
            environment=TaskEnvironmentRef(
                oracle="susvibes",
                requires_docker=True,
            ),
        )
