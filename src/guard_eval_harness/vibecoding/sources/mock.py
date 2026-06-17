"""In-memory mock task source for conformance testing.

Produces deterministic ``VibeTask`` records without touching any upstream
dataset, Docker, or network. Used by the conformance harness and as a smoke
target for the runner/CLI before real adapters land.
"""

from __future__ import annotations

from pathlib import Path

from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Cycled across generated tasks so per-CWE breakdowns have >1 bucket.
_MOCK_CWES = ["CWE-22", "CWE-79", "CWE-89", "CWE-787"]

# Default number of tasks emitted when no limit is supplied.
_DEFAULT_COUNT = 4


@task_source_registry.register("mock")
class MockTaskSource(TaskSource):
    """Yield N deterministic ``repo_patch`` tasks scored by the mock oracle.

    Splits:

    - ``None`` / ``"default"``: plain tasks.
    - ``"null_verdicts"``: tasks whose oracle returns ``None`` verdicts.
    - ``"infra"``: tasks whose oracle reports an infra failure.

    The split is recorded in ``environment`` only indirectly; the mock oracle
    derives its per-task outcome from ``AgentArtifact.metadata`` or from the
    artifact hash, so the source just stamps a deterministic outcome hint into
    each task id suffix consumed by fixtures/tests.
    """

    name = "mock"

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[VibeTask]:
        """Return up to ``limit`` deterministic mock tasks (no upstream)."""
        count = _DEFAULT_COUNT if limit is None else max(0, int(limit))
        tasks: list[VibeTask] = []
        for index in range(count):
            cwe = _MOCK_CWES[index % len(_MOCK_CWES)]
            task_id = self._task_id(split, index)
            tasks.append(
                VibeTask(
                    id=task_id,
                    source_dataset="mock",
                    task_type="repo_patch",
                    instructions=(
                        f"Mock task {index}: patch the repository to "
                        f"remediate {cwe}."
                    ),
                    repo=RepoSpec(
                        url="https://example.invalid/mock",
                        base_commit="0" * 40,
                        workdir=".",
                    ),
                    labels=TaskLabels(cwe=[cwe]),
                    environment=TaskEnvironmentRef(
                        oracle="mock",
                        requires_docker=False,
                    ),
                )
            )
        return tasks

    @staticmethod
    def _task_id(split: str | None, index: int) -> str:
        """Build a stable ``mock/<split>-<index>`` task id."""
        bucket = "default" if split is None else split
        return f"mock/{bucket}-{index}"
