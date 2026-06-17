"""Reusable conformance checks for vibecoding oracle adapters.

Every registered oracle should pass the same eight adapter-behavior checks
(load / stage / parse / capabilities / parallelism / infra). The checks are
plain functions of ``(adapter, fixtures_dir)`` so Stage-D adapters can plug in
later by dropping fixtures under ``tests/fixtures/vibecoding/<name>/`` and
parametrizing over them. The checks deliberately exercise *adapter* behavior,
not metric math.

Fixture layout (per adapter name):

    tests/fixtures/vibecoding/<name>/
        raw/results.json     # a scoreable upstream output the parser maps
        infra/results.json   # an infra-failure upstream output
        tasks/tasks.jsonl    # mini task metadata (one VibeTask per line)

For the in-process mock oracle the env provider computes ``raw/`` on the fly,
but the parser is also exercised against the committed fixtures so the same
checks transfer to out-of-process adapters.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    task_source_registry,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    VibeTask,
)

_PARALLELISM_MODELS = {
    "per_task_external",
    "batch_internal",
    "service_internal",
    "serial",
}


def _load_source_tasks(
    adapter: Any,
    *,
    limit: int | None = None,
) -> list[VibeTask]:
    """Resolve the task source registered under the adapter's name."""
    ensure_vibe_registrations()
    source_cls = task_source_registry.get(adapter.name)
    source = source_cls()
    return source.load(limit=limit)


def _artifact_for(task: VibeTask, **metadata: Any) -> AgentArtifact:
    """Build a minimal compatible ``patch`` artifact for a task."""
    return AgentArtifact(
        task_id=task.id,
        model="mock-model",
        kind="patch",
        patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+ok\n",
        base_commit=task.repo.base_commit,
        metadata=dict(metadata),
    )


@contextmanager
def _scratch_run_dir() -> Iterator[Path]:
    """Yield a throwaway run dir so checks never write under fixtures/."""
    with tempfile.TemporaryDirectory(prefix="vibe-conformance-") as tmp:
        yield Path(tmp)


def _read_fixture_task_ids(results_path: Path) -> list[str]:
    """Pull task ids from a committed results.json fixture."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return list(payload.keys())


def _raw_from_fixture(adapter: Any, fixture_dir: Path) -> RawOracleResult:
    """Build a ``RawOracleResult`` pointing at a committed fixture dir."""
    results_path = fixture_dir / "results.json"
    return RawOracleResult(
        adapter_name=adapter.name,
        outputs_dir=str(fixture_dir),
        logs_dir=str(fixture_dir),
        exit_code=0,
        task_ids=_read_fixture_task_ids(results_path),
    )


# --- The eight conformance checks ---------------------------------------


def load_returns_valid_task(adapter: Any, fixtures_dir: Path) -> None:
    """(1) ``TaskSource.load(limit=1)`` returns a valid ``VibeTask``."""
    tasks = _load_source_tasks(adapter, limit=1)
    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, VibeTask)
    assert task.id
    assert task.source_dataset
    assert task.task_type in adapter.task_types


def stage_writes_expected_inputs(adapter: Any, fixtures_dir: Path) -> None:
    """(2) ``stage`` writes upstream inputs and records task ids."""
    tasks = _load_source_tasks(adapter, limit=2)
    artifacts = [_artifact_for(task) for task in tasks]
    with _scratch_run_dir() as run_dir:
        staged = adapter.stage(tasks, artifacts, run_dir)
        assert isinstance(staged, StagedOracleInput)
        assert staged.adapter_name == adapter.name
        assert set(staged.task_ids) == {task.id for task in tasks}
        inputs_dir = Path(staged.inputs_dir)
        assert inputs_dir.exists()
        assert any(inputs_dir.iterdir())


def parse_fixture_to_result(adapter: Any, fixtures_dir: Path) -> None:
    """(3) parser maps a fixture raw output into valid results."""
    raw = _raw_from_fixture(adapter, fixtures_dir / "raw")
    results = adapter.parse(raw)
    assert results
    for result in results:
        assert isinstance(result, VibeTaskResult)
        assert result.task_id
    statuses = {result.status for result in results}
    assert "completed" in statuses


def unsupported_artifact_errors(adapter: Any, fixtures_dir: Path) -> None:
    """(4) an unsupported artifact kind fails with a clear error."""
    tasks = _load_source_tasks(adapter, limit=1)
    unsupported_kind = next(
        kind
        for kind in ("repo_dir", "completion", "full_file", "staged_chain")
        if kind not in adapter.artifact_kinds
    )
    bad = _bad_artifact(tasks[0], unsupported_kind)
    with _scratch_run_dir() as run_dir:
        try:
            adapter.stage(tasks, [bad], run_dir)
        except UnsupportedArtifactError as exc:
            assert str(exc)
            return
    raise AssertionError(
        f"{adapter.name}: stage accepted unsupported kind "
        f"{unsupported_kind!r}"
    )


def doctor_reports_missing_env(adapter: Any, fixtures_dir: Path) -> None:
    """(5) ``doctor`` reports missing env clearly (tolerant if absent)."""
    doctor = getattr(adapter, "doctor", None)
    if not callable(doctor):
        # Mock and other inline adapters need no environment doctor.
        return
    report = doctor()
    assert report is not None


def capabilities_accurate(adapter: Any, fixtures_dir: Path) -> None:
    """(6) adapter exposes accurate capabilities + shapes."""
    assert isinstance(adapter.capabilities, OracleCapabilities)
    assert isinstance(adapter.artifact_kinds, set)
    assert adapter.artifact_kinds
    assert isinstance(adapter.task_types, set)
    assert adapter.task_types
    assert adapter.granularity in {"per_task", "batch"}
    assert isinstance(adapter.parallelism, OracleParallelism)
    assert adapter.parallelism.model in _PARALLELISM_MODELS


def parallelism_honors_budget(adapter: Any, fixtures_dir: Path) -> None:
    """(7) a small resource budget is honored end-to-end."""
    from guard_eval_harness.vibecoding.oracles.mock import InMemoryEnvProvider

    tasks = _load_source_tasks(adapter, limit=2)
    artifacts = [_artifact_for(task) for task in tasks]
    with _scratch_run_dir() as run_dir:
        staged = adapter.stage(tasks, artifacts, run_dir)
        budget = ResourceBudget(max_workers=1, cpu_cores=1, memory_gb=1.0)
        run_config = OracleRunConfig(
            run_id="conformance",
            run_dir=str(run_dir),
        )
        raw = adapter.evaluate(
            staged, run_config, budget, InMemoryEnvProvider()
        )
        assert isinstance(raw, RawOracleResult)
        workers = raw.metadata.get("workers")
        if workers is not None:
            assert workers <= budget.max_workers
        results = adapter.parse(raw)
        assert len(results) == len(tasks)


def infra_failure_not_model_failure(
    adapter: Any, fixtures_dir: Path
) -> None:
    """(8) upstream infra failure parses to ``infra_failure``."""
    raw = _raw_from_fixture(adapter, fixtures_dir / "infra")
    results = adapter.parse(raw)
    assert results
    statuses = {result.status for result in results}
    assert "infra_failure" in statuses
    for result in results:
        if result.status == "infra_failure":
            assert result.failure_origin == "infra"


def _bad_artifact(task: VibeTask, kind: str) -> AgentArtifact:
    """Construct a valid-but-unsupported artifact of the given kind."""
    common = {"task_id": task.id, "model": "mock-model", "kind": kind}
    if kind == "repo_dir":
        return AgentArtifact(**common, worktree="/tmp/mock-worktree")
    if kind == "completion":
        return AgentArtifact(**common, completion="print('x')\n")
    if kind == "full_file":
        return AgentArtifact(**common, files={"a.py": "x = 1\n"})
    return AgentArtifact(**common, metadata={"stages": ["one"]})


CONFORMANCE_CHECKS = [
    load_returns_valid_task,
    stage_writes_expected_inputs,
    parse_fixture_to_result,
    unsupported_artifact_errors,
    doctor_reports_missing_env,
    capabilities_accurate,
    parallelism_honors_budget,
    infra_failure_not_model_failure,
]
