"""End-to-end tests for the vibecoding ``VibeRunner`` (Stage C1).

These exercise the integration join point with in-process mock source +
oracle adapters and the :class:`InMemoryEnvProvider`, so nothing touches
Docker, a venv, or the network:

- one normalized result row per task;
- the ``batch_internal`` path calls ``stage`` exactly once (spy);
- a ``repo_dir`` oracle against a patch-only artifact yields a single
  ``unsupported`` row;
- a ``mock_outcome=infra`` artifact yields ``infra_failure`` excluded from
  the model denominators;
- ``derive_task_metrics`` is consistent on every scored row;
- a 2nd identical run hits the cache and never re-invokes ``evaluate``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from guard_eval_harness.execution.artifacts import dump_json
from guard_eval_harness.vibecoding.artifacts import (
    AgentArtifact,
    artifact_sha256,
)
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.base import OracleAdapter
from guard_eval_harness.vibecoding.oracles.mock import (
    InMemoryEnvProvider,
    MockOracleAdapter,
)
from guard_eval_harness.vibecoding.agents.base import (
    AgentDriver,
    AgentResult,
)
from guard_eval_harness.vibecoding.registry import (
    agent_registry,
    ensure_vibe_registrations,
    oracle_registry,
    task_source_registry,
)
from guard_eval_harness.vibecoding.metrics import (
    compute_vibe_metrics,
    in_denominator,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.runner import VibeRunResult, VibeRunner
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)
from guard_eval_harness.vibecoding.sources.mock import MockTaskSource

# --- spy / variant adapters registered under unique names --------------

_STAGE_CALLS: dict[str, int] = {}
_EVALUATE_CALLS: dict[str, int] = {}
_AGENT_WORKDIRS: list[object] = []
_AGENT_GEN_SPECS: list[object] = []


def _reset_counters() -> None:
    _STAGE_CALLS.clear()
    _EVALUATE_CALLS.clear()
    _AGENT_WORKDIRS.clear()
    _AGENT_GEN_SPECS.clear()


class _CountingEnvProvider(InMemoryEnvProvider):
    """An :class:`InMemoryEnvProvider` that counts ``evaluate`` calls."""

    def evaluate(self, env, staged, run_config, resource_budget):
        _EVALUATE_CALLS[staged.adapter_name] = (
            _EVALUATE_CALLS.get(staged.adapter_name, 0) + 1
        )
        return super().evaluate(env, staged, run_config, resource_budget)


@oracle_registry.register("spy_batch")
class _SpyBatchOracle(MockOracleAdapter):
    """A batch oracle that records how often ``stage`` is called."""

    name = "spy_batch"
    env = EnvSpec(
        name="spy_batch",
        kind="inline",
        license_policy="vendor_allowed",
        parallelism=OracleParallelism(
            model="batch_internal", default_workers=1, max_workers=4
        ),
    )
    parser_version = "spy-batch-1"

    def stage(self, tasks, artifacts, run_dir):
        _STAGE_CALLS[self.name] = _STAGE_CALLS.get(self.name, 0) + 1
        return super().stage(tasks, artifacts, run_dir)


@task_source_registry.register("spy_batch")
class _SpyBatchSource(MockTaskSource):
    """Task source whose name matches the spy oracle."""

    name = "spy_batch"


@oracle_registry.register("repo_dir_only")
class _RepoDirOracle(OracleAdapter):
    """An oracle that only accepts ``repo_dir`` artifacts.

    Used to prove a patch-only artifact is rejected before staging and
    surfaces as exactly one ``unsupported`` row.
    """

    name = "repo_dir_only"
    env = EnvSpec(name="repo_dir_only", kind="inline")
    artifact_kinds = {"repo_dir"}
    task_types = {"repo_dir"}
    granularity = "batch"
    capabilities = OracleCapabilities(
        runs_functional_tests=True, detects_target_vuln=True
    )
    parallelism = OracleParallelism(
        model="batch_internal", default_workers=1, max_workers=2
    )
    parser_version = "repo-dir-1"

    def stage(self, tasks, artifacts, run_dir):  # pragma: no cover
        raise AssertionError("stage must not run for unsupported artifacts")

    def evaluate(  # pragma: no cover
        self, staged, run_config, resource_budget, env_provider
    ):
        raise AssertionError("evaluate must not run for unsupported")

    def parse(self, raw):  # pragma: no cover
        return []


@task_source_registry.register("repo_dir_only")
class _RepoDirSource(MockTaskSource):
    """Task source named to match the repo_dir-only oracle."""

    name = "repo_dir_only"

    def load(self, *, split=None, limit=None, cache_dir=None):
        return [
            VibeTask(
                id="repo_dir_only/task-0",
                source_dataset="repo_dir_only",
                task_type="repo_dir",
                repo=RepoSpec(base_commit="0" * 40),
                labels=TaskLabels(cwe=["CWE-22"]),
                environment=TaskEnvironmentRef(oracle="repo_dir_only"),
            )
        ]


@oracle_registry.register("per_artifact")
class _PerArtifactOracle(MockOracleAdapter):
    """A batch oracle that returns one row per artifact (duplicates kept).

    Models an upstream that scores every candidate distinctly, so the runner
    must preserve multiple candidates sharing a ``task_id`` rather than
    collapsing them.
    """

    name = "per_artifact"
    env = EnvSpec(
        name="per_artifact",
        kind="inline",
        license_policy="vendor_allowed",
        parallelism=OracleParallelism(
            model="batch_internal", default_workers=1, max_workers=4
        ),
    )
    parser_version = "per-artifact-1"

    def stage(self, tasks, artifacts, run_dir):
        inputs = Path(run_dir) / "upstream" / self.name / "inputs"
        ids = [a.task_id for a in artifacts]  # ordered, duplicates kept
        dump_json(inputs / "ids.json", {"task_ids": ids})
        return StagedOracleInput(
            adapter_name=self.name, inputs_dir=str(inputs), task_ids=ids
        )

    def evaluate(self, staged, run_config, resource_budget, env_provider):
        out = Path(staged.inputs_dir).parent / "outputs"
        ids = json.loads(
            (Path(staged.inputs_dir) / "ids.json").read_text("utf-8")
        )["task_ids"]
        dump_json(out / "rows.json", {"task_ids": ids})
        return RawOracleResult(
            adapter_name=self.name, outputs_dir=str(out), task_ids=ids
        )

    def parse(self, raw):
        ids = json.loads(
            (Path(raw.outputs_dir) / "rows.json").read_text("utf-8")
        )["task_ids"]
        return [
            VibeTaskResult(
                task_id=tid,
                source_dataset="per_artifact",
                model="mock-model",
                status="completed",
                functional_pass=True,
                security_oracle_pass=True,
                known_vuln_present=False,
            )
            for tid in ids
        ]


@task_source_registry.register("per_artifact")
class _PerArtifactSource(MockTaskSource):
    """Task source whose name matches the per-artifact oracle."""

    name = "per_artifact"


_LOADED_CACHE_DIRS: list = []


@oracle_registry.register("cachedir_spy")
class _CacheDirSpyOracle(MockOracleAdapter):
    """Mock oracle whose name matches the cache-dir spy source."""

    name = "cachedir_spy"
    env = EnvSpec(
        name="cachedir_spy",
        kind="inline",
        license_policy="vendor_allowed",
        parallelism=OracleParallelism(
            model="batch_internal", default_workers=1, max_workers=4
        ),
    )
    parser_version = "cachedir-spy-1"


@task_source_registry.register("cachedir_spy")
class _CacheDirSpySource(MockTaskSource):
    """Records the ``cache_dir`` the runner threads into ``load``."""

    name = "cachedir_spy"

    def load(self, *, split=None, limit=None, cache_dir=None):
        _LOADED_CACHE_DIRS.append(cache_dir)
        return super().load(split=split, limit=limit)


@oracle_registry.register("stage_reject")
class _StageRejectOracle(MockOracleAdapter):
    """A batch oracle whose ``stage`` rejects an artifact shape.

    Models an adapter (e.g. SecCodeBench's multi-file check) that only detects
    a shape problem inside ``stage()`` and raises ``UnsupportedArtifactError``.
    The runner must label this ``unsupported`` (adapter), never ``infra``.
    """

    name = "stage_reject"
    env = EnvSpec(
        name="stage_reject",
        kind="inline",
        license_policy="vendor_allowed",
        parallelism=OracleParallelism(
            model="batch_internal", default_workers=1, max_workers=4
        ),
    )
    parser_version = "stage-reject-1"

    def stage(self, tasks, artifacts, run_dir):
        raise UnsupportedArtifactError("simulated bad artifact shape")

    def evaluate(  # pragma: no cover - stage rejects first
        self, staged, run_config, resource_budget, env_provider
    ):
        raise AssertionError("evaluate must not run after a stage rejection")


@task_source_registry.register("stage_reject")
class _StageRejectSource(MockTaskSource):
    """Task source named to match the stage-rejecting oracle."""

    name = "stage_reject"


@agent_registry.register("mock_agent")
class _MockAgentDriver(AgentDriver):
    """Deterministic live driver: one patch artifact per task (no network)."""

    name = "mock_agent"

    def generate(self, task, *, workdir=None, model=None, gen_spec=None):
        _AGENT_WORKDIRS.append(workdir)
        _AGENT_GEN_SPECS.append(gen_spec)
        resolved = model or "mock-model"
        artifact = AgentArtifact(
            task_id=task.id,
            model=resolved,
            kind="patch",
            patch=f"--- a\n+++ b\n@@ -0,0 +1 @@\n+{task.id}\n",
        )
        return AgentResult(
            artifact=artifact, model=resolved, metadata={"generated": True}
        )


@agent_registry.register("telemetry_agent")
class _TelemetryAgentDriver(AgentDriver):
    """Live driver reporting non-trivial usage/cost telemetry per task.

    Mirrors a real API driver (e.g. ``ClaudeAgentDriver``) that packs token
    counts + cost into the :class:`AgentResult` *around* the artifact, so the
    runner is exercised on a result whose sibling fields must survive into the
    persisted artifact metadata rather than being dropped after generation.
    """

    name = "telemetry_agent"

    def generate(self, task, *, workdir=None, model=None, gen_spec=None):
        resolved = model or "telemetry-model"
        artifact = AgentArtifact(
            task_id=task.id,
            model=resolved,
            kind="patch",
            patch=f"--- a\n+++ b\n@@ -0,0 +1 @@\n+{task.id}\n",
            metadata={"orig": "kept"},
        )
        return AgentResult(
            artifact=artifact,
            model=resolved,
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            cost_usd=0.0023,
            metadata={"note": "driver-meta-survives"},
        )


# Set by the live-base test before a run; read by ``_LiveBaseSource.load`` to
# point the task at an on-disk repo the oracle's ``live_base`` resolves.
_LIVE_BASE_REPO: dict[str, str] = {}


@oracle_registry.register("live_base_spy")
class _LiveBaseOracle(MockOracleAdapter):
    """Mock oracle exposing a ``live_base`` hook.

    Stands in for SecureVibeBench's PVIC resolution: it hands the runner an
    explicit host-side ``(repo_dir, ref)`` so the live agent is seeded from the
    real repository instead of generating blind.
    """

    name = "live_base_spy"

    def live_base(self, task, cache_dir):
        url = task.repo.url
        ref = task.repo.base_commit
        if not url or not ref:
            return None
        return Path(url), ref


@task_source_registry.register("live_base_spy")
class _LiveBaseSource(MockTaskSource):
    """Source whose task carries a real on-disk repo URL + commit."""

    name = "live_base_spy"

    def load(self, *, split=None, limit=None, cache_dir=None):
        return [
            VibeTask(
                id="live_base_spy/task-0",
                source_dataset="live_base_spy",
                task_type="repo_patch",
                repo=RepoSpec(
                    url=_LIVE_BASE_REPO.get("url"),
                    base_commit=_LIVE_BASE_REPO.get("ref"),
                ),
                environment=TaskEnvironmentRef(oracle="live_base_spy"),
            )
        ]


def _make_git_repo(path: Path, content: str) -> str:
    """Init a git repo at ``path`` with one commit; return the commit sha."""
    path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True, capture_output=True, text=True, env=env,
        ).stdout.strip()

    _git("init", "-q")
    (path / "app.py").write_text(content, encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-qm", "init")
    return _git("rev-parse", "HEAD")


def _patch_artifact(task_id: str, **metadata) -> AgentArtifact:
    """A minimal compatible ``patch`` artifact for a task."""
    return AgentArtifact(
        task_id=task_id,
        model="mock-model",
        kind="patch",
        patch=f"--- a\n+++ b\n@@ -0,0 +1 @@\n+{task_id}\n",
        metadata=dict(metadata),
    )


class RunnerEndToEndTest(unittest.TestCase):
    """Happy-path: one result per task, consistent derived metrics."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_one_result_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            source = MockTaskSource()
            tasks = source.load(limit=3)
            preds = [_patch_artifact(t.id) for t in tasks]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=3,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertIsInstance(result, VibeRunResult)
            self.assertEqual(len(result.results), 3)
            self.assertEqual(
                {r.task_id for r in result.results},
                {t.id for t in tasks},
            )

    def test_run_dir_artifacts_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=2)
            preds = [_patch_artifact(t.id) for t in tasks]
            runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=2,
                env_provider=InMemoryEnvProvider(),
                run_dir=run_dir,
            )
            self.assertTrue((run_dir / "results.jsonl").exists())
            self.assertTrue((run_dir / "tasks.jsonl").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "artifacts").is_dir())

    def test_derive_task_metrics_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=4)
            preds = [
                _patch_artifact(tasks[0].id, mock_outcome="secure_pass"),
                _patch_artifact(tasks[1].id, mock_outcome="null_functional"),
                _patch_artifact(tasks[2].id, mock_outcome="model_failure"),
                _patch_artifact(tasks[3].id, mock_outcome="secure_pass"),
            ]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=4,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            by_id = {r.task_id: r for r in result.results}
            secure = by_id[tasks[0].id]
            self.assertIs(secure.target_secure_success, True)
            null_fn = by_id[tasks[1].id]
            self.assertIsNone(null_fn.target_secure_success)
            # strict_secure stays null (detects_new_vuln=False everywhere).
            for row in result.results:
                if row.status not in ("completed",):
                    continue
                self.assertIsNone(row.strict_secure_success)


class RunnerDuplicateArtifactTest(unittest.TestCase):
    """A second candidate for a task in one batch is preserved + flagged.

    These benchmark upstreams evaluate one candidate per task per run, so the
    surplus candidate is never silently dropped or mis-attributed: it scores
    the first candidate and routes the duplicate to a clearly-labeled
    ``unsupported`` row (run trials/pass@k as separate runs).
    """

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_duplicate_task_in_batch_flagged_not_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = _PerArtifactSource().load(limit=2)
            t0, t1 = tasks[0].id, tasks[1].id
            # Two candidate artifacts for t0 (distinct patches), plus one t1.
            preds = [
                AgentArtifact(
                    task_id=t0, model="m", kind="patch",
                    patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+candidate-A\n",
                ),
                AgentArtifact(
                    task_id=t0, model="m", kind="patch",
                    patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+candidate-B\n",
                ),
                _patch_artifact(t1),
            ]
            result = runner.run(
                "per_artifact",
                "per_artifact",
                predictions=preds,
                limit=2,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
                no_cache=True,
            )
            # 3 rows total: t0 scored once + t0 duplicate flagged + t1 scored.
            self.assertEqual(len(result.results), 3)
            t0_rows = [r for r in result.results if r.task_id == t0]
            self.assertEqual(len(t0_rows), 2)
            statuses = sorted(r.status for r in t0_rows)
            self.assertEqual(statuses, ["completed", "unsupported"])
            # The first candidate is the one that scored (no mis-attribution).
            scored = next(r for r in t0_rows if r.status == "completed")
            self.assertEqual(scored.failure_origin, "none")
            dup = next(r for r in t0_rows if r.status == "unsupported")
            self.assertEqual(dup.failure_reason, "unsupported_artifact")


class RunnerCacheDirTest(unittest.TestCase):
    """The run's cache dir is threaded into task loading."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()
        _LOADED_CACHE_DIRS.clear()

    def test_cache_dir_threaded_into_source_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "mycache"
            runner = VibeRunner(cache_dir=cache)
            preds = [_patch_artifact("mock/default-0")]
            runner.run(
                "cachedir_spy",
                "cachedir_spy",
                predictions=preds,
                limit=1,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
                no_cache=True,
            )
            self.assertEqual(len(_LOADED_CACHE_DIRS), 1)
            self.assertEqual(
                Path(_LOADED_CACHE_DIRS[0]).resolve(), cache.resolve()
            )


class RunnerZeroTaskTest(unittest.TestCase):
    """A source that loads zero tasks fails loudly (missing checkout)."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_zero_tasks_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            with self.assertRaises(ValueError):
                runner.run(
                    "mock",
                    "mock",
                    predictions=[_patch_artifact("mock/default-0")],
                    limit=0,  # mock yields 0 tasks
                    env_provider=InMemoryEnvProvider(),
                    run_dir=Path(tmp) / "run",
                )

    def test_zero_tasks_allowed_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            result = runner.run(
                "mock",
                "mock",
                predictions=[],
                limit=0,
                allow_empty=True,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(len(result.results), 0)


class RunnerBatchOnceTest(unittest.TestCase):
    """The batch path calls stage exactly once for the whole list."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_stage_called_once_for_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = _SpyBatchSource().load(limit=3)
            preds = [_patch_artifact(t.id) for t in tasks]
            runner.run(
                "spy_batch",
                "spy_batch",
                predictions=preds,
                limit=3,
                env_provider=_CountingEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(_STAGE_CALLS.get("spy_batch"), 1)
            self.assertEqual(_EVALUATE_CALLS.get("spy_batch"), 1)


class RunnerUnsupportedTest(unittest.TestCase):
    """A repo_dir oracle vs a patch-only artifact -> single unsupported."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_patch_artifact_unsupported_by_repo_dir_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            pred = _patch_artifact("repo_dir_only/task-0")
            result = runner.run(
                "repo_dir_only",
                "repo_dir_only",
                predictions=[pred],
                limit=1,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(len(result.results), 1)
            row = result.results[0]
            self.assertEqual(row.status, "unsupported")
            self.assertEqual(row.failure_origin, "adapter")
            self.assertEqual(row.failure_reason, "unsupported_artifact")
            # Unsupported rows are excluded from official denominators.
            self.assertEqual(result.metrics["n_in_denominator"], 0)
            self.assertEqual(result.metrics["excluded_unsupported"], 1)


class RunnerInfraTest(unittest.TestCase):
    """An infra outcome is excluded from denominators, never crashes."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_infra_excluded_from_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=3)
            preds = [
                _patch_artifact(tasks[0].id, mock_outcome="secure_pass"),
                _patch_artifact(tasks[1].id, mock_outcome="infra"),
                _patch_artifact(tasks[2].id, mock_outcome="secure_pass"),
            ]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=3,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            by_id = {r.task_id: r for r in result.results}
            infra = by_id[tasks[1].id]
            self.assertEqual(infra.status, "infra_failure")
            self.assertEqual(result.metrics["excluded_infra"], 1)
            # Only the two secure-pass rows count toward denominators.
            self.assertEqual(result.metrics["n_in_denominator"], 2)
            self.assertNotIn(
                "infra_failure",
                {
                    r.status
                    for r in result.results
                    if r.status in ("completed", "model_failure")
                },
            )

    def test_evaluate_exception_maps_to_infra_rows(self) -> None:
        """An exception out of evaluate -> infra rows, not a crash."""

        class _BoomProvider(InMemoryEnvProvider):
            def evaluate(self, env, staged, run_config, resource_budget):
                raise RuntimeError("simulated docker pull failure")

        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=2)
            preds = [_patch_artifact(t.id) for t in tasks]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=2,
                env_provider=_BoomProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(len(result.results), 2)
            for row in result.results:
                self.assertEqual(row.status, "infra_failure")
                self.assertEqual(row.failure_origin, "infra")
            self.assertEqual(result.metrics["n_in_denominator"], 0)
            self.assertEqual(result.metrics["excluded_infra"], 2)


class RunnerStageRejectionTest(unittest.TestCase):
    """A stage-time UnsupportedArtifactError is adapter-scoped, not infra."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_stage_rejection_maps_to_unsupported_not_infra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = _StageRejectSource().load(limit=2)
            preds = [_patch_artifact(t.id) for t in tasks]
            result = runner.run(
                "stage_reject",
                "stage_reject",
                predictions=preds,
                limit=2,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(len(result.results), 2)
            for row in result.results:
                # Adapter rejection: NOT blamed on the environment.
                self.assertEqual(row.status, "unsupported")
                self.assertEqual(row.failure_origin, "adapter")
                self.assertEqual(row.failure_reason, "unsupported_artifact")
            self.assertEqual(result.metrics["excluded_infra"], 0)
            self.assertEqual(result.metrics["excluded_unsupported"], 2)
            self.assertEqual(result.metrics["n_in_denominator"], 0)


class RunnerRepoDirCacheKeyTest(unittest.TestCase):
    """repo_dir cache keys reflect materialized CONTENT, not the path string."""

    def setUp(self) -> None:
        ensure_vibe_registrations()

    def test_content_hash_keys_cache_for_repo_dir(self) -> None:
        runner = VibeRunner(cache_dir=Path("/tmp") / "unused-geh")
        oracle = oracle_registry.get("repo_dir_only")()
        run_config = OracleRunConfig(run_id="r", run_dir="/tmp/run")
        task = VibeTask(
            id="repo_dir_only/task-0",
            source_dataset="repo_dir_only",
            task_type="repo_dir",
            repo=RepoSpec(base_commit="0" * 40),
            environment=TaskEnvironmentRef(oracle="repo_dir_only"),
        )
        # Same artifact (identical worktree path string) but two different
        # materialized tree digests must yield different cache keys, so a
        # regenerated worktree can never return a stale cached verdict.
        artifact = AgentArtifact(
            task_id=task.id, model="m", kind="repo_dir", worktree="/gen/dir"
        )
        key_a = runner._cache_key_builder(
            oracle, run_config, {task.id: "AAA"}
        )(task, artifact)
        key_b = runner._cache_key_builder(
            oracle, run_config, {task.id: "BBB"}
        )(task, artifact)
        self.assertEqual(key_a.artifact_sha256, "AAA")
        self.assertEqual(key_b.artifact_sha256, "BBB")
        self.assertNotEqual(key_a.artifact_sha256, key_b.artifact_sha256)
        # With no content hash (non-worktree oracle), it falls back to the
        # artifact's own sha so existing keying is unchanged.
        fallback = runner._cache_key_builder(oracle, run_config, {})(
            task, artifact
        )
        self.assertEqual(fallback.artifact_sha256, artifact_sha256(artifact))


class RunnerRepoDirDedupTest(unittest.TestCase):
    """Duplicate repo_dir candidates dedup BEFORE materialization.

    Guards the materialize/ASE layer: _materialize_if_needed iterates only
    inputs.pairs, and _classify_artifacts dedups duplicate task_ids into that
    list first, so two repo_dir candidates for one task can never both
    materialize at artifacts/<task_id>/materialized-worktree or share one
    content hash. The surplus candidate becomes an unsupported row (run
    trials/pass@k as separate runs with distinct run_id).
    """

    def setUp(self) -> None:
        ensure_vibe_registrations()

    def test_duplicate_repo_dir_candidates_not_materialized_twice(self) -> None:
        runner = VibeRunner(cache_dir=Path("/tmp") / "unused-geh")
        oracle = oracle_registry.get("repo_dir_only")()
        task = _RepoDirSource().load(limit=1)[0]
        a1 = AgentArtifact(
            task_id=task.id, model="m", kind="repo_dir", worktree="/gen/A"
        )
        a2 = AgentArtifact(
            task_id=task.id, model="m", kind="repo_dir", worktree="/gen/B"
        )
        inputs = runner._classify_artifacts(oracle, [task], [a1, a2])
        # Exactly one candidate reaches materialization/scoring; the duplicate
        # is flagged, so no shared materialized-worktree or content hash.
        self.assertEqual(len(inputs.pairs), 1)
        self.assertEqual(inputs.pairs[0][1].worktree, "/gen/A")
        self.assertEqual(len(inputs.unsupported), 1)
        self.assertEqual(inputs.unsupported[0].status, "unsupported")
        self.assertEqual(
            inputs.unsupported[0].failure_reason, "unsupported_artifact"
        )


class RunnerCacheTest(unittest.TestCase):
    """A 2nd identical run hits the cache and skips evaluate."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_second_run_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "geh"
            tasks = _SpyBatchSource().load(limit=2)
            preds = [
                _patch_artifact(t.id, mock_outcome="secure_pass")
                for t in tasks
            ]

            first = VibeRunner(cache_dir=cache_dir).run(
                "spy_batch",
                "spy_batch",
                predictions=preds,
                limit=2,
                env_provider=_CountingEnvProvider(),
                run_dir=Path(tmp) / "run1",
            )
            self.assertEqual(_EVALUATE_CALLS.get("spy_batch"), 1)
            self.assertTrue(
                all(r.status == "completed" for r in first.results)
            )

            _reset_counters()
            second = VibeRunner(cache_dir=cache_dir).run(
                "spy_batch",
                "spy_batch",
                predictions=preds,
                limit=2,
                env_provider=_CountingEnvProvider(),
                run_dir=Path(tmp) / "run2",
            )
            # Full cache hit: evaluate is never invoked on the rerun.
            self.assertIsNone(_EVALUATE_CALLS.get("spy_batch"))
            self.assertEqual(len(second.results), 2)
            self.assertEqual(
                {r.task_id for r in second.results},
                {t.id for t in tasks},
            )

    def test_infra_origin_rows_never_cached(self) -> None:
        # A row whose status is cacheable (completed) but whose verdict came
        # from an infra problem (failure_origin="infra", e.g. SecCodeBench's
        # upstream-parity scored fail for an unverifiable submission) is
        # environment-dependent: caching it would replay a transient outage
        # as a permanent 0 against a later healthy verifier.
        class _RecordingCache:
            def __init__(self) -> None:
                self.puts: list[str] = []

            def put(self, key, row) -> None:
                self.puts.append(row.task_id)

        task = MockTaskSource().load(limit=1)[0]
        artifact = _patch_artifact(task.id)
        infra_row = VibeTaskResult(
            task_id=task.id,
            source_dataset=task.source_dataset,
            model="m",
            status="completed",
            failure_origin="infra",
            failure_reason="verifier_unavailable",
            functional_pass=False,
            security_oracle_pass=False,
        )
        ok_row = infra_row.model_copy(
            update={"failure_origin": "none", "failure_reason": None}
        )
        cache = _RecordingCache()
        VibeRunner()._store(
            [infra_row, ok_row],
            [(task, artifact), (task, artifact)],
            cache,
            lambda t, a: "key",
            no_cache=False,
        )
        # Only the environment-independent row is stored.
        self.assertEqual(cache.puts, [task.id])

    def test_stale_infra_origin_cache_hits_are_ignored(self) -> None:
        # Rows cached BEFORE the infra-origin write guard (completed status,
        # failure_origin=infra) must not replay: the read side treats them
        # as misses so the task re-scores against the current environment.
        class _StubCache:
            def __init__(self, hit) -> None:
                self._hit = hit

            def get(self, key):
                return self._hit

        task = MockTaskSource().load(limit=1)[0]
        stale = VibeTaskResult(
            task_id=task.id,
            source_dataset=task.source_dataset,
            model="m",
            status="completed",
            failure_origin="infra",
            failure_reason="verifier_unavailable",
            functional_pass=False,
            security_oracle_pass=False,
        )
        self.assertIsNone(
            VibeRunner._cache_hit(_StubCache(stale), object(), False)
        )
        healthy = stale.model_copy(
            update={"failure_origin": "none", "failure_reason": None}
        )
        self.assertIs(
            VibeRunner._cache_hit(_StubCache(healthy), object(), False),
            healthy,
        )

    def test_no_cache_forces_reevaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "geh"
            tasks = _SpyBatchSource().load(limit=2)
            preds = [
                _patch_artifact(t.id, mock_outcome="secure_pass")
                for t in tasks
            ]
            VibeRunner(cache_dir=cache_dir).run(
                "spy_batch",
                "spy_batch",
                predictions=preds,
                limit=2,
                env_provider=_CountingEnvProvider(),
                run_dir=Path(tmp) / "run1",
            )
            _reset_counters()
            VibeRunner(cache_dir=cache_dir).run(
                "spy_batch",
                "spy_batch",
                predictions=preds,
                limit=2,
                env_provider=_CountingEnvProvider(),
                run_dir=Path(tmp) / "run2",
                no_cache=True,
            )
            self.assertEqual(_EVALUATE_CALLS.get("spy_batch"), 1)


class RunnerLiveAgentTest(unittest.TestCase):
    """live_agent mode drives a registered agent, then scores its output."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_live_agent_generates_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            result = runner.run(
                "mock",
                "mock",
                mode="live_agent",
                agent="mock_agent",
                agent_model="claude-opus-4-8",
                limit=3,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            # One scored row per loaded task, each from the live driver's
            # generated artifact (carrying the resolved model).
            self.assertEqual(len(result.results), 3)
            self.assertTrue(
                all(r.model == "claude-opus-4-8" for r in result.results)
            )
            # No cached upstream checkout for 'mock' -> the driver gets no
            # workdir (its content would live in a dataset Docker image).
            self.assertTrue(all(w is None for w in _AGENT_WORKDIRS))
            # The runner threads the oracle's GenerationSpec to the driver so
            # generation emits the kind the oracle scores (mock -> patch).
            self.assertEqual(len(_AGENT_GEN_SPECS), 3)
            self.assertTrue(
                all(
                    getattr(s, "artifact_kind", None) == "patch"
                    for s in _AGENT_GEN_SPECS
                )
            )
            # The run dir records the live_agent provenance.
            manifest = json.loads(
                (Path(tmp) / "run" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["mode"], "live_agent")

    def test_live_agent_skips_non_matching_checkout(self) -> None:
        """A cached upstream that is NOT the task repo is never handed to the agent.

        The seeded ``upstreams/mock`` tree stands in for the benchmark
        harness/dataset checkout (e.g. SusVibes' evaluator tree). The mock
        task's ``base_commit`` is not part of it, so ``restore_base`` must
        return ``None`` -- the in-container driver extracts the real per-task
        repo from the eval image -- rather than leak harness files into the
        agent workdir.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "geh"
            upstream = cache / "upstreams" / "mock"
            upstream.mkdir(parents=True)
            (upstream / "app.py").write_text("x = 1\n", encoding="utf-8")
            (upstream / ".git").mkdir()
            (upstream / ".git" / "HEAD").write_text("ref: x\n")
            runner = VibeRunner(cache_dir=cache)
            runner.run(
                "mock",
                "mock",
                mode="live_agent",
                agent="mock_agent",
                limit=1,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            # The harness checkout does not contain the task base_commit, so the
            # driver gets no workdir (not the leaked harness tree).
            self.assertIsNone(_AGENT_WORKDIRS[-1])

    def test_live_agent_uses_oracle_resolved_base(self) -> None:
        """When the oracle exposes ``live_base``, the agent is handed that
        sealed checkout -- not None -- even with no ``upstreams/<dataset>``
        tree. This is the SecureVibeBench path: the repo is available via
        ``repo.url``, so the host-side driver generates against it instead of
        blind.
        """
        with tempfile.TemporaryDirectory() as tmp:
            origin = Path(tmp) / "origin"
            sha = _make_git_repo(origin, "x = 1  # base\n")
            _LIVE_BASE_REPO.clear()
            _LIVE_BASE_REPO.update({"url": str(origin), "ref": sha})
            try:
                runner = VibeRunner(cache_dir=Path(tmp) / "geh")
                runner.run(
                    "live_base_spy",
                    "live_base_spy",
                    mode="live_agent",
                    agent="mock_agent",
                    limit=1,
                    env_provider=InMemoryEnvProvider(),
                    run_dir=Path(tmp) / "run",
                )
            finally:
                _LIVE_BASE_REPO.clear()

            workdir = _AGENT_WORKDIRS[-1]
            # The driver received a real sealed checkout, not None/blind.
            self.assertIsNotNone(workdir)
            wt = Path(str(workdir))
            self.assertEqual(
                (wt / "app.py").read_text(encoding="utf-8"), "x = 1  # base\n"
            )
            # Sealed: no .git history the agent could mine for the upstream fix.
            self.assertFalse((wt / ".git").exists())

    def test_live_agent_requires_agent_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            with self.assertRaises(ValueError):
                runner.run(
                    "mock",
                    "mock",
                    mode="live_agent",
                    agent=None,
                    limit=1,
                    env_provider=InMemoryEnvProvider(),
                    run_dir=Path(tmp) / "run",
                )

    def test_unknown_mode_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            with self.assertRaises(ValueError):
                runner.run(
                    "mock",
                    "mock",
                    predictions=[],
                    mode="bogus_mode",
                    run_dir=Path(tmp) / "run",
                )


class RunnerArtifactCoercionTest(unittest.TestCase):
    """Predictions may be dicts coerced into AgentArtifacts."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_dict_predictions_are_coerced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=1)
            pred = {
                "task_id": tasks[0].id,
                "model": "mock-model",
                "kind": "patch",
                "patch": "--- a\n+++ b\n@@ -0,0 +1 @@\n+x\n",
                "metadata": {"mock_outcome": "secure_pass"},
            }
            result = runner.run(
                "mock",
                "mock",
                predictions=[pred],
                limit=1,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            self.assertEqual(len(result.results), 1)
            self.assertIsInstance(result.results[0], VibeTaskResult)
            self.assertEqual(result.results[0].status, "completed")


class FoldAgentTelemetryTest(unittest.TestCase):
    """``_fold_agent_telemetry`` carries usage onto the artifact, intact."""

    def test_fold_carries_usage_and_preserves_metadata(self) -> None:
        from guard_eval_harness.vibecoding.runner import (
            _fold_agent_telemetry,
        )

        art = AgentArtifact(
            task_id="t/1",
            model="m",
            kind="patch",
            patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+x\n",
            metadata={"orig": "kept"},
        )
        result = AgentResult(
            artifact=art,
            model="resolved-model",
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
            cost_usd=0.01,
            metadata={"error": "boom"},
        )
        folded = _fold_agent_telemetry(result)
        # Usage/cost/model land under the reserved telemetry sub-key.
        tele = folded.metadata["agent_telemetry"]
        self.assertEqual(tele["prompt_tokens"], 3)
        self.assertEqual(tele["completion_tokens"], 4)
        self.assertEqual(tele["total_tokens"], 7)
        self.assertEqual(tele["cost_usd"], 0.01)
        self.assertEqual(tele["agent_model"], "resolved-model")
        # Driver metadata and the artifact's own metadata both survive, at the
        # top level (part of the candidate's scoreable identity).
        self.assertEqual(folded.metadata["error"], "boom")
        self.assertEqual(folded.metadata["orig"], "kept")
        # The artifact payload itself is unchanged (only metadata augmented).
        self.assertEqual(folded.patch, art.patch)
        self.assertEqual(folded.task_id, "t/1")
        self.assertEqual(folded.model, "m")


class ArtifactScoringHashTest(unittest.TestCase):
    """Folded telemetry must not perturb the scoring cache identity."""

    def test_scoring_hash_ignores_telemetry_but_sha256_does_not(self) -> None:
        from guard_eval_harness.vibecoding.artifacts import (
            artifact_scoring_sha256,
            artifact_sha256,
        )
        from guard_eval_harness.vibecoding.runner import (
            _fold_agent_telemetry,
        )

        def _candidate() -> AgentArtifact:
            return AgentArtifact(
                task_id="t/1",
                model="m",
                kind="patch",
                patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+x\n",
                metadata={"mock_outcome": "secure_pass"},
            )

        # Same scoreable patch + scoring-relevant metadata, but two runs with
        # different per-generation usage/cost.
        run_a = _fold_agent_telemetry(
            AgentResult(artifact=_candidate(), model="m", total_tokens=10)
        )
        run_b = _fold_agent_telemetry(
            AgentResult(artifact=_candidate(), model="m", total_tokens=999)
        )
        # Telemetry differs in the persisted metadata...
        self.assertNotEqual(
            run_a.metadata["agent_telemetry"],
            run_b.metadata["agent_telemetry"],
        )
        # ...and the full-artifact hash (used for provenance) reflects that...
        self.assertNotEqual(artifact_sha256(run_a), artifact_sha256(run_b))
        # ...but the SCORING identity is stable, so the oracle cache is not
        # needlessly missed across runs of an identical candidate.
        self.assertEqual(
            artifact_scoring_sha256(run_a), artifact_scoring_sha256(run_b)
        )
        # Scoring-relevant metadata is still honored: a different mock_outcome
        # yields a different scoring hash (no over-broad exclusion).
        other = _candidate().model_copy(
            update={"metadata": {"mock_outcome": "model_failure"}}
        )
        self.assertNotEqual(
            artifact_scoring_sha256(run_a), artifact_scoring_sha256(other)
        )
        # For a BYO artifact (no telemetry key) scoring == full sha256.
        byo = _candidate()
        self.assertEqual(
            artifact_scoring_sha256(byo), artifact_sha256(byo)
        )

    def test_cache_key_stable_across_telemetry_for_batch_oracle(self) -> None:
        """End-to-end: the runner's cache key for a non-worktree oracle is
        unchanged by folded telemetry, so an identical candidate re-scores
        from cache instead of needlessly re-running the oracle."""
        from guard_eval_harness.vibecoding.runner import (
            _fold_agent_telemetry,
        )

        ensure_vibe_registrations()
        runner = VibeRunner(cache_dir=Path("/tmp") / "unused-geh")
        oracle = oracle_registry.get("mock")()
        run_config = OracleRunConfig(run_id="r", run_dir="/tmp/run")
        task = MockTaskSource().load(limit=1)[0]

        def _folded(total_tokens: int) -> AgentArtifact:
            art = AgentArtifact(
                task_id=task.id,
                model="m",
                kind="patch",
                patch="--- a\n+++ b\n@@ -0,0 +1 @@\n+x\n",
            )
            return _fold_agent_telemetry(
                AgentResult(artifact=art, model="m", total_tokens=total_tokens)
            )

        # No content hash (non-worktree oracle) -> the artifact-hash fallback.
        build = runner._cache_key_builder(oracle, run_config, {})
        key_a = build(task, _folded(10))
        key_b = build(task, _folded(999))
        self.assertEqual(key_a.artifact_sha256, key_b.artifact_sha256)


class RunnerLiveTelemetryTest(unittest.TestCase):
    """Live-agent token/cost/model telemetry reaches the persisted artifact."""

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_live_agent_persists_token_and_cost_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            runner.run(
                "mock",
                "mock",
                mode="live_agent",
                agent="telemetry_agent",
                agent_model="claude-opus-4-8",
                limit=2,
                env_provider=InMemoryEnvProvider(),
                run_dir=run_dir,
            )
            # Telemetry must reach the on-disk artifact.json (run store), not
            # vanish after generation with the AgentResult wrapper.
            artifact_jsons = sorted(
                (run_dir / "artifacts").rglob("artifact.json")
            )
            self.assertEqual(len(artifact_jsons), 2)
            for path in artifact_jsons:
                meta = json.loads(path.read_text())["metadata"]
                tele = meta["agent_telemetry"]
                self.assertEqual(tele["prompt_tokens"], 11)
                self.assertEqual(tele["completion_tokens"], 7)
                self.assertEqual(tele["total_tokens"], 18)
                self.assertAlmostEqual(tele["cost_usd"], 0.0023)
                self.assertEqual(tele["agent_model"], "claude-opus-4-8")
                # Driver metadata + the artifact's own metadata survive too.
                self.assertEqual(meta["note"], "driver-meta-survives")
                self.assertEqual(meta["orig"], "kept")


class RunnerCoverageReconciliationTest(unittest.TestCase):
    """A loaded task with no prediction is an in-denominator non-submission.

    ``compute_vibe_metrics`` keys its denominators on the RESULTS list, so a
    loaded task absent from the predictions would silently shrink every
    denominator. The runner must reconcile coverage against the loaded task set
    and score each missing task as an in-denominator ``model_failure`` /
    ``no_submission`` fail, exactly as every upstream benchmark counts a
    missing submission. The reconciliation is a no-op when coverage is already
    complete.
    """

    def setUp(self) -> None:
        ensure_vibe_registrations()
        _reset_counters()

    def test_missing_predictions_become_in_denominator_fails(self) -> None:
        # 5 loaded tasks, predictions for only 3 -> 2 must be reconciled.
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=5)
            present = tasks[:3]
            missing = tasks[3:]
            preds = [
                _patch_artifact(t.id, mock_outcome="secure_pass")
                for t in present
            ]
            with self.assertLogs(
                "guard_eval_harness.vibecoding.runner", level="WARNING"
            ) as captured:
                result = runner.run(
                    "mock",
                    "mock",
                    predictions=preds,
                    limit=5,
                    env_provider=InMemoryEnvProvider(),
                    run_dir=Path(tmp) / "run",
                )

            # Every loaded task has exactly one row now (N rows for N tasks).
            self.assertEqual(len(result.results), len(tasks))
            self.assertEqual(
                {r.task_id for r in result.results},
                {t.id for t in tasks},
            )

            by_id = {r.task_id: r for r in result.results}
            for task in missing:
                row = by_id[task.id]
                self.assertEqual(row.status, "model_failure")
                self.assertEqual(row.failure_origin, "model")
                self.assertEqual(row.failure_reason, "no_submission")
                self.assertTrue(in_denominator(row))
                # A non-submission is a definite not-secure fail (not null).
                self.assertIs(row.functional_pass, False)
                self.assertIs(row.security_oracle_pass, False)
                self.assertIs(row.target_secure_success, False)
                # Same source_dataset + oracle capabilities as a real row.
                self.assertEqual(row.source_dataset, task.source_dataset)
                self.assertEqual(
                    row.oracle_capabilities,
                    by_id[present[0].id].oracle_capabilities,
                )
                # Best-effort model reuse from the other scored rows.
                self.assertEqual(row.model, by_id[present[0].id].model)

            # Exactly one warning naming the count and the missing ids.
            warnings = [
                rec.getMessage()
                for rec in captured.records
                if rec.levelname == "WARNING"
            ]
            self.assertEqual(len(warnings), 1)
            self.assertIn("2 of 5 loaded task(s) had no prediction", warnings[0])
            for task in missing:
                self.assertIn(task.id, warnings[0])

    def test_full_predictions_no_placeholder_no_warning(self) -> None:
        # A full predictions file leaves zero missing: byte-identical results.
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=4)
            preds = [
                _patch_artifact(t.id, mock_outcome="secure_pass")
                for t in tasks
            ]
            logger = logging.getLogger(
                "guard_eval_harness.vibecoding.runner"
            )
            with self.assertNoLogs(logger, level="WARNING"):
                result = runner.run(
                    "mock",
                    "mock",
                    predictions=preds,
                    limit=4,
                    env_provider=InMemoryEnvProvider(),
                    run_dir=Path(tmp) / "run",
                )
            self.assertEqual(len(result.results), 4)
            # No reconciliation row was synthesized.
            self.assertFalse(
                any(
                    r.failure_reason == "no_submission"
                    for r in result.results
                )
            )

    def test_placeholders_lower_metric_rate(self) -> None:
        # 2 scored secure passes + 2 reconciled fails -> rate halves.
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=4)
            preds = [
                _patch_artifact(t.id, mock_outcome="secure_pass")
                for t in tasks[:2]
            ]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=4,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            metrics = compute_vibe_metrics(result.results, result.tasks)
            self.assertEqual(metrics["n_total"], 4)
            self.assertEqual(metrics["n_in_denominator"], 4)
            self.assertEqual(metrics["excluded_infra"], 0)
            self.assertEqual(metrics["excluded_unsupported"], 0)
            # 2 of 4 in-denominator rows are target-secure -> rate 0.5.
            cell = metrics["cells"]["target_secure_success"]
            self.assertEqual(cell["n_scored"], 4)
            self.assertEqual(cell["rate"], 0.5)

    def test_unmatched_prediction_still_unsupported_not_double_counted(
        self,
    ) -> None:
        # A prediction with no matching task is an 'unsupported' row (existing
        # behavior). It must NOT be turned into a reconciliation placeholder,
        # and the loaded task it does not cover IS reconciled exactly once.
        with tempfile.TemporaryDirectory() as tmp:
            runner = VibeRunner(cache_dir=Path(tmp) / "geh")
            tasks = MockTaskSource().load(limit=2)
            preds = [
                _patch_artifact(tasks[0].id, mock_outcome="secure_pass"),
                # No loaded task matches this id -> 'unsupported' row.
                _patch_artifact("mock/does-not-exist", mock_outcome="secure_pass"),
            ]
            result = runner.run(
                "mock",
                "mock",
                predictions=preds,
                limit=2,
                env_provider=InMemoryEnvProvider(),
                run_dir=Path(tmp) / "run",
            )
            by_status = defaultdict(list)
            for row in result.results:
                by_status[row.status].append(row)
            # 3 rows: 1 scored, 1 unsupported (orphan pred), 1 reconciled fail.
            self.assertEqual(len(result.results), 3)

            unsupported = by_status["unsupported"]
            self.assertEqual(len(unsupported), 1)
            self.assertEqual(unsupported[0].task_id, "mock/does-not-exist")
            self.assertEqual(
                unsupported[0].failure_reason, "unsupported_artifact"
            )

            # The orphan id is NOT also reconciled as a no_submission row.
            no_sub = [
                r for r in result.results
                if r.failure_reason == "no_submission"
            ]
            self.assertEqual(len(no_sub), 1)
            self.assertEqual(no_sub[0].task_id, tasks[1].id)

            # Metrics: 1 scored pass + 1 reconciled fail in-denominator; the
            # orphan unsupported row is excluded, never double-counted.
            self.assertEqual(result.metrics["n_in_denominator"], 2)
            self.assertEqual(result.metrics["excluded_unsupported"], 1)


if __name__ == "__main__":
    unittest.main()
