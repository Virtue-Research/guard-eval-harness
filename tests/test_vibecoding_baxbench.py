"""Unit tests for the BaxBench task source + oracle adapter.

These tests never touch Docker or the network. They exercise:

- ``BaxBenchTaskSource.load`` against tiny ``scenarios/<id>/scenario.json``
  fixtures, expanding each scenario across its envs into ``project_scaffold``
  ``VibeTask`` records (``baxbench/<scenario>__<env>``).
- ``BaxBenchOracle.stage`` writing each candidate's generated app files into
  the upstream ``results/<model>/<scenario>/<env>/<leaf>/sample<N>/code/``
  layout and rejecting non-``full_file`` artifacts.
- ``BaxBenchOracle.evaluate`` driving a STUBBED env provider (no real
  subprocess / Docker) and building the expected ``test`` + ``evaluate`` argv,
  honoring a small ``ResourceBudget``.
- ``BaxBenchOracle.parse`` mapping each upstream ``test_results.json`` onto a
  normalized ``VibeTaskResult`` with correct status / failure attribution /
  tri-state verdicts, including a COMPLETED secure case (target_secure True),
  a COMPLETED insecure case (target vuln present), a MODEL-FAILURE build case,
  and an INFRA-FAILURE missing-results case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.baxbench import (
    BaxBenchOracle,
    _parse_single_file,
    esc,
)
from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    oracle_registry,
    task_source_registry,
)
from guard_eval_harness.vibecoding.results import derive_task_metrics
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    VibeTask,
)
from guard_eval_harness.vibecoding.sources.baxbench import (
    _ALL_ENVS,
    BaxBenchTaskSource,
    split_task_id,
    task_id_for,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "vibecoding" / "baxbench"
_SCENARIOS_DIR = _FIXTURES / "scenarios"

_SAVE_LEAF = "temp0.0-openapi-none"
_MODEL = "byo-model"

_SAMPLE_FILES = {
    "app.py": (
        "from flask import Flask, request, jsonify\n"
        "app = Flask(__name__)\n"
        "@app.post('/calculator')\n"
        "def calc():\n"
        "    return jsonify(result='7')\n"
    ),
}

# Per-case upstream output reproduced inline (no committed output fixtures).
# Each value is either a ``TestResult``-shaped ``test_results.json`` payload
# (the exact keys ``tasks.TestResult.to_dict`` emits) written into the sample
# dir, or ``None`` for the infra-missing case, which writes NO
# ``test_results.json`` -- only a ``code/app.py`` (so the oracle finds the
# sample dir / code but never a scoreable result).
_CASE_TEST_RESULTS: dict[str, dict[str, object] | None] = {
    # Built + ran; all functional tests pass; no CWE flagged -> target-secure.
    "secure": {
        "cwes": [],
        "num_ft_exceptions": 0,
        "num_passed_ft": 2,
        "num_st_exceptions": 0,
        "num_total_ft": 2,
        "num_total_st": 3,
    },
    # Built + ran; all functional tests pass; an exploit flagged CWE-94.
    "insecure": {
        "cwes": [94],
        "num_ft_exceptions": 0,
        "num_passed_ft": 2,
        "num_st_exceptions": 0,
        "num_total_ft": 2,
        "num_total_st": 3,
    },
    # Did not build/run: every functional + security test errored out.
    "build_failed": {
        "cwes": [],
        "num_ft_exceptions": 2,
        "num_passed_ft": 0,
        "num_st_exceptions": 3,
        "num_total_ft": 2,
        "num_total_st": 3,
    },
    # Built + ran; only some functional tests pass; no CWE flagged.
    "func_fail": {
        "cwes": [],
        "num_ft_exceptions": 0,
        "num_passed_ft": 1,
        "num_st_exceptions": 0,
        "num_total_ft": 2,
        "num_total_st": 3,
    },
    # No test_results.json at all -- only generated code present.
    "infra_missing": None,
}

# Generated app file present for the infra-missing case (code, no results).
_INFRA_MISSING_APP_PY = "print('hi')\n"


# --- stub env provider -------------------------------------------------


class _StubResolved:
    """Minimal stand-in for ``ResolvedEnv`` returned by ``ensure_ready``."""

    def __init__(
        self, upstream_dir: str, venv_python: str,
        cache_dir: str = "/fake/cache",
    ) -> None:
        self.upstream_dir = upstream_dir
        self.venv_python = venv_python
        self.cache_dir = cache_dir


class _StubCommandResult:
    """Minimal stand-in for ``CommandResult``."""

    def __init__(
        self,
        returncode: int = 0,
        timed_out: bool = False,
        stderr: str = "",
        stderr_path: str | None = None,
    ) -> None:
        self.returncode = returncode
        self.timed_out = timed_out
        self.stderr = stderr
        self.stderr_path = stderr_path


class _StubEnvProvider:
    """Records ``run`` calls; never spawns a process or touches Docker."""

    def __init__(
        self,
        *,
        upstream_dir: str = "/fake/upstream",
        returncode: int = 0,
    ) -> None:
        self.upstream_dir = upstream_dir
        self.returncode = returncode
        self.ensure_ready_called = False
        self.run_calls: list[list[str]] = []
        self.run_budgets: list[object] = []

    def ensure_ready(self, *, force: bool = False) -> _StubResolved:
        self.ensure_ready_called = True
        return _StubResolved(
            upstream_dir=self.upstream_dir,
            venv_python=f"{self.upstream_dir}/.venv/bin/python",
        )

    def run(
        self,
        argv: list[str],
        *,
        run_dir,
        timeout_s=None,
        budget=None,
        extra_env=None,
    ) -> _StubCommandResult:
        self.run_calls.append(list(argv))
        self.run_budgets.append(budget)
        return _StubCommandResult(returncode=self.returncode)


# --- helpers -----------------------------------------------------------


def _source() -> BaxBenchTaskSource:
    return BaxBenchTaskSource(scenarios_dir=_SCENARIOS_DIR)


def _task(scenario: str, env: str) -> VibeTask:
    """Load a single fixture task by (scenario, env) via the task source."""
    target = task_id_for(scenario, env)
    for task in _source().load():
        if task.id == target:
            return task
    raise AssertionError(f"fixture task {target} not found")


def _artifact(
    scenario: str,
    env: str,
    *,
    files: dict[str, str] | None = None,
    model: str = _MODEL,
) -> AgentArtifact:
    return AgentArtifact(
        task_id=task_id_for(scenario, env),
        model=model,
        kind="full_file",
        files=dict(files or _SAMPLE_FILES),
    )


def _write_case_outputs(
    root: Path,
    case: str,
    *,
    model: str,
    scenario: str,
    env: str,
) -> Path:
    """Materialize one upstream output case under ``root``.

    Reproduces the exact ``results/<model>/<scenario>/<env>/<leaf>/sample0/``
    layout the parser reads: for a scoreable case it writes the inline
    ``test_results.json`` payload; for the infra-missing case it writes only
    ``code/app.py`` and NO ``test_results.json``. Returns the sample dir.
    """
    sample_dir = (
        root
        / esc(model)
        / esc(scenario)
        / esc(env)
        / _SAVE_LEAF
        / "sample0"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = _CASE_TEST_RESULTS[case]
    if payload is None:
        # Infra-missing: code present, but no scoreable result was produced.
        code_dir = sample_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "app.py").write_text(
            _INFRA_MISSING_APP_PY, encoding="utf-8"
        )
    else:
        (sample_dir / "test_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return sample_dir


def _raw_for_case(
    root: Path,
    case: str,
    scenario: str = "Calculator",
    env: str = "Python-Flask",
    *,
    model: str = _MODEL,
    timed_out: bool = False,
) -> RawOracleResult:
    """Build a ``RawOracleResult`` over a freshly written output case in ``root``.

    The upstream ``test_results`` output is reproduced inline under ``root``
    (no committed output fixtures) before the raw result is assembled.
    """
    task_id = task_id_for(scenario, env)
    results_root = root / case
    sample_dir = _write_case_outputs(
        results_root, case, model=model, scenario=scenario, env=env
    )
    return RawOracleResult(
        adapter_name="baxbench",
        outputs_dir=str(results_root),
        logs_dir=str(results_root / "logs"),
        exit_code=0,
        task_ids=[task_id],
        metadata={
            "results_root": str(results_root),
            "sample": 0,
            "timed_out": timed_out,
            "per_task": {
                task_id: {
                    "scenario": scenario,
                    "env": env,
                    "model": model,
                    "sample": 0,
                    "sample_dir": str(sample_dir),
                    "code_dir": str(sample_dir / "code"),
                    "test_results_path": str(
                        sample_dir / "test_results.json"
                    ),
                    "source_dataset": "baxbench",
                    "artifact_sha256": "deadbeef",
                    "task_sha256": "cafef00d",
                }
            },
        },
    )


def _parse_one(root: Path, case: str, **kwargs):
    raw = _raw_for_case(root, case, **kwargs)
    rows = BaxBenchOracle().parse(raw)
    assert len(rows) == 1
    return rows[0]


# --- registration ------------------------------------------------------


def test_registered_in_registries() -> None:
    ensure_vibe_registrations()
    assert oracle_registry.get("baxbench") is BaxBenchOracle
    assert task_source_registry.get("baxbench") is BaxBenchTaskSource


# --- task source -------------------------------------------------------


def test_load_builds_vibe_tasks() -> None:
    tasks = _source().load()
    ids = {t.id for t in tasks}
    # Calculator restricts to two envs; Logger to one.
    assert ids == {
        "baxbench/Calculator__Python-Flask",
        "baxbench/Calculator__Python-FastAPI",
        "baxbench/Logger__Python-Flask",
    }

    task = _task("Calculator", "Python-Flask")
    assert task.source_dataset == "baxbench"
    assert task.task_type == "project_scaffold"
    assert task.environment is not None
    assert task.environment.oracle == "baxbench"
    assert task.environment.requires_docker is True
    # Security requirement -> CWE labels.
    assert task.labels.cwe == ["CWE-94", "CWE-400"]
    # The functional spec is carried as the instructions.
    assert "calculator" in task.instructions.lower()
    assert "/calculator" in task.instructions


def test_load_respects_limit() -> None:
    assert len(_source().load(limit=1)) == 1


def test_load_scenario_without_cwes_has_empty_labels() -> None:
    task = _task("Logger", "Python-Flask")
    assert task.labels.cwe == []
    assert task.task_type == "project_scaffold"


_TEST_UPSTREAM_REF = "testref0123"


class _SpecStub:
    upstream_ref = _TEST_UPSTREAM_REF


class _MaterializingProvider:
    """Fake provider whose ``run`` simulates the upstream descriptor export:
    it writes ``write_ids`` valid ``scenario.json`` files into the
    ``BAX_SCEN_DIR`` the source passes via ``extra_env``."""

    env_spec = _SpecStub()

    def __init__(
        self,
        *,
        write_ids: tuple[str, ...] = ("Calculator", "Logger"),
        returncode: int = 0,
    ) -> None:
        self.write_ids = write_ids
        self.returncode = returncode
        self.ensure_ready_called = False
        self.run_calls = 0
        self.last_env: dict[str, str] | None = None

    def ensure_ready(self, *, force: bool = False) -> _StubResolved:
        self.ensure_ready_called = True
        return _StubResolved(
            upstream_dir="/fake/upstream",
            venv_python="/fake/upstream/.venv/bin/python",
        )

    def run(self, argv, *, run_dir, timeout_s=None, extra_env=None, **kw):
        self.run_calls += 1
        self.last_env = dict(extra_env or {})
        if self.returncode == 0:
            scen_dir = Path(self.last_env["BAX_SCEN_DIR"])
            envs = json.loads(self.last_env["BAX_ENV_IDS"])
            for sid in self.write_ids:
                d = scen_dir / sid
                d.mkdir(parents=True, exist_ok=True)
                (d / "scenario.json").write_text(
                    json.dumps(
                        {"id": sid, "instructions": "spec", "cwes": [],
                         "envs": envs}
                    ),
                    encoding="utf-8",
                )
            # Mirror the real export: descriptors for scenarios that no
            # longer exist upstream are pruned.
            for stale in scen_dir.glob("*/scenario.json"):
                if stale.parent.name not in self.write_ids:
                    stale.unlink()
        return _StubCommandResult(returncode=self.returncode)


def test_ensure_descriptors_materializes_when_missing(tmp_path: Path) -> None:
    scen = tmp_path / "scenarios"
    scen.mkdir()
    provider = _MaterializingProvider(write_ids=("Calculator", "Logger"))
    source = BaxBenchTaskSource()
    source._ensure_descriptors(provider, scen)

    assert provider.ensure_ready_called
    assert provider.run_calls == 1
    # The export was asked to write the full 14-env list (paper parity).
    assert json.loads(provider.last_env["BAX_ENV_IDS"]) == list(_ALL_ENVS)
    # A manifest recording the completed export was written last.
    manifest = json.loads(
        (scen / ".geh_descriptors.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "env_ids": list(_ALL_ENVS),
        "descriptor_count": 2,
        "upstream_ref": _TEST_UPSTREAM_REF,
    }
    # The descriptors now load into 2 scenarios x 14 envs = 28 tasks.
    tasks = BaxBenchTaskSource(scenarios_dir=scen).load()
    assert len(tasks) == 2 * len(_ALL_ENVS)
    # A second load verifies the manifest and never provisions again.
    source._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1


def test_ensure_descriptors_refreshes_stale_unmanifested_set(
    tmp_path: Path,
) -> None:
    # Descriptors WITHOUT a manifest (e.g. written by the older single-env
    # exporter, or left by an interrupted export) are stale evidence, not
    # proof of currency: the export must re-run and refresh them.
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text(
        json.dumps({"id": "Calculator", "envs": ["Python-Flask"]}),
        encoding="utf-8",
    )
    provider = _MaterializingProvider(write_ids=("Calculator", "Logger"))
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1
    # The refreshed descriptor carries the full env list again.
    refreshed = json.loads(
        (scen / "Calculator" / "scenario.json").read_text(encoding="utf-8")
    )
    assert refreshed["envs"] == list(_ALL_ENVS)


def test_ensure_descriptors_refreshes_on_env_list_change(
    tmp_path: Path,
) -> None:
    # A manifest from an export with a DIFFERENT env list (older or partial
    # materializer) must not satisfy the guard.
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text("{}", encoding="utf-8")
    (scen / ".geh_descriptors.json").write_text(
        json.dumps({"env_ids": ["Python-Flask"], "descriptor_count": 1}),
        encoding="utf-8",
    )
    provider = _MaterializingProvider()
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1


def test_ensure_descriptors_refreshes_partial_set(tmp_path: Path) -> None:
    # Manifest says 2 descriptors but only 1 survives on disk: the set is
    # partial and must be re-materialized.
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text("{}", encoding="utf-8")
    (scen / ".geh_descriptors.json").write_text(
        json.dumps(
            {
                "env_ids": list(_ALL_ENVS),
                "descriptor_count": 2,
                "upstream_ref": _TEST_UPSTREAM_REF,
            }
        ),
        encoding="utf-8",
    )
    provider = _MaterializingProvider(write_ids=("Calculator", "Logger"))
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1


def test_ensure_descriptors_refreshes_on_upstream_ref_bump(
    tmp_path: Path,
) -> None:
    # A pin bump must invalidate descriptors exported from the previous
    # checkout: the manifest records the ref it was exported from, and a
    # mismatch with the current EnvSpec pin forces a refresh.
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text(
        json.dumps({"id": "Calculator", "envs": list(_ALL_ENVS)}),
        encoding="utf-8",
    )
    (scen / ".geh_descriptors.json").write_text(
        json.dumps(
            {
                "env_ids": list(_ALL_ENVS),
                "descriptor_count": 1,
                "upstream_ref": "previous-pin",
            }
        ),
        encoding="utf-8",
    )
    provider = _MaterializingProvider(write_ids=("Calculator",))
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1
    manifest = json.loads(
        (scen / ".geh_descriptors.json").read_text(encoding="utf-8")
    )
    assert manifest["upstream_ref"] == _TEST_UPSTREAM_REF


def test_ensure_descriptors_prunes_ghost_scenarios(tmp_path: Path) -> None:
    # A pin refresh leaves untracked descriptors behind for scenarios the
    # new upstream no longer ships; the refresh prunes them so load() cannot
    # return ghost tasks, and the manifest counts only current descriptors.
    scen = tmp_path / "scenarios"
    (scen / "RemovedScenario").mkdir(parents=True)
    (scen / "RemovedScenario" / "scenario.json").write_text(
        json.dumps({"id": "RemovedScenario", "envs": ["Python-Flask"]}),
        encoding="utf-8",
    )
    provider = _MaterializingProvider(write_ids=("Calculator",))
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1
    assert not (scen / "RemovedScenario" / "scenario.json").exists()
    manifest = json.loads(
        (scen / ".geh_descriptors.json").read_text(encoding="utf-8")
    )
    assert manifest["descriptor_count"] == 1
    tasks = BaxBenchTaskSource(scenarios_dir=scen).load()
    assert {t.id.split("/")[1].split("__")[0] for t in tasks} == {
        "Calculator"
    }


def test_ensure_descriptors_noop_when_manifest_current(
    tmp_path: Path,
) -> None:
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text(
        json.dumps({"id": "Calculator", "envs": list(_ALL_ENVS)}),
        encoding="utf-8",
    )
    (scen / ".geh_descriptors.json").write_text(
        json.dumps(
            {
                "env_ids": list(_ALL_ENVS),
                "descriptor_count": 1,
                "upstream_ref": _TEST_UPSTREAM_REF,
            }
        ),
        encoding="utf-8",
    )
    provider = _MaterializingProvider()
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    # Verified current -> never provisions or runs the export.
    assert provider.run_calls == 0
    assert not provider.ensure_ready_called


def test_ensure_descriptors_refreshes_legacy_overwrite_under_valid_manifest(
    tmp_path: Path,
) -> None:
    # The legacy single-env exporter can overwrite the descriptor FILES with
    # envs=["Python-Flask"] while leaving a previously-written manifest
    # untouched. The manifest alone must not vouch for files it did not
    # write: per-descriptor env validation forces a refresh.
    scen = tmp_path / "scenarios"
    (scen / "Calculator").mkdir(parents=True)
    (scen / "Calculator" / "scenario.json").write_text(
        json.dumps({"id": "Calculator", "envs": ["Python-Flask"]}),
        encoding="utf-8",
    )
    (scen / ".geh_descriptors.json").write_text(
        json.dumps(
            {
                "env_ids": list(_ALL_ENVS),
                "descriptor_count": 1,
                "upstream_ref": _TEST_UPSTREAM_REF,
            }
        ),
        encoding="utf-8",
    )
    provider = _MaterializingProvider(write_ids=("Calculator",))
    BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    assert provider.run_calls == 1
    refreshed = json.loads(
        (scen / "Calculator" / "scenario.json").read_text(encoding="utf-8")
    )
    assert refreshed["envs"] == list(_ALL_ENVS)


def test_ensure_descriptors_raises_on_export_failure(tmp_path: Path) -> None:
    scen = tmp_path / "scenarios"
    scen.mkdir()
    provider = _MaterializingProvider(returncode=1)
    with pytest.raises(RuntimeError, match="descriptor materialization failed"):
        BaxBenchTaskSource()._ensure_descriptors(provider, scen)
    # No manifest is written on failure, so the next load retries.
    assert not (scen / ".geh_descriptors.json").exists()


def test_task_id_helpers_roundtrip() -> None:
    tid = task_id_for("Calculator", "Python-Flask")
    assert tid == "baxbench/Calculator__Python-Flask"
    assert split_task_id(tid) == ("Calculator", "Python-Flask")
    assert split_task_id("Calculator__Python-Flask") == (
        "Calculator",
        "Python-Flask",
    )


# --- code extraction ---------------------------------------------------


def test_parse_strips_code_tags_from_fence_wrapping_code() -> None:
    """A model may wrap its file in BOTH a markdown fence and the upstream
    ``<CODE>`` sentinel (observed for sonnet-4.5). The fence is extracted
    first, so the residual ``<CODE>``/``</CODE>`` wrapper lines must still be
    stripped -- otherwise they reach the staged file and break compilation
    (Go's ``goimports`` and Rust reject them, deflating every such task to a
    spurious build failure).
    """
    # fence-wrapping-CODE: the real deflation case.
    files = _parse_single_file(
        "```go\n<CODE>\npackage main\nfunc main() {}\n</CODE>\n```", "main.go"
    )
    assert files is not None
    assert "<CODE>" not in files["main.go"]
    assert "</CODE>" not in files["main.go"]
    assert files["main.go"].startswith("package main")
    # bare <CODE> (no fence) stays clean.
    bare = _parse_single_file(
        "<CODE>\npackage main\nfunc main() {}\n</CODE>", "main.go"
    )
    assert "<CODE>" not in bare["main.go"]
    # an inner <CODE> inside a string literal must be preserved.
    lit = _parse_single_file(
        '```go\npackage main\nvar s = "x<CODE>y"\n```', "main.go"
    )
    assert "<CODE>" in lit["main.go"]


# --- staging -----------------------------------------------------------


def test_stage_writes_expected_files(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    artifact = _artifact("Calculator", "Python-Flask")

    staged = oracle.stage([task], [artifact], tmp_path)

    assert isinstance(staged, StagedOracleInput)
    assert staged.adapter_name == "baxbench"
    assert staged.task_ids == ["baxbench/Calculator__Python-Flask"]

    per_task = staged.metadata["per_task"][
        "baxbench/Calculator__Python-Flask"
    ]
    code_dir = Path(per_task["code_dir"])
    # Layout: results/<model>/<scenario>/<env>/<leaf>/sample0/code/app.py
    app = code_dir / "app.py"
    assert app.exists()
    assert app.read_text() == _SAMPLE_FILES["app.py"]
    assert code_dir.name == "code"
    assert code_dir.parent.name == "sample0"
    assert code_dir.parent.parent.name == _SAVE_LEAF
    assert code_dir.parent.parent.parent.name == "Python-Flask"
    assert code_dir.parent.parent.parent.parent.name == "Calculator"
    assert code_dir.parent.parent.parent.parent.parent.name == "byo-model"
    # The path the parser will read is recorded.
    assert per_task["test_results_path"].endswith(
        "sample0/test_results.json"
    )
    assert per_task["scenario"] == "Calculator"
    assert per_task["env"] == "Python-Flask"


def test_stage_rejects_escaping_file_key(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    for bad in ("../escape.py", "/abs/evil.py", "a/../../escape.py"):
        artifact = _artifact("Calculator", "Python-Flask", files={bad: "x"})
        with pytest.raises(ValueError):
            oracle.stage([task], [artifact], tmp_path)


def test_stage_rejects_escaping_model(tmp_path: Path) -> None:
    """A model that climbs out of the results tree is rejected.

    ``esc`` only neutralizes ``/`` (mirroring upstream), so a bare ``..``
    survives as a save-dir component; without confinement it would write to
    the parent of the results root. ``safe_relpath`` must reject it.
    """
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    artifact = _artifact("Calculator", "Python-Flask", model="..")
    with pytest.raises(ValueError):
        oracle.stage([task], [artifact], tmp_path)


def test_stage_slugifies_hf_style_model_in_path(tmp_path: Path) -> None:
    """A normal ``org/model`` is slugified like upstream ``esc`` and confined.

    ``/`` -> ``-`` keeps the model a single in-bounds save-dir component, so
    the path the unmodified upstream harness reconstructs still matches.
    """
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    staged = oracle.stage(
        [task],
        [_artifact("Calculator", "Python-Flask", model="acme/coder")],
        tmp_path,
    )
    code_dir = Path(staged.metadata["per_task"][task.id]["code_dir"])
    results_root = Path(staged.metadata["results_root"]).resolve()
    assert code_dir.resolve().is_relative_to(results_root)
    assert "acme-coder" in code_dir.parts
    # Non-vacuity: the confined path is still written (not just rejected).
    assert code_dir.is_dir() and any(code_dir.iterdir())


def test_stage_rejects_escaping_scenario_from_task_id(tmp_path: Path) -> None:
    """Traversal injected via the task id (scenario/env) is also confined.

    scenario/env are parsed from the candidate's task id, so the same
    ``safe_relpath`` guard must reject a ``..`` arriving through that arm.
    """
    oracle = BaxBenchOracle()
    base = _task("Calculator", "Python-Flask")
    bad_task = base.model_copy(update={"id": "baxbench/..__Python-Flask"})
    artifact = _artifact("..", "Python-Flask")
    with pytest.raises(ValueError):
        oracle.stage([bad_task], [artifact], tmp_path)


def test_stage_clears_stale_sample_dir(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    oracle.stage(
        [task],
        [_artifact("Calculator", "Python-Flask",
                   files={"app.py": "OLD", "stale.py": "x"})],
        tmp_path,
    )
    staged = oracle.stage(
        [task],
        [_artifact("Calculator", "Python-Flask", files={"app.py": "NEW"})],
        tmp_path,
    )
    code_dir = Path(staged.metadata["per_task"][task.id]["code_dir"])
    assert (code_dir / "app.py").read_text() == "NEW"
    assert not (code_dir / "stale.py").exists()


def test_stage_writes_nested_relpaths(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-FastAPI")
    artifact = _artifact(
        "Calculator",
        "Python-FastAPI",
        files={
            "app.py": "x = 1\n",
            "mysite/settings.py": "DEBUG = False\n",
        },
    )
    staged = oracle.stage([task], [artifact], tmp_path)
    code_dir = Path(
        staged.metadata["per_task"][
            "baxbench/Calculator__Python-FastAPI"
        ]["code_dir"]
    )
    assert (code_dir / "app.py").exists()
    nested = code_dir / "mysite" / "settings.py"
    assert nested.exists()
    assert nested.read_text() == "DEBUG = False\n"


def test_stage_rejects_non_full_file_artifact(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    bad = AgentArtifact(
        task_id="baxbench/Calculator__Python-Flask",
        model=_MODEL,
        kind="patch",
        patch="diff --git a/app.py b/app.py\n",
    )
    with pytest.raises(UnsupportedArtifactError):
        oracle.stage([task], [bad], tmp_path)


def test_stage_rejects_unmatched_task(tmp_path: Path) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    orphan = _artifact("Calculator", "Go-Gin")
    with pytest.raises(UnsupportedArtifactError):
        oracle.stage([task], [orphan], tmp_path)


def test_validate_rejects_wrong_kind() -> None:
    # Exposed so the runner scopes an artifact-shape rejection to the offending
    # candidate (one unsupported row) instead of failing the whole batch.
    oracle = BaxBenchOracle()
    wrong_kind = AgentArtifact(
        task_id="baxbench/Calculator__Python-Flask",
        model=_MODEL,
        kind="patch",
        patch="diff --git a/app.py b/app.py\n",
    )
    with pytest.raises(UnsupportedArtifactError):
        oracle.validate(wrong_kind)
    # A well-formed single/multi-file app passes.
    oracle.validate(_artifact("Calculator", "Python-Flask"))


# --- evaluate (stubbed env provider, no Docker) ------------------------


def test_evaluate_builds_expected_argv_and_honors_budget(
    tmp_path: Path,
) -> None:
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    artifact = _artifact("Calculator", "Python-Flask")
    staged = oracle.stage([task], [artifact], tmp_path)

    stub = _StubEnvProvider()
    run_config = OracleRunConfig(run_id="t", run_dir=str(tmp_path))
    # A small budget: max_workers below the oracle's max -> clamped to 2.
    budget = ResourceBudget(max_workers=2)
    raw = oracle.evaluate(staged, run_config, budget, stub)

    assert stub.ensure_ready_called is True
    # Two upstream invocations: test then evaluate.
    assert len(stub.run_calls) == 2
    test_argv, eval_argv = stub.run_calls

    # Drives the real upstream entry point through the provider.
    assert any(a.endswith("src/main.py") for a in test_argv)
    assert "--mode" in test_argv
    assert test_argv[test_argv.index("--mode") + 1] == "test"
    assert "--mode" in eval_argv
    assert eval_argv[eval_argv.index("--mode") + 1] == "evaluate"

    # Scoped to the staged model / scenario / env.
    assert test_argv[test_argv.index("--models") + 1] == "byo-model"
    assert "Calculator" in test_argv
    assert "Python-Flask" in test_argv
    assert "--results_dir" in test_argv
    # Sample index is forwarded to both modes.
    assert test_argv[test_argv.index("--only_samples") + 1] == "0"
    # Budget honored: max_concurrent_runs clamped to the small budget (2).
    assert test_argv[test_argv.index("--max_concurrent_runs") + 1] == "2"
    # The budget object is passed through to the provider's run().
    assert stub.run_budgets[0] is budget

    assert isinstance(raw, RawOracleResult)
    assert raw.task_ids == ["baxbench/Calculator__Python-Flask"]
    assert raw.exit_code == 0
    assert raw.metadata["per_task"]


def test_evaluate_derives_disjoint_port_base_for_shards(
    tmp_path: Path, monkeypatch
) -> None:
    """A sharded run without an explicit GEH_VIBE_PORT_BASE still gets a
    disjoint --min_port per shard (derived from the shard index), so concurrent
    BaxBench shards never collide on the process-local SlotManager's ports."""
    monkeypatch.delenv("GEH_VIBE_PORT_BASE", raising=False)
    monkeypatch.delenv("GEH_VIBE_NUM_PORTS", raising=False)
    monkeypatch.setenv("GEH_VIBE_SHARD", "2/4")  # idx=2
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    artifact = _artifact("Calculator", "Python-Flask")
    staged = oracle.stage([task], [artifact], tmp_path)
    stub = _StubEnvProvider()
    oracle.evaluate(
        staged,
        OracleRunConfig(run_id="t", run_dir=str(tmp_path)),
        ResourceBudget(max_workers=2),
        stub,
    )
    test_argv = stub.run_calls[0]
    # idx=2, default window 1000 -> base = 12345 + 2*1000 = 14345.
    assert "--min_port" in test_argv
    assert test_argv[test_argv.index("--min_port") + 1] == "14345"
    assert test_argv[test_argv.index("--num_ports") + 1] == "1000"


def test_evaluate_unsharded_uses_upstream_default_ports(
    tmp_path: Path, monkeypatch
) -> None:
    """Without sharding (and no explicit base) evaluate omits --min_port, so the
    upstream default port range is used -- no behavior change for single runs."""
    monkeypatch.delenv("GEH_VIBE_PORT_BASE", raising=False)
    monkeypatch.delenv("GEH_VIBE_SHARD", raising=False)
    oracle = BaxBenchOracle()
    task = _task("Calculator", "Python-Flask")
    artifact = _artifact("Calculator", "Python-Flask")
    staged = oracle.stage([task], [artifact], tmp_path)
    stub = _StubEnvProvider()
    oracle.evaluate(
        staged,
        OracleRunConfig(run_id="t", run_dir=str(tmp_path)),
        ResourceBudget(max_workers=2),
        stub,
    )
    assert "--min_port" not in stub.run_calls[0]


def test_parallelism_declares_batch_internal() -> None:
    oracle = BaxBenchOracle()
    assert oracle.granularity == "batch"
    assert oracle.parallelism == OracleParallelism(
        model="batch_internal",
        default_workers=4,
        max_workers=10,
    )
    assert oracle.parallelism.model == "batch_internal"


# --- capabilities ------------------------------------------------------


def test_capabilities_exact() -> None:
    assert BaxBenchOracle.capabilities == OracleCapabilities(
        runs_functional_tests=True,
        detects_target_vuln=True,
        detects_new_vuln=False,
        dynamic_pov=True,
        static_analysis=False,
        fuzzing=False,
        llm_judge=False,
        deterministic=False,
    )
    assert BaxBenchOracle.task_types == {"project_scaffold"}
    assert BaxBenchOracle.artifact_kinds == {"full_file"}


# --- parse: per-case mapping -------------------------------------------


def test_parse_secure_is_target_secure(tmp_path: Path) -> None:
    row = _parse_one(tmp_path, "secure")
    assert row.status == "completed"
    assert row.failure_origin == "none"
    assert row.failure_reason is None
    assert row.build_pass is True
    assert row.functional_pass is True
    assert row.security_oracle_pass is True
    assert row.known_vuln_present is False
    # Functional pass AND security pass -> target-secure True.
    assert row.target_secure_success is True
    # BaxBench has no new-vuln signal -> strict-secure stays None.
    assert row.new_vuln_introduced is None
    assert row.strict_secure_success is None
    assert row.raw.extra["scenario"] == "Calculator"
    assert row.raw.extra["env"] == "Python-Flask"


def test_parse_insecure_target_vuln_present(tmp_path: Path) -> None:
    row = _parse_one(tmp_path, "insecure")
    # App built + ran, all functional pass, but an exploit flagged a CWE:
    # the model is insecure, not a pipeline failure.
    assert row.status == "completed"
    assert row.build_pass is True
    assert row.functional_pass is True
    assert row.security_oracle_pass is False
    assert row.known_vuln_present is True
    assert row.failure_reason == "target_vuln_present"
    # target-secure is False (functional True, security False).
    assert row.target_secure_success is False
    # Kleene AND: security False already makes strict a definite False even
    # though the new-vuln gate is None (False dominates AND over unknowns).
    assert row.strict_secure_success is False
    assert row.raw.extra["cwes"] == [94]


def test_parse_build_failed_is_model_failure(tmp_path: Path) -> None:
    row = _parse_one(tmp_path, "build_failed")
    assert row.status == "model_failure"
    assert row.failure_origin == "model"
    assert row.failure_reason == "build_failed"
    assert row.build_pass is False
    assert row.patch_applied is False
    # The app never built: a definite functional FAILURE, so the row is an
    # in-denominator target-secure fail (not an excluded None). Security stays
    # None (no security test could run); Kleene False dominates the AND.
    assert row.functional_pass is False
    assert row.security_oracle_pass is None
    assert row.target_secure_success is False
    assert row.strict_secure_success is False


def test_parse_functional_fail_but_secure(tmp_path: Path) -> None:
    row = _parse_one(tmp_path, "func_fail")
    assert row.status == "completed"
    assert row.build_pass is True
    assert row.functional_pass is False
    assert row.security_oracle_pass is True
    assert row.failure_reason == "functional_tests_failed"
    # functional False -> target-secure False.
    assert row.target_secure_success is False
    # Kleene AND: functional False makes strict a definite False even though
    # the new-vuln gate is None (False dominates AND over unknowns).
    assert row.strict_secure_success is False


def test_parse_missing_results_is_infra_failure(tmp_path: Path) -> None:
    # The sample dir exists (code present) but NO test_results.json and no
    # ``code/failed`` marker -> the oracle never scored it -> INFRA, never a
    # fabricated model verdict.
    row = _parse_one(tmp_path, "infra_missing")
    assert row.status == "infra_failure"
    assert row.failure_origin == "infra"
    assert row.failure_reason == "verifier_unavailable"
    assert row.build_pass is None
    assert row.functional_pass is None
    assert row.security_oracle_pass is None
    assert row.target_secure_success is None


def test_parse_missing_results_timeout_is_infra_timeout(
    tmp_path: Path,
) -> None:
    # Same missing-results case, but the whole run timed out -> infra timeout
    # (still infra, not model).
    row = _parse_one(tmp_path, "infra_missing", timed_out=True)
    assert row.status == "infra_failure"
    assert row.failure_origin == "infra"
    assert row.failure_reason == "oracle_timeout"
    assert row.raw.upstream_status == "timeout"


def test_parse_code_failed_marker_is_model_failure(tmp_path: Path) -> None:
    # A ``code/failed`` generation-failure marker (no test_results.json) is a
    # MODEL build failure, distinct from the infra case above.
    sample_dir = tmp_path / "sample0"
    code_dir = sample_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "failed").write_text("generation failed\n")
    task_id = task_id_for("Calculator", "Python-Flask")
    raw = RawOracleResult(
        adapter_name="baxbench",
        outputs_dir=str(tmp_path),
        logs_dir=None,
        task_ids=[task_id],
        metadata={
            "results_root": str(tmp_path),
            "per_task": {
                task_id: {
                    "scenario": "Calculator",
                    "env": "Python-Flask",
                    "model": _MODEL,
                    "sample_dir": str(sample_dir),
                    "code_dir": str(code_dir),
                    "test_results_path": str(
                        sample_dir / "test_results.json"
                    ),
                    "source_dataset": "baxbench",
                }
            },
        },
    )
    rows = BaxBenchOracle().parse(raw)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "model_failure"
    assert row.failure_origin == "model"
    assert row.failure_reason == "build_failed"
    assert row.build_pass is False
    assert row.patch_applied is False


# --- metric / nullity invariants ---------------------------------------


def test_strict_secure_never_true_no_new_vuln_signal(
    tmp_path: Path,
) -> None:
    # BaxBench never produces a new-vuln signal (it checks only target
    # exploits), so new_vuln_introduced is always None and strict_secure_success
    # can never be True. Under Kleene AND it is None only while no other gate has
    # definitely failed; once functional or security is False, strict is a
    # definite False (False dominates AND over the unknown new-vuln gate).
    for case in ("secure", "insecure", "func_fail", "build_failed"):
        row = _parse_one(tmp_path, case)
        assert row.new_vuln_introduced is None, case
        assert row.strict_secure_success is not True, case
        if row.functional_pass is False or row.security_oracle_pass is False:
            assert row.strict_secure_success is False, case
        else:
            assert row.strict_secure_success is None, case
        assert row.oracle_capabilities.detects_new_vuln is False, case


def test_secure_row_metric_rederives_stably(tmp_path: Path) -> None:
    row = _parse_one(tmp_path, "secure")
    rederived = derive_task_metrics(row.model_copy(deep=True))
    assert rederived.target_secure_success is True
    assert rederived.strict_secure_success is None
