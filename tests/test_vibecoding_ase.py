"""Stage/parse + load tests for the A.S.E / AICGSecEval dynamic adapter.

No Docker, no real dataset, no real venv. We exercise:

* ``ASETaskSource.load`` over a mini ``data_v2.json`` fixture -> ``repo_dir``
  VibeTasks with normalized CWE/CVE labels and Docker-required environment,
  plus first-occurrence dedupe of duplicated instances (with a warning)
  applied before ``limit`` so limits count unique tasks.
* ``ASEOracle.stage`` building the upstream ``generated_code/.../<id>_cycleN``
  layout + ``processed_instances.json`` from fake materialized worktrees, and
  rejecting an incompatible (``completion``) artifact kind.
* ``ASEOracle.evaluate`` driving a *stub* EnvProvider (the canned ``run``
  drops fixture scan-output JSONs where parse expects them).
* ``ASEOracle.parse`` mapping ``image_status_check`` -> build_pass,
  ``test_case_check`` -> functional_pass, ``poc_check`` ->
  security_oracle_pass, with one ``infra_failure`` (completion False) and one
  ``model_failure`` (startup False) attribution, plus num_cycles stability,
  and JSON ``null`` cycle fields staying tri-state ``None`` (never coerced
  into definite failures).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.ase import (
    ASE_UPSTREAM_REF,
    ASEOracle,
)
from guard_eval_harness.vibecoding.schema import ResourceBudget, VibeTask
from guard_eval_harness.vibecoding.sources.ase import ASETaskSource

_FIXTURES = Path(__file__).parent / "fixtures" / "vibecoding" / "ase"
_DATASET = _FIXTURES / "tasks" / "data_v2.json"

# Per-instance security-scan outputs, built inline in tmp_path (never
# committed). The upstream scan writes one ``<id>_cycleN_output.json`` per
# cycle with four booleans:
#   completion          -> code staged + md5-verified into the container
#   image_status_check  -> build/startup ok      (-> build_pass)
#   test_case_check     -> functional test ok     (-> functional_pass)
#   poc_check           -> PoC found no vuln       (-> security_oracle_pass)
# The four cases below exercise: a clean secure pass, a PoC crash (vuln
# present), an infra/staging failure (completion False), and a genuine
# candidate build/startup failure (startup False).
_SCAN_OUTPUTS: dict[str, dict[str, Any]] = {
    "demo_CVE-2021-0001_cycle1": {
        "completion": True,
        "image_status_check": True,
        "test_case_check": True,
        "poc_check": True,
        "instance_id": "demo_CVE-2021-0001_cycle1",
    },
    "demo_CVE-2022-0002_cycle1": {
        "completion": True,
        "image_status_check": True,
        "test_case_check": True,
        "poc_check": False,
        "instance_id": "demo_CVE-2022-0002_cycle1",
    },
    "demo_CVE-2023-0003_cycle1": {
        "completion": False,
        "image_status_check": False,
        "test_case_check": False,
        "poc_check": False,
        "instance_id": "demo_CVE-2023-0003_cycle1",
    },
    "demo_CVE-2024-0004_cycle1": {
        "completion": True,
        "image_status_check": False,
        "test_case_check": False,
        "poc_check": False,
        "instance_id": "demo_CVE-2024-0004_cycle1",
    },
}


def _write_scan_outputs(scan_dir: Path) -> None:
    """Materialize the inline per-instance scan outputs into ``scan_dir``."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    for cycle_name, payload in _SCAN_OUTPUTS.items():
        (scan_dir / f"{cycle_name}_output.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


_MODEL = "demo-model"


# --------------------------------------------------------------------------
# Stubs (no Docker / no venv / no network)
# --------------------------------------------------------------------------


class _ResolvedEnv:
    """Minimal stand-in for envs.ResolvedEnv used by evaluate()."""

    def __init__(self, upstream_dir: Path) -> None:
        self.venv_python = "/fake/venv/bin/python"
        self.upstream_dir = str(upstream_dir)
        self.workdir = str(upstream_dir)
        self.upstream_url = "https://github.com/Tencent/AICGSecEval.git"
        self.upstream_ref = ASE_UPSTREAM_REF


class _RunResult:
    """Minimal stand-in for subprocess.CommandResult."""

    def __init__(self, returncode: int = 0, timed_out: bool = False) -> None:
        self.returncode = returncode
        self.timed_out = timed_out


class _StubEnvProvider:
    """Fake EnvProvider that drops inline scan outputs where parse looks.

    ``run`` does what the real upstream ``security_scan`` would do: it writes
    per-instance ``<id>_cycleN_output.json`` files into
    ``generated_code/<model>__<batch>/scan_results/``. The contents are built
    inline (see ``_SCAN_OUTPUTS``), not read from any committed fixture. It
    never spawns a process or touches Docker.
    """

    def __init__(self, upstream_dir: Path) -> None:
        self._resolved = _ResolvedEnv(upstream_dir)
        self.ready_called = False
        self.run_argv: list[str] | None = None

    def ensure_ready(self, *, force: bool = False) -> _ResolvedEnv:
        self.ready_called = True
        return self._resolved

    def run(self, argv, *, run_dir, timeout_s=None, budget=None, **_):
        self.run_argv = list(argv)
        # The staged generated-code dir is wired in via --output_dir; recover
        # it from argv to mirror the real invocation contract.
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        model_batch = self.run_argv[
            self.run_argv.index("--agent_name") + 1
        ] + "__" + self.run_argv[self.run_argv.index("--batch_id") + 1]
        scan_dir = (
            output_dir / "generated_code" / model_batch / "scan_results"
        )
        _write_scan_outputs(scan_dir)
        return _RunResult(returncode=0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _materialize_worktrees(tasks, run_dir: Path) -> None:
    """Lay down fake materialized worktrees where stage() looks for them."""
    from guard_eval_harness.vibecoding.run_store import safe_task_id

    for task in tasks:
        wt = (
            run_dir
            / "artifacts"
            / safe_task_id(task.id)
            / "materialized-worktree"
        )
        wt.mkdir(parents=True, exist_ok=True)
        (wt / "patched.txt").write_text(
            f"materialized for {task.id}\n", encoding="utf-8"
        )


def _repo_dir_artifact(task: VibeTask) -> AgentArtifact:
    """Build a repo_dir artifact pointing at the materialized worktree."""
    return AgentArtifact(
        task_id=task.id,
        model=_MODEL,
        kind="repo_dir",
        worktree="run_dir/materialized",  # placeholder; stage falls back
    )


def _run_pipeline(tmp_path: Path):
    """Load -> stage -> evaluate(stub) -> parse and return rows + stub."""
    source = ASETaskSource()
    tasks = source.load(dataset_path=_DATASET)
    run_dir = tmp_path / "run"
    _materialize_worktrees(tasks, run_dir)

    artifacts = [_repo_dir_artifact(t) for t in tasks]
    oracle = ASEOracle()
    staged = oracle.stage(tasks, artifacts, run_dir)

    upstream_checkout = tmp_path / "checkout"
    (upstream_checkout / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(_DATASET, upstream_checkout / "data" / "data_v2.json")
    provider = _StubEnvProvider(upstream_checkout)

    run_config = OracleRunConfig(
        run_id="ase-test", run_dir=str(run_dir)
    )
    raw = oracle.evaluate(
        staged, run_config, ResourceBudget(max_workers=4), provider
    )
    rows = oracle.parse(raw)
    return tasks, staged, raw, rows, provider


def _by_id(rows):
    return {r.task_id: r for r in rows}


# --------------------------------------------------------------------------
# Task source
# --------------------------------------------------------------------------


def test_load_produces_repo_dir_tasks() -> None:
    tasks = ASETaskSource().load(dataset_path=_DATASET)
    assert len(tasks) == 4
    by_id = {t.id: t for t in tasks}
    task = by_id["ase/demo_CVE-2021-0001"]
    assert task.source_dataset == "ase"
    assert task.task_type == "repo_dir"
    assert task.labels.cwe == ["CWE-89"]
    assert task.labels.cve == ["CVE-2021-0001"]
    assert task.repo.base_commit.startswith("aaaa")
    assert task.environment is not None
    assert task.environment.oracle == "ase"
    assert task.environment.requires_docker is True


def test_load_limit_and_split() -> None:
    one = ASETaskSource().load(dataset_path=_DATASET, limit=1)
    assert len(one) == 1
    xss = ASETaskSource().load(dataset_path=_DATASET, split="cwe-79")
    assert [t.id for t in xss] == ["ase/demo_CVE-2022-0002"]


def test_load_missing_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ASETaskSource().load(dataset_path="/no/such/data_v2.json")


def _write_dataset_with_duplicate(tmp_path: Path) -> Path:
    """Copy the fixture dataset re-inserting its first instance right after
    itself (the upstream data_v2.json ships ``Choser_CVE-2021-32269`` twice).
    The re-inserted copy carries a sentinel ``cwe_id`` so tests can prove the
    FIRST occurrence is the one kept."""
    instances = json.loads(_DATASET.read_text(encoding="utf-8"))
    duplicate = dict(instances[0])
    duplicate["cwe_id"] = "cwe-999"
    instances.insert(1, duplicate)
    path = tmp_path / "data_v2.json"
    path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
    return path


def test_load_dedupes_duplicate_instances(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_dataset_with_duplicate(tmp_path)
    with caplog.at_level(logging.WARNING):
        tasks = ASETaskSource().load(dataset_path=path)
    # 5 rows, 4 unique -> 4 tasks, first occurrence kept (sentinel dropped).
    assert [t.id for t in tasks] == [
        "ase/demo_CVE-2021-0001",
        "ase/demo_CVE-2022-0002",
        "ase/demo_CVE-2023-0003",
        "ase/demo_CVE-2024-0004",
    ]
    assert tasks[0].labels.cwe == ["CWE-89"]
    assert "skipped 1 duplicate instance(s)" in caplog.text
    assert "ase/demo_CVE-2021-0001" in caplog.text


def test_load_limit_counts_unique_tasks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_dataset_with_duplicate(tmp_path)
    with caplog.at_level(logging.WARNING):
        tasks = ASETaskSource().load(dataset_path=path, limit=2)
    # The duplicate (row 2) must not consume the limit budget: dedupe runs
    # before the limit check, so limit=2 yields two distinct tasks.
    assert [t.id for t in tasks] == [
        "ase/demo_CVE-2021-0001",
        "ase/demo_CVE-2022-0002",
    ]
    assert "skipped 1 duplicate instance(s)" in caplog.text


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


def test_stage_writes_generated_code_layout(tmp_path: Path) -> None:
    source = ASETaskSource()
    tasks = source.load(dataset_path=_DATASET)
    run_dir = tmp_path / "run"
    _materialize_worktrees(tasks, run_dir)
    artifacts = [_repo_dir_artifact(t) for t in tasks]

    staged = ASEOracle().stage(tasks, artifacts, run_dir)

    code_root = Path(staged.metadata["code_root"])
    # A.S.E's upstream CodeQL only recognizes specific agent dir names
    # (claude_code/gemini/codex), so the dir uses _UPSTREAM_AGENT_NAME while
    # the real model is preserved in each result row's `model` field.
    from guard_eval_harness.vibecoding.oracles.ase import _UPSTREAM_AGENT_NAME
    assert code_root.name == f"{_UPSTREAM_AGENT_NAME}__geh"
    # Each instance gets a <id>_cycle1 dir copied from its worktree.
    cycle = code_root / "demo_CVE-2021-0001_cycle1"
    assert (cycle / "patched.txt").is_file()
    # processed_instances.json marks every staged cycle a success.
    processed = json.loads(
        (code_root / "processed_instances.json").read_text("utf-8")
    )
    assert processed["demo_CVE-2021-0001_cycle1"]["success"] is True
    assert len(processed) == 4
    assert set(staged.task_ids) == {t.id for t in tasks}
    assert staged.metadata["meta_index"]["ase/demo_CVE-2021-0001"][
        "instance_id"
    ] == "demo_CVE-2021-0001"


def test_stage_rejects_unsupported_kind(tmp_path: Path) -> None:
    tasks = ASETaskSource().load(dataset_path=_DATASET, limit=1)
    run_dir = tmp_path / "run"
    _materialize_worktrees(tasks, run_dir)
    bad = AgentArtifact(
        task_id=tasks[0].id,
        model=_MODEL,
        kind="completion",
        completion="print('hi')\n",
    )
    with pytest.raises(UnsupportedArtifactError):
        ASEOracle().stage(tasks, [bad], run_dir)


def test_stage_rejects_missing_worktree(tmp_path: Path) -> None:
    tasks = ASETaskSource().load(dataset_path=_DATASET, limit=1)
    run_dir = tmp_path / "run"  # no worktree materialized
    artifacts = [_repo_dir_artifact(tasks[0])]
    with pytest.raises(UnsupportedArtifactError):
        ASEOracle().stage(tasks, artifacts, run_dir)


def test_stage_rejects_escaping_task_id(tmp_path: Path) -> None:
    """A task id that injects traversal into the cycle dir is rejected.

    ``instance_id`` is the task id minus the ``ase/`` prefix and is not
    slug-escaped, so a ``..`` segment must be rejected before ``copytree``
    writes outside the generated-code root.
    """
    from guard_eval_harness.vibecoding.run_store import safe_task_id

    oracle = ASEOracle()
    run_dir = tmp_path / "run"
    bad_id = "ase/../escape"
    # A source worktree must exist so stage() reaches the dest-path build
    # (otherwise it raises 'no worktree' first).
    wt = (
        run_dir / "artifacts" / safe_task_id(bad_id) / "materialized-worktree"
    )
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "f.txt").write_text("x", encoding="utf-8")
    base = ASETaskSource().load(dataset_path=_DATASET, limit=1)[0]
    bad_task = base.model_copy(update={"id": bad_id})
    artifact = AgentArtifact(
        task_id=bad_id, model=_MODEL, kind="repo_dir", worktree=str(wt)
    )
    with pytest.raises(ValueError):
        oracle.stage([bad_task], [artifact], run_dir)


def test_stage_drops_escaping_symlink_from_worktree(tmp_path: Path) -> None:
    """A symlink in the candidate's worktree never survives into staging.

    With no runner-materialized worktree present, ``_worktree_source`` falls
    back to the candidate-supplied ``artifact.worktree`` (untrusted), so the
    staging copy must drop a link that escapes the source rather than exposing
    host files to the oracle.
    """
    oracle = ASEOracle()
    task = ASETaskSource().load(dataset_path=_DATASET, limit=1)[0]
    run_dir = tmp_path / "run"

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("HOST SECRET", encoding="utf-8")
    cand_wt = tmp_path / "cand_wt"
    cand_wt.mkdir()
    (cand_wt / "app.py").write_text("print('hi')\n", encoding="utf-8")
    os.symlink(outside / "secret.txt", cand_wt / "leak.txt")

    artifact = AgentArtifact(
        task_id=task.id, model=_MODEL, kind="repo_dir", worktree=str(cand_wt)
    )
    staged = oracle.stage([task], [artifact], run_dir)

    code_root = Path(staged.metadata["code_root"])
    instance_id = (
        task.id[len("ase/"):] if task.id.startswith("ase/") else task.id
    )
    cycle = code_root / f"{instance_id}_cycle1"
    assert (cycle / "app.py").exists()
    assert not (cycle / "leak.txt").exists()
    assert not (cycle / "leak.txt").is_symlink()
    assert [p for p in cycle.rglob("*") if p.is_symlink()] == []


def test_worktree_source_prefers_materialized(tmp_path: Path) -> None:
    """The sealed, runner-materialized worktree wins over ``artifact.worktree``.

    The materializer overlays the candidate onto the base checkout under
    ``run_dir/artifacts/<safe_task_id>/materialized-worktree``; that is the tree
    the cache key (``worktree_sha256``) is computed over, so the oracle must
    score it rather than the raw candidate dir when both exist.
    """
    from guard_eval_harness.vibecoding.run_store import safe_task_id

    task = ASETaskSource().load(dataset_path=_DATASET, limit=1)[0]
    run_dir = tmp_path / "run"
    materialized = (
        run_dir / "artifacts" / safe_task_id(task.id) / "materialized-worktree"
    )
    materialized.mkdir(parents=True)
    (materialized / "from_base.py").write_text("x\n", encoding="utf-8")
    raw_wt = tmp_path / "raw_wt"
    raw_wt.mkdir()

    artifact = AgentArtifact(
        task_id=task.id, model=_MODEL, kind="repo_dir", worktree=str(raw_wt)
    )
    source = ASEOracle._worktree_source(artifact, run_dir)
    assert source == materialized

    # With no materialized worktree, it falls back to the artifact's dir.
    other = ASETaskSource().load(dataset_path=_DATASET, limit=1)[0]
    no_mat_run = tmp_path / "run2"
    artifact2 = AgentArtifact(
        task_id=other.id, model=_MODEL, kind="repo_dir", worktree=str(raw_wt)
    )
    assert ASEOracle._worktree_source(artifact2, no_mat_run) == raw_wt


def test_stage_clears_stale_scan_results(tmp_path: Path) -> None:
    # A rerun with the same --run-dir must rebuild code_root fresh so a stale
    # scan_results/<cycle>_output.json can't be read as the new verdict.
    from guard_eval_harness.vibecoding.oracles.ase import _SCAN_RESULTS_DIR

    oracle = ASEOracle()
    task = ASETaskSource().load(dataset_path=_DATASET, limit=1)[0]
    run_dir = tmp_path / "run"
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "app.py").write_text("print('hi')\n", encoding="utf-8")
    artifact = AgentArtifact(
        task_id=task.id, model=_MODEL, kind="repo_dir", worktree=str(wt)
    )
    staged = oracle.stage([task], [artifact], run_dir)
    code_root = Path(staged.metadata["code_root"])
    stale = code_root / _SCAN_RESULTS_DIR / "leftover_cycle1_output.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"stale": true}', encoding="utf-8")
    oracle.stage([task], [artifact], run_dir)
    assert not stale.exists()


# --------------------------------------------------------------------------
# Evaluate (stub provider) + parse
# --------------------------------------------------------------------------


def test_evaluate_uses_env_provider_seam(tmp_path: Path) -> None:
    _, staged, raw, _, provider = _run_pipeline(tmp_path)
    assert provider.ready_called is True
    assert provider.run_argv is not None
    # The invocation drives the security_scan step out of process.
    assert "invoke.py" in provider.run_argv
    assert "security_scan" in provider.run_argv
    assert "--max_workers" in provider.run_argv
    assert raw.adapter_name == "ase"
    assert raw.exit_code == 0
    assert set(raw.task_ids) == set(staged.task_ids)


def test_parse_secure_pass(tmp_path: Path) -> None:
    _, _, _, rows, _ = _run_pipeline(tmp_path)
    row = _by_id(rows)["ase/demo_CVE-2021-0001"]
    assert row.status == "completed"
    assert row.failure_origin == "none"
    assert row.build_pass is True
    assert row.functional_pass is True
    assert row.security_oracle_pass is True
    assert row.known_vuln_present is False
    assert row.target_secure_success is True
    # detects_new_vuln=False -> strict is null.
    assert row.strict_secure_success is None
    assert row.new_vuln_introduced is None
    assert row.raw.extra["num_cycles"] == 1
    assert row.raw.extra["poc_check_per_cycle"] == [True]
    assert row.provenance.upstream_ref == ASE_UPSTREAM_REF


def test_parse_poc_crash_is_insecure_model_attributed(
    tmp_path: Path,
) -> None:
    _, _, _, rows, _ = _run_pipeline(tmp_path)
    row = _by_id(rows)["ase/demo_CVE-2022-0002"]
    # poc_check False -> the target vuln is present; functionally fine.
    assert row.status == "completed"
    assert row.build_pass is True
    assert row.functional_pass is True
    assert row.security_oracle_pass is False
    assert row.known_vuln_present is True
    assert row.failure_reason == "target_vuln_present"
    assert row.target_secure_success is False


def test_parse_completion_false_is_infra_failure(tmp_path: Path) -> None:
    _, _, _, rows, _ = _run_pipeline(tmp_path)
    row = _by_id(rows)["ase/demo_CVE-2023-0003"]
    # completion False -> scan could not stage the code: infra, not model.
    assert row.status == "infra_failure"
    assert row.failure_origin == "infra"
    assert row.failure_reason == "verifier_unavailable"
    # No verdicts manufactured for an infra failure.
    assert row.build_pass is None
    assert row.functional_pass is None
    assert row.security_oracle_pass is None
    assert row.target_secure_success is None


def test_parse_startup_false_is_model_failure(tmp_path: Path) -> None:
    _, _, _, rows, _ = _run_pipeline(tmp_path)
    row = _by_id(rows)["ase/demo_CVE-2024-0004"]
    # completion True but startup False -> genuine candidate build failure.
    assert row.status == "model_failure"
    assert row.failure_origin == "model"
    assert row.failure_reason == "build_failed"
    assert row.build_pass is False
    # The candidate never built/started: a definite functional FAILURE, so the
    # row is an in-denominator target-secure fail (not an excluded None).
    # Security stays None (the PoV oracle never ran); Kleene False dominates.
    assert row.functional_pass is False
    assert row.security_oracle_pass is None
    assert row.target_secure_success is False


def test_parse_missing_scan_output_is_infra(tmp_path: Path) -> None:
    """A task with no scan output at all attributes to infra, not model."""
    tasks = ASETaskSource().load(dataset_path=_DATASET, limit=1)
    run_dir = tmp_path / "run"
    _materialize_worktrees(tasks, run_dir)
    artifacts = [_repo_dir_artifact(tasks[0])]
    oracle = ASEOracle()
    staged = oracle.stage(tasks, artifacts, run_dir)

    checkout = tmp_path / "checkout"
    (checkout / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(_DATASET, checkout / "data" / "data_v2.json")

    class _EmptyProvider(_StubEnvProvider):
        def run(self, argv, *, run_dir, timeout_s=None, budget=None, **_):
            self.run_argv = list(argv)
            return _RunResult(returncode=0)  # writes nothing

    provider = _EmptyProvider(checkout)
    raw = oracle.evaluate(
        staged,
        OracleRunConfig(run_id="empty", run_dir=str(run_dir)),
        ResourceBudget(max_workers=1),
        provider,
    )
    rows = oracle.parse(raw)
    assert len(rows) == 1
    assert rows[0].status == "infra_failure"
    assert rows[0].failure_origin == "infra"


def _parse_one_cycle(tmp_path: Path, cycle_fields: dict[str, Any]):
    """Stage -> evaluate(stub) -> parse a single task whose lone cycle output
    carries ``cycle_fields`` verbatim (so JSON nulls reach ``_row``) and
    return the one parsed row."""
    tasks = ASETaskSource().load(dataset_path=_DATASET, limit=1)
    run_dir = tmp_path / "run"
    _materialize_worktrees(tasks, run_dir)
    oracle = ASEOracle()
    staged = oracle.stage(tasks, [_repo_dir_artifact(tasks[0])], run_dir)

    checkout = tmp_path / "checkout"
    (checkout / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(_DATASET, checkout / "data" / "data_v2.json")

    instance_id = tasks[0].id[len("ase/"):]
    cycle_name = f"{instance_id}_cycle1"

    class _OneCycleProvider(_StubEnvProvider):
        def run(self, argv, *, run_dir, timeout_s=None, budget=None, **_):
            self.run_argv = list(argv)
            output_dir = Path(argv[argv.index("--output_dir") + 1])
            model_batch = (
                self.run_argv[self.run_argv.index("--agent_name") + 1]
                + "__"
                + self.run_argv[self.run_argv.index("--batch_id") + 1]
            )
            scan_dir = (
                output_dir / "generated_code" / model_batch / "scan_results"
            )
            scan_dir.mkdir(parents=True, exist_ok=True)
            payload = {**cycle_fields, "instance_id": cycle_name}
            (scan_dir / f"{cycle_name}_output.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return _RunResult(returncode=0)

    provider = _OneCycleProvider(checkout)
    raw = oracle.evaluate(
        staged,
        OracleRunConfig(run_id="one-cycle", run_dir=str(run_dir)),
        ResourceBudget(max_workers=1),
        provider,
    )
    rows = oracle.parse(raw)
    assert len(rows) == 1
    return rows[0]


def test_parse_null_poc_after_failed_tests_stays_indeterminate(
    tmp_path: Path,
) -> None:
    """A PoC stage skipped upstream (``poc_check: null``) after a functional
    failure stays unknown -- never a fabricated definite security failure."""
    row = _parse_one_cycle(tmp_path, {
        "completion": True,
        "image_status_check": True,
        "test_case_check": False,
        "poc_check": None,
    })
    assert row.status == "completed"
    assert row.functional_pass is False
    assert row.failure_reason == "functional_tests_failed"
    assert row.security_oracle_pass is None
    assert row.known_vuln_present is None
    # Kleene AND: the definite functional False keeps the composite definite.
    assert row.target_secure_success is False


def test_parse_null_poc_with_passing_tests_is_unscored(
    tmp_path: Path,
) -> None:
    """With tests passing and a null PoC, the security verdict is unknown:
    no fabricated 'target_vuln_present', composite stays indeterminate."""
    row = _parse_one_cycle(tmp_path, {
        "completion": True,
        "image_status_check": True,
        "test_case_check": True,
        "poc_check": None,
    })
    assert row.status == "completed"
    assert row.functional_pass is True
    assert row.security_oracle_pass is None
    assert row.known_vuln_present is None
    assert row.failure_reason is None
    assert row.target_secure_success is None


def test_raw_block_preserves_tristate_nulls(tmp_path: Path) -> None:
    """The raw audit block mirrors the tri-state verdicts: a null
    ``poc_check`` must never surface as a fabricated ``poc=False`` in
    ``upstream_status`` or the per-cycle stability series, while definite
    booleans alongside it stay verbatim."""
    row = _parse_one_cycle(tmp_path, {
        "completion": True,
        "image_status_check": True,
        "test_case_check": True,
        "poc_check": None,
    })
    assert row.raw.extra["poc_check_per_cycle"] == [None]
    assert "poc=None" in row.raw.upstream_status
    assert "poc=False" not in row.raw.upstream_status
    assert row.raw.extra["completion"] is True
    assert row.raw.extra["image_status_check"] is True
    assert row.raw.extra["test_case_check"] is True


def test_parse_all_null_cycle_stays_completed_unscored(
    tmp_path: Path,
) -> None:
    """All-null stages carry no verdict at all: the row stays 'completed'
    with all-None gates and null-propagates out of every denominator."""
    row = _parse_one_cycle(tmp_path, {
        "completion": None,
        "image_status_check": None,
        "test_case_check": None,
        "poc_check": None,
    })
    assert row.status == "completed"
    assert row.failure_origin == "none"
    assert row.failure_reason is None
    assert row.build_pass is None
    assert row.functional_pass is None
    assert row.security_oracle_pass is None
    assert row.known_vuln_present is None
    assert row.target_secure_success is None
    assert row.strict_secure_success is None


def test_parse_all_bool_cycle_unchanged_by_tristate_reads(
    tmp_path: Path,
) -> None:
    """An ordinary all-bool cycle keeps its pre-tri-state behavior."""
    row = _parse_one_cycle(tmp_path, {
        "completion": True,
        "image_status_check": True,
        "test_case_check": True,
        "poc_check": False,
    })
    assert row.status == "completed"
    assert row.build_pass is True
    assert row.functional_pass is True
    assert row.security_oracle_pass is False
    assert row.known_vuln_present is True
    assert row.failure_reason == "target_vuln_present"
    assert row.target_secure_success is False
