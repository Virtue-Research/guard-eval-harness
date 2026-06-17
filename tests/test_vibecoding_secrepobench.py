"""Stage/parse conformance tests for the SecRepoBench oracle adapter.

These run with NO Docker and NO real dataset: the task source reads mini
fixture metadata, ``stage`` writes upstream completion files from fixture
artifacts, and ``parse`` is driven off the upstream eval OUTPUT report that is
built inline in ``tmp_path`` (see ``_write_report_eval``) plus the kept
ground-truth baseline ``report.json`` INPUT fixture, via a stubbed env provider
that returns a canned :class:`RawOracleResult`.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.secrepobench import (
    SecRepoBenchOracle,
    _completion_filename,
)
from guard_eval_harness.vibecoding.results import derive_task_metrics
from guard_eval_harness.vibecoding.schema import ResourceBudget
from guard_eval_harness.vibecoding.sources.secrepobench import (
    SecRepoBenchTaskSource,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "vibecoding" / "secrepobench"
_TASKS = _FIXTURES / "tasks"
_RAW = _FIXTURES / "raw"


# The upstream eval OUTPUT is built inline (no committed fixture): three tasks
# exercising the pass / crash / compile-error verdicts, with unit-test pass-sets
# chosen so the baseline (report.json INPUT) is a subset (functional_pass=True).
_REPORT_EVAL: dict[str, dict] = {
    "910": {
        "none": {
            "gpt-test": {
                "full": {
                    "instruct": {
                        "perturbed": {
                            "testcase": "pass",
                            "unittest": {
                                "pass": [
                                    "Base types",
                                    "endianess",
                                    "quick floor",
                                ],
                                "fail": [],
                                "skip": [],
                                "total": 3,
                            },
                        }
                    }
                }
            }
        }
    },
    "1065": {
        "none": {
            "gpt-test": {
                "full": {
                    "instruct": {
                        "perturbed": {
                            "testcase": "crash",
                            "unittest": {
                                "pass": ["ascii test", "compress test"],
                                "fail": [],
                                "skip": [],
                                "total": 2,
                            },
                        }
                    }
                }
            }
        }
    },
    "1427": {
        "none": {
            "gpt-test": {
                "full": {
                    "instruct": {
                        "perturbed": {
                            "testcase": "error: compile error (1)",
                            "unittest": {
                                "pass": [],
                                "fail": [],
                                "skip": [],
                                "total": 0,
                            },
                        }
                    }
                }
            }
        }
    },
}


def _write_report_eval(directory: Path) -> Path:
    """Write the inline upstream eval OUTPUT report into ``directory``.

    Replaces the previously committed ``report_eval.json`` output fixture: the
    content is reproduced here as a Python literal so the parse() assertions
    (pass/crash/compile-error verdicts, secure-pass@1, unit-test baseline
    subset) hold without shipping an upstream-eval-output file in the repo.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report_eval.json"
    path.write_text(json.dumps(_REPORT_EVAL), encoding="utf-8")
    return path


# --- task source ---------------------------------------------------------


def _source() -> SecRepoBenchTaskSource:
    return SecRepoBenchTaskSource(
        metadata_path=_TASKS / "sample_metadata.json",
        ids_path=_TASKS / "ids.txt",
    )


def test_load_returns_repo_completion_tasks() -> None:
    """load() joins ids.txt + metadata into repo_completion VibeTasks."""
    tasks = _source().load()
    assert [t.id for t in tasks] == [
        "secrepobench/910",
        "secrepobench/1065",
        "secrepobench/1427",
    ]
    first = tasks[0]
    assert first.source_dataset == "secrepobench"
    assert first.task_type == "repo_completion"
    assert first.environment is not None
    assert first.environment.oracle == "secrepobench"
    assert first.environment.requires_docker is True
    # crash_type -> CWE label.
    assert first.labels.cwe == ["CWE-122"]
    assert first.repo.base_commit == (
        "f9d75ccef0b54c9f4167d95088d4727985133c52"
    )


def test_load_respects_limit() -> None:
    """load(limit=1) returns a single task."""
    tasks = _source().load(limit=1)
    assert len(tasks) == 1
    assert tasks[0].id == "secrepobench/910"


def test_load_warns_on_ids_missing_from_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An id in ids.txt with no metadata entry is skipped LOUDLY.

    The skip itself is legitimate (fixtures subset metadata), but it must
    leave a trace: in production one real task (secrepobench/1065) was
    silently dropped for one model, deflating coverage 317 vs 318. One
    warning must name the count and the exact ids.
    """
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("id\n910\n9999\n1427\n", encoding="utf-8")
    source = SecRepoBenchTaskSource(
        metadata_path=_TASKS / "sample_metadata.json",
        ids_path=ids_path,
    )
    with caplog.at_level(
        "WARNING", logger="guard_eval_harness.vibecoding.sources.secrepobench"
    ):
        tasks = source.load()
    # The skip behavior is preserved: present ids load, absent ones do not.
    assert [t.id for t in tasks] == [
        "secrepobench/910",
        "secrepobench/1427",
    ]
    warnings = [
        r for r in caplog.records if r.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "secrepobench" in message
    assert "1 id(s)" in message
    assert "9999" in message


def test_load_emits_no_warning_when_all_ids_have_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The clean join (every id has metadata) stays silent."""
    with caplog.at_level(
        "WARNING", logger="guard_eval_harness.vibecoding.sources.secrepobench"
    ):
        tasks = _source().load()
    assert len(tasks) == 3
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# --- staging -------------------------------------------------------------


def _completion_artifact(task_id: str, text: str) -> AgentArtifact:
    return AgentArtifact(
        task_id=task_id,
        model="gpt-test",
        kind="completion",
        completion=text,
    )


def test_stage_writes_completion_text_not_a_diff(tmp_path: Path) -> None:
    """stage() writes raw completion code text under completions/<id>/."""
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    completion_code = "    if (TagSize < 8) goto Error;\n"
    artifacts = [
        _completion_artifact("secrepobench/910", completion_code),
    ]
    staged = oracle.stage(tasks, artifacts, tmp_path)

    assert isinstance(staged, StagedOracleInput)
    assert staged.adapter_name == "secrepobench"
    assert staged.task_ids == ["secrepobench/910"]

    expected_name = _completion_filename(
        agent="none",
        model="gpt-test",
        context_type="full",
        prompt_type="instruct",
        mode="perturbed",
    )
    written = (
        Path(staged.inputs_dir)
        / "completions"
        / "910"
        / expected_name
    )
    assert written.exists()
    content = written.read_text(encoding="utf-8")
    # It is code text, NOT a unified diff.
    assert content == completion_code
    assert "--- a/" not in content
    assert "+++ b/" not in content
    assert "@@" not in content
    assert not content.lstrip().startswith("diff --git")


def test_stage_full_file_artifact_writes_single_file(
    tmp_path: Path,
) -> None:
    """A full_file artifact stages the single target file's contents."""
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    file_text = "int main(void) { return 0; }\n"
    artifact = AgentArtifact(
        task_id="secrepobench/910",
        model="gpt-test",
        kind="full_file",
        files={"src/cmsio0.c": file_text},
    )
    staged = oracle.stage(tasks, [artifact], tmp_path)
    written = (
        Path(staged.inputs_dir)
        / "completions"
        / "910"
        / _completion_filename(
            agent="none",
            model="gpt-test",
            context_type="full",
            prompt_type="instruct",
            mode="perturbed",
        )
    )
    assert written.read_text(encoding="utf-8") == file_text


def test_validate_rejects_patch_kind_artifact() -> None:
    """A generic unified diff cannot become a masked-region completion."""
    oracle = SecRepoBenchOracle()
    patch_artifact = AgentArtifact(
        task_id="secrepobench/910",
        model="gpt-test",
        kind="patch",
        patch=(
            "--- a/src/cmsio0.c\n"
            "+++ b/src/cmsio0.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
    )
    with pytest.raises(UnsupportedArtifactError):
        oracle.validate(patch_artifact)


def test_stage_rejects_patch_kind_artifact(tmp_path: Path) -> None:
    """stage() also rejects a patch-kind artifact (defense in depth)."""
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    patch_artifact = AgentArtifact(
        task_id="secrepobench/910",
        model="gpt-test",
        kind="patch",
        patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
    )
    with pytest.raises(UnsupportedArtifactError):
        oracle.stage(tasks, [patch_artifact], tmp_path)


def test_stage_rejects_escaping_model(tmp_path: Path) -> None:
    """A model that injects traversal into the completion filename is rejected.

    The model is embedded verbatim in
    ``completions/<id>/<model>-filled-code-...txt``; a value containing
    ``/../`` (or an absolute path) would otherwise climb out of
    ``completions/``. ``validate`` now rejects it as a per-candidate
    ``UnsupportedArtifactError`` (the runner demotes that per artifact) before
    ``safe_relpath`` would raise a batch-aborting ValueError at write time.
    """
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    for bad_model in ("../../evil", "/abs"):
        artifact = AgentArtifact(
            task_id="secrepobench/910",
            model=bad_model,
            kind="full_file",
            files={"src/cmsio0.c": "int main(void) { return 0; }\n"},
        )
        with pytest.raises(UnsupportedArtifactError):
            oracle.stage(tasks, [artifact], tmp_path)


def test_stage_accepts_hf_style_model_confined(tmp_path: Path) -> None:
    """A normal ``org/model`` stays inside ``completions/`` (no escape)."""
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    artifact = AgentArtifact(
        task_id="secrepobench/910",
        model="acme/coder",
        kind="full_file",
        files={"src/cmsio0.c": "int main(void) { return 0; }\n"},
    )
    staged = oracle.stage(tasks, [artifact], tmp_path)
    completions = (Path(staged.inputs_dir) / "completions").resolve()
    written = Path(staged.metadata["entries"]["secrepobench/910"][
        "completion_path"
    ]).resolve()
    assert written.is_relative_to(completions)
    # Non-vacuity: the confined completion file is actually written.
    assert written.exists()


def test_stage_rejects_escaping_upstream_id_from_task_id(
    tmp_path: Path,
) -> None:
    """Traversal injected via the task id (upstream_id) is also confined.

    ``upstream_id`` is the candidate's task id minus the ``secrepobench/``
    prefix, so the same ``safe_relpath`` guard must reject a ``..`` arriving
    through that arm (not only through the model-in-filename arm).
    """
    oracle = SecRepoBenchOracle()
    base = next(t for t in _source().load() if t.id == "secrepobench/910")
    bad_id = "secrepobench/../../../etc/passwd"
    bad_task = base.model_copy(update={"id": bad_id})
    artifact = AgentArtifact(
        task_id=bad_id,
        model="gpt-test",
        kind="full_file",
        files={"src/cmsio0.c": "int main(void) { return 0; }\n"},
    )
    with pytest.raises(ValueError):
        oracle.stage([bad_task], [artifact], tmp_path)


# --- parsing -------------------------------------------------------------


class _StubEnvProvider:
    """Canned env provider: drops upstream reports where evaluate() looks.

    Never runs Docker, a venv, or the network. ``evaluate`` shells the
    adapter's stage/evaluate seam by faking ``ensure_ready`` / ``resolve`` /
    ``run`` and dropping the inline-built eval OUTPUT report plus the kept
    baseline ``report.json`` INPUT fixture where the adapter expects upstream
    outputs.
    """

    def __init__(
        self, run_dir: Path, returncode: int = 0, produce_report: bool = True,
    ) -> None:
        self._run_dir = run_dir
        self._returncode = returncode
        self._produce_report = produce_report
        self.last_timeout_s: float | None = None

    def ensure_ready(self, *, force: bool = False):
        return self

    def resolve(self):
        # Minimal duck-typed ResolvedEnv: only attributes the adapter reads.
        class _Resolved:
            workdir = str(self._run_dir / "checkout")
            venv_python = "/usr/bin/python3"

        Path(_Resolved.workdir).mkdir(parents=True, exist_ok=True)
        # Only the ground-truth baseline report.json is a pre-existing INPUT
        # read from the kept fixture. The eval OUTPUT report_eval.json is
        # produced by the upstream run (see ``run`` below), since ``evaluate``
        # now clears any stale canonical report_eval.json before the run.
        shutil.copy(
            _RAW / "report.json",
            Path(_Resolved.workdir) / "report.json",
        )
        return _Resolved()

    def run(self, argv, *, run_dir, timeout_s=None, budget=None):
        self.last_timeout_s = timeout_s
        logs_dir = Path(run_dir) / "upstream" / "secrepobench" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Simulate run_eval.py producing the eval OUTPUT report in the checkout
        # (a timed-out / failed run produces nothing -- produce_report=False).
        if self._produce_report:
            _write_report_eval(self._run_dir / "checkout")

        class _Result:
            returncode = self._returncode

        return _Result()


def _stage_three(tmp_path: Path) -> tuple[SecRepoBenchOracle, StagedOracleInput]:
    oracle = SecRepoBenchOracle()
    tasks = _source().load()
    artifacts = [
        _completion_artifact("secrepobench/910", "secure fill\n"),
        _completion_artifact("secrepobench/1065", "insecure fill\n"),
        _completion_artifact("secrepobench/1427", "broken fill\n"),
    ]
    staged = oracle.stage(tasks, artifacts, tmp_path)
    return oracle, staged


def _evaluate_and_parse(tmp_path: Path):
    oracle, staged = _stage_three(tmp_path)
    run_config = OracleRunConfig(
        run_id="secrepo-test", run_dir=str(tmp_path)
    )
    env = _StubEnvProvider(tmp_path)
    raw = oracle.evaluate(
        staged, run_config, ResourceBudget(max_workers=2), env
    )
    rows = oracle.parse(raw)
    return oracle, raw, {row.task_id: row for row in rows}


def test_evaluate_locates_and_copies_reports(tmp_path: Path) -> None:
    """evaluate() copies report_eval.json + report.json into outputs_dir."""
    _, raw, _ = _evaluate_and_parse(tmp_path)
    assert isinstance(raw, RawOracleResult)
    outputs = Path(raw.outputs_dir)
    assert (outputs / "report_eval.json").exists()
    assert (outputs / "report.json").exists()
    assert raw.metadata["report_eval_present"] is True
    assert raw.metadata["base_report_present"] is True
    assert raw.exit_code == 0
    # Score every staged target fresh: bypass the upstream report_eval.json
    # cache (keyed on model name, not the candidate) via --rerun.
    assert "--rerun" in raw.metadata["upstream_command"]


def test_evaluate_clears_stale_canonical_report_cache(tmp_path: Path) -> None:
    """A stale checkout-global report_eval.json is removed before the run so
    GEH (which prefers the canonical name) reads this run's fresh report."""
    oracle, staged = _stage_three(tmp_path)
    env = _StubEnvProvider(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / "report_eval.json").write_text('{"stale": true}')
    run_config = OracleRunConfig(run_id="r", run_dir=str(tmp_path))
    raw = oracle.evaluate(
        staged, run_config, ResourceBudget(max_workers=2), env
    )
    # The copied report is the fresh one the run produced, not the stale stub.
    copied = json.loads((Path(raw.outputs_dir) / "report_eval.json").read_text())
    assert "stale" not in copied


def test_evaluate_ignores_stale_timestamped_report(tmp_path: Path) -> None:
    """A timed-out/failed run that writes no report must not fall back to a
    prior run's timestamped report_eval_*.json (it would mis-attribute the old
    candidate's verdict); evaluate clears those too."""
    oracle, staged = _stage_three(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / "report_eval_20200101_000000.json").write_text('{"stale": 1}')
    env = _StubEnvProvider(tmp_path, produce_report=False)
    run_config = OracleRunConfig(run_id="r", run_dir=str(tmp_path))
    raw = oracle.evaluate(
        staged, run_config, ResourceBudget(max_workers=2), env
    )
    # No fresh report was produced, and the stale timestamped one was cleared,
    # so nothing is located/copied -- not a stale verdict.
    assert raw.metadata["report_eval_present"] is False


def test_copy_outputs_is_binary_safe(tmp_path: Path) -> None:
    """_copy_outputs must byte-copy opaque upstream reports, not re-decode.

    A non-UTF-8 byte in an upstream report (crash-log fragment, mojibake
    path) must not raise UnicodeDecodeError out of evaluate(); the copy is
    binary and the destination bytes match the source exactly.
    """
    src_dir = tmp_path / "src"
    outputs = tmp_path / "outputs"
    src_dir.mkdir()
    outputs.mkdir()
    payload = b'{"x": 1}\xff\xfe'
    (src_dir / "report_eval.json").write_bytes(payload)
    (src_dir / "report.json").write_bytes(payload)
    present = SecRepoBenchOracle._copy_outputs(
        src_dir / "report_eval.json", src_dir / "report.json", outputs
    )
    assert present == {"report_eval": True, "base_report": True}
    assert (outputs / "report_eval.json").read_bytes() == payload
    assert (outputs / "report.json").read_bytes() == payload


def test_copy_outputs_missing_sources_are_flagged_absent(
    tmp_path: Path,
) -> None:
    """Missing upstream reports => present flags False, nothing written."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    present = SecRepoBenchOracle._copy_outputs(
        tmp_path / "report_eval.json", tmp_path / "report.json", outputs
    )
    assert present == {"report_eval": False, "base_report": False}
    assert list(outputs.iterdir()) == []


def test_evaluate_applies_default_timeout(tmp_path: Path) -> None:
    """With an empty run_config.extra, evaluate() must still cap the run.

    A bare timeout_s=None would let a hung Docker/upstream eval wait forever;
    the adapter supplies a finite default (overridable via extra["timeout_s"]).
    """
    from guard_eval_harness.vibecoding.oracles.secrepobench import (
        _DEFAULT_TIMEOUT_S,
    )

    oracle, staged = _stage_three(tmp_path)
    env = _StubEnvProvider(tmp_path)
    # Default: no timeout_s in extra -> the adapter's finite default.
    oracle.evaluate(
        staged,
        OracleRunConfig(run_id="t", run_dir=str(tmp_path)),
        ResourceBudget(max_workers=2),
        env,
    )
    assert env.last_timeout_s == _DEFAULT_TIMEOUT_S
    # An explicit override is honored.
    oracle.evaluate(
        staged,
        OracleRunConfig(
            run_id="t", run_dir=str(tmp_path), extra={"timeout_s": 123.0}
        ),
        ResourceBudget(max_workers=2),
        env,
    )
    assert env.last_timeout_s == 123.0


def test_parse_pass_case_is_completed_secure(tmp_path: Path) -> None:
    """testcase=pass + baseline-subset unit tests => completed/secure."""
    _, _, by_id = _evaluate_and_parse(tmp_path)
    row = by_id["secrepobench/910"]
    assert row.status == "completed"
    assert row.failure_origin == "none"
    assert row.failure_reason is None
    assert row.security_oracle_pass is True
    assert row.known_vuln_present is False
    assert row.functional_pass is True
    assert row.patch_applied is True
    assert row.new_vuln_introduced is None
    # secure-pass@1 == target_secure_success.
    assert row.target_secure_success is True
    # No new-vuln capability => strict secure stays null.
    assert row.strict_secure_success is None
    assert row.raw.upstream_status == "pass"


def test_parse_crash_case_is_completed_insecure(tmp_path: Path) -> None:
    """testcase=crash => completed but target vuln present (insecure)."""
    _, _, by_id = _evaluate_and_parse(tmp_path)
    row = by_id["secrepobench/1065"]
    assert row.status == "completed"
    assert row.failure_origin == "none"
    assert row.security_oracle_pass is False
    assert row.known_vuln_present is True
    # Unit tests still pass (baseline subset holds), but security fails.
    assert row.functional_pass is True
    assert row.patch_applied is True
    assert row.target_secure_success is False
    # Kleene AND: security False makes strict a definite False even though the
    # new-vuln gate is None (False dominates AND over unknowns).
    assert row.strict_secure_success is False
    assert row.raw.upstream_status == "crash"


def test_parse_compile_error_is_model_failure(tmp_path: Path) -> None:
    """A compile-error testcase => model_failure/build_failed."""
    _, _, by_id = _evaluate_and_parse(tmp_path)
    row = by_id["secrepobench/1427"]
    assert row.status == "model_failure"
    assert row.failure_origin == "model"
    assert row.failure_reason == "build_failed"
    assert row.security_oracle_pass is None
    assert row.known_vuln_present is None
    assert row.patch_applied is False
    # Build failed => functional tests cannot pass (functional_pass False),
    # while the security verdict is None. Under Kleene AND the functional False
    # is a definite failure that dominates the unknown security gate, so both
    # metrics are a definite False (a real failure counted in the denominator),
    # not an excluded None.
    assert row.functional_pass is False
    assert row.target_secure_success is False
    assert row.strict_secure_success is False
    assert str(row.raw.upstream_status).startswith("error:")


def test_parse_missing_leaf_is_infra_failure(tmp_path: Path) -> None:
    """A task with no upstream result row => infra_failure, not model."""
    oracle = SecRepoBenchOracle()
    raw = RawOracleResult(
        adapter_name="secrepobench",
        outputs_dir=str(tmp_path),
        logs_dir=str(tmp_path),
        exit_code=0,
        task_ids=["secrepobench/910"],
        metadata={
            "entries": {
                "secrepobench/910": {
                    "upstream_id": "910",
                    "model": "gpt-test",
                    "source_dataset": "secrepobench",
                }
            },
            "agent": "none",
            "context_type": "full",
            "prompt_type": "instruct",
            "mode": "perturbed",
        },
    )
    # No report_eval.json / report.json in outputs_dir => empty reports.
    rows = oracle.parse(raw)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "infra_failure"
    assert row.failure_origin == "infra"
    assert row.failure_reason == "verifier_unavailable"
    # Infra failure must not fabricate verdicts.
    assert row.security_oracle_pass is None
    assert row.functional_pass is None
    assert row.target_secure_success is None


def test_load_json_returns_empty_on_non_utf8_bytes(tmp_path: Path) -> None:
    """_load_json treats undecodable bytes like missing/bad JSON ({}).

    _copy_outputs is binary-safe, so a non-UTF-8 upstream report reaches
    parse(); UnicodeDecodeError is a ValueError (not an OSError or a
    JSONDecodeError), so it needs its own arm in the except tuple or the
    decode error would still abort the batch — just from parse() instead
    of evaluate().
    """
    path = tmp_path / "report_eval.json"
    path.write_bytes(b'{"x": 1}\xff\xfe')
    assert SecRepoBenchOracle._load_json(path) == {}


def test_parse_undecodable_report_degrades_to_infra(tmp_path: Path) -> None:
    """A non-UTF-8 report_eval.json => infra rows from parse(), no raise."""
    oracle = SecRepoBenchOracle()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "report_eval.json").write_bytes(b'{"x": 1}\xff\xfe')
    (outputs / "report.json").write_bytes(b'{"x": 1}\xff\xfe')
    raw = RawOracleResult(
        adapter_name="secrepobench",
        outputs_dir=str(outputs),
        logs_dir=str(tmp_path),
        exit_code=0,
        task_ids=["secrepobench/910"],
        metadata={
            "entries": {
                "secrepobench/910": {
                    "upstream_id": "910",
                    "model": "gpt-test",
                    "source_dataset": "secrepobench",
                }
            },
            "agent": "none",
            "context_type": "full",
            "prompt_type": "instruct",
            "mode": "perturbed",
        },
    )
    rows = oracle.parse(raw)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "infra_failure"
    assert row.failure_origin == "infra"
    assert row.failure_reason == "verifier_unavailable"
    # Undecodable report must not fabricate verdicts either.
    assert row.security_oracle_pass is None
    assert row.functional_pass is None
    assert row.target_secure_success is None


def test_capabilities_and_parallelism() -> None:
    """Adapter declares accurate capabilities + batch_internal parallelism."""
    oracle = SecRepoBenchOracle()
    caps = oracle.capabilities
    assert caps.runs_functional_tests is True
    assert caps.detects_target_vuln is True
    assert caps.detects_new_vuln is False
    assert caps.dynamic_pov is True
    assert caps.deterministic is False
    assert oracle.parallelism.model == "batch_internal"
    assert oracle.granularity == "batch"
    assert oracle.task_types == {"repo_completion"}
    assert oracle.artifact_kinds == {"completion", "full_file"}
    assert oracle.env.license_policy == "external_only"
    assert oracle.env.requires_docker is True


def test_functional_pass_false_when_baseline_not_subset(
    tmp_path: Path,
) -> None:
    """If the candidate drops a baseline-passing test, functional_pass=False."""
    oracle = SecRepoBenchOracle()
    # Build a report where eval pass-set is missing a baseline test (deep-copy
    # the inline OUTPUT literal so the shared fixture is not mutated).
    report_eval = copy.deepcopy(_REPORT_EVAL)
    report_eval["910"]["none"]["gpt-test"]["full"]["instruct"][
        "perturbed"
    ]["unittest"]["pass"] = ["quick floor"]
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "report_eval.json").write_text(
        json.dumps(report_eval), encoding="utf-8"
    )
    shutil.copy(_RAW / "report.json", outputs / "report.json")

    raw = RawOracleResult(
        adapter_name="secrepobench",
        outputs_dir=str(outputs),
        logs_dir=str(tmp_path),
        exit_code=0,
        task_ids=["secrepobench/910"],
        metadata={
            "entries": {
                "secrepobench/910": {
                    "upstream_id": "910",
                    "model": "gpt-test",
                    "source_dataset": "secrepobench",
                }
            },
            "agent": "none",
            "context_type": "full",
            "prompt_type": "instruct",
            "mode": "perturbed",
        },
    )
    row = oracle.parse(raw)[0]
    assert row.security_oracle_pass is True
    assert row.functional_pass is False
    # Secure testcase but functional fail => not target-secure.
    assert row.target_secure_success is False


def test_derive_metrics_consistency(tmp_path: Path) -> None:
    """Parsed rows already have derived metrics matching derive_task_metrics."""
    _, _, by_id = _evaluate_and_parse(tmp_path)
    for row in by_id.values():
        before_target = row.target_secure_success
        before_strict = row.strict_secure_success
        rederived = derive_task_metrics(row.model_copy(deep=True))
        assert rederived.target_secure_success == before_target
        assert rederived.strict_secure_success == before_strict
