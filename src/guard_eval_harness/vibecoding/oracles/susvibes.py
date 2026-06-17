"""SusVibes oracle adapter.

Wraps the upstream SusVibes evaluator (``susvibes.run_evaluation``) as an
out-of-process oracle. SusVibes is the headline ``repo_patch`` / target-secure
adapter: it runs the project's functional test suite *and* a security test
suite (the ``func`` and ``sec`` runs) inside a per-instance Docker image and
reports whether each passed.

Pipeline:

- ``stage`` writes a SWE-bench-style ``predictions.jsonl`` (one row per task)
  with the **raw, unfiltered** model patch. Upstream itself strips test/binary
  files from the diff, so the adapter must not pre-filter.
- ``evaluate`` acquires the venv + pinned checkout, then runs
  ``python -m susvibes.run_evaluation --run_id ... --predictions_path ...
  --max_workers ... --force`` (no ``--strategy`` => the ``generic`` strategy)
  via the injected :class:`EnvProvider`. Upstream writes its logs into the
  checkout under ``logs/run_evaluation/<run_id>/generic/<model__key>/``; the
  adapter copies that subtree into ``run_dir/upstream/susvibes/outputs/`` and
  returns a :class:`RawOracleResult`.
- ``parse`` reads each instance's ``report.json`` (``func`` / ``sec`` blocks)
  and maps it to a normalized :class:`VibeTaskResult` with tri-state verdicts
  and the crucial infra-vs-model attribution. The per-model ``summary.json``
  is authoritative for ``no_patch`` / ``model_patch_error`` (upstream writes
  no per-instance report for those): it recovers missing reports and
  overrides stale leftover ones from a prior attempt in a reused task dir.

The GEH process never imports any ``susvibes`` upstream module; all execution
goes through the env provider's subprocess seam.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from guard_eval_harness.execution.artifacts import dump_jsonl
from guard_eval_harness.vibecoding.artifacts import (
    AgentArtifact,
    artifact_sha256,
    task_sha256,
)
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.base import OracleAdapter
from guard_eval_harness.vibecoding.registry import oracle_registry
from guard_eval_harness.vibecoding.results import (
    ProvenanceBlock,
    RawBlock,
    VibeTaskResult,
    derive_task_metrics,
)
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    ResourceEstimate,
    VibeTask,
)

# Upstream pinned ref (commit) for reproducibility.
_UPSTREAM_REF = "dd28a7e224b09e3ee666ffbcb56b95d109d2f8d7"

# The "generic" strategy directory upstream writes into (no --strategy flag).
_STRATEGY = "generic"

# Upstream file names (mirror susvibes.tasks / susvibes.constants).
_SUMMARY_FILE = "summary.json"
_REPORT_FILE = "report.json"

# Staged predictions file name within the adapter's inputs dir.
_PREDICTIONS_FILE = "predictions.jsonl"

# Default per-evaluation timeout (s). One SusVibes instance runs two Docker
# test suites with an upstream 1800s container timeout each; the runner can
# override per call. Kept generous so legitimate runs are not cut short.
_DEFAULT_TIMEOUT_S = 6 * 60 * 60

# Map upstream EvalStatus values -> (geh status, failure_origin, reason).
# Only model/infra terminal statuses are listed; ``completion`` is scoreable
# and handled separately (it may still be a functional/security failure).
_NO_PATCH = "no_patch"
_MODEL_PATCH_ERROR = "model_patch_error"
_STARTUP_ERROR = "startup_error"
_TIMEOUT = "timeout"
_COMPLETION = "completion"


def _model_key(model: str) -> str:
    """Upstream's per-model output dir key (``/`` -> ``__``)."""
    return (model or "none").replace("/", "__")


def _strip_prefix(task_id: str) -> str:
    """Strip the ``susvibes/`` task-id prefix to the upstream instance_id."""
    prefix = "susvibes/"
    return task_id[len(prefix):] if task_id.startswith(prefix) else task_id


@oracle_registry.register("susvibes")
class SusVibesOracle(OracleAdapter):
    """Out-of-process wrapper around ``susvibes.run_evaluation``."""

    name = "susvibes"
    env = EnvSpec(
        name="susvibes",
        kind="venv",
        upstream_url="https://github.com/LeiLiLab/susvibes",
        upstream_ref=_UPSTREAM_REF,
        install=[
            "pip install -r requirements.txt",
            "pip install -e .",
        ],
        requires_docker=True,
        requires_network_for_eval=True,
        disk_gb_estimate=400.0,
        resource_estimate=ResourceEstimate(
            cpu_per_worker=4,
            memory_gb_per_worker=4.0,
            disk_gb_per_worker=20.0,
            gpu_required=False,
        ),
        parallelism=OracleParallelism(
            model="batch_internal",
            default_workers=4,
            max_workers=8,
        ),
        license_policy="vendor_allowed",
    )
    artifact_kinds = {"patch"}
    task_types = {"repo_patch"}
    granularity = "batch"
    capabilities = OracleCapabilities(
        runs_functional_tests=True,
        detects_target_vuln=True,
        detects_new_vuln=False,
        dynamic_pov=True,
        static_analysis=False,
        fuzzing=False,
        llm_judge=False,
        deterministic=False,
    )
    parallelism = OracleParallelism(
        model="batch_internal",
        default_workers=4,
        max_workers=8,
    )
    parser_version = "susvibes-parser/1"

    # --- stage --------------------------------------------------------

    def stage(
        self,
        tasks: list[VibeTask],
        artifacts: list[AgentArtifact],
        run_dir: Path,
    ) -> StagedOracleInput:
        """Write ``predictions.jsonl`` with raw patches; reject non-patch."""
        inputs_dir = Path(run_dir) / "upstream" / self.name / "inputs"
        by_task = {task.id: task for task in tasks}
        rows: list[dict[str, Any]] = []
        task_ids: list[str] = []
        # Per-task provenance the parser reuses (artifact/task hashes, model).
        index: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            if artifact.kind not in self.artifact_kinds:
                raise UnsupportedArtifactError(
                    f"susvibes oracle accepts {sorted(self.artifact_kinds)} "
                    f"artifacts, got kind={artifact.kind!r} for "
                    f"task_id={artifact.task_id!r}"
                )
            instance_id = _strip_prefix(artifact.task_id)
            model = artifact.model or "none"
            rows.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": model,
                    # RAW, unfiltered diff: upstream does its own test/binary
                    # filtering; pre-filtering here would diverge from the
                    # reference harness.
                    "model_patch": artifact.patch or "",
                }
            )
            task_ids.append(artifact.task_id)
            task = by_task.get(artifact.task_id)
            index[artifact.task_id] = {
                "instance_id": instance_id,
                "model": model,
                "model_key": _model_key(model),
                "artifact_sha256": artifact_sha256(artifact),
                "task_sha256": task_sha256(task) if task else None,
                "source_dataset": (
                    task.source_dataset if task else "susvibes"
                ),
            }
        predictions_path = inputs_dir / _PREDICTIONS_FILE
        dump_jsonl(predictions_path, rows)
        return StagedOracleInput(
            adapter_name=self.name,
            inputs_dir=str(inputs_dir),
            task_ids=task_ids,
            metadata={
                "predictions_file": _PREDICTIONS_FILE,
                "predictions_path": str(predictions_path),
                "index": index,
            },
        )

    # --- evaluate -----------------------------------------------------

    def evaluate(
        self,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
        env_provider: Any,
    ) -> RawOracleResult:
        """Run upstream out-of-process, then collect its log subtree."""
        run_dir = Path(run_config.run_dir)
        outputs_dir = run_dir / "upstream" / self.name / "outputs"
        logs_dir = run_dir / "upstream" / self.name / "logs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = staged.metadata.get(
            "predictions_path",
            str(Path(staged.inputs_dir) / _PREDICTIONS_FILE),
        )
        max_workers = min(
            int(resource_budget.max_workers),
            int(self.parallelism.max_workers),
        )
        max_workers = max(1, max_workers)
        run_id = run_config.run_id

        # No ``--strategy`` => the upstream default ``generic`` strategy.
        argv = [
            "python",
            "-m",
            "susvibes.run_evaluation",
            "--run_id",
            run_id,
            "--predictions_path",
            str(Path(predictions_path).resolve()),
            "--max_workers",
            str(max_workers),
            "--force",
        ]

        # The real provider acquires the venv + pinned checkout, then runs the
        # command with the venv on PATH and cwd at the checkout. The adapter
        # never spawns processes itself.
        env_provider.ensure_ready()
        result = env_provider.run(
            argv,
            run_dir=Path(run_config.run_dir),
            timeout_s=_DEFAULT_TIMEOUT_S,
            budget=resource_budget,
        )

        checkout_dir = self._checkout_dir(env_provider)
        upstream_log_root = (
            checkout_dir
            / "logs"
            / "run_evaluation"
            / run_id
            / _STRATEGY
        )
        self._collect_outputs(upstream_log_root, outputs_dir)

        exit_code = getattr(result, "returncode", None)
        timed_out = bool(getattr(result, "timed_out", False))
        if timed_out:
            # Surface the timeout to the parser as an infra signal; the
            # per-instance reports may still be absent.
            exit_code = exit_code if exit_code is not None else 124

        return RawOracleResult(
            adapter_name=self.name,
            outputs_dir=str(outputs_dir),
            logs_dir=str(logs_dir),
            exit_code=exit_code,
            task_ids=list(staged.task_ids),
            metadata={
                "run_id": run_id,
                "strategy": _STRATEGY,
                "max_workers": max_workers,
                "timed_out": timed_out,
                "upstream_command": list(argv),
                "upstream_workdir": str(checkout_dir),
                "index": staged.metadata.get("index", {}),
            },
        )

    def _checkout_dir(self, env_provider: Any) -> Path:
        """Resolve the upstream checkout dir from the env provider."""
        resolve = getattr(env_provider, "resolve", None)
        if callable(resolve):
            resolved = resolve()
            workdir = getattr(resolved, "workdir", None)
            if workdir:
                return Path(workdir)
            upstream_dir = getattr(resolved, "upstream_dir", None)
            if upstream_dir:
                return Path(upstream_dir)
        # Fallback for stubbed providers that expose a plain attribute.
        checkout = getattr(env_provider, "checkout_dir", None)
        if checkout:
            return Path(checkout)
        raise RuntimeError(
            "susvibes oracle could not resolve the upstream checkout dir "
            "from the env provider"
        )

    def _collect_outputs(
        self, upstream_log_root: Path, outputs_dir: Path
    ) -> None:
        """Copy upstream ``<run>/generic/<model__key>/`` into outputs_dir.

        Idempotent and tolerant of a missing source tree (e.g. when upstream
        crashed before writing any report); the parser then maps absent
        reports to infra failures.
        """
        if not upstream_log_root.exists():
            return
        for model_dir in sorted(upstream_log_root.iterdir()):
            if not model_dir.is_dir():
                continue
            dest = outputs_dir / model_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(model_dir, dest)

    # --- parse --------------------------------------------------------

    def parse(self, raw: RawOracleResult) -> list[VibeTaskResult]:
        """Map per-instance upstream reports to normalized result rows."""
        outputs_dir = Path(raw.outputs_dir)
        index: dict[str, dict[str, Any]] = raw.metadata.get("index", {})
        run_timed_out = bool(raw.metadata.get("timed_out", False))
        rows: list[VibeTaskResult] = []
        for task_id in raw.task_ids:
            meta = index.get(task_id, {})
            rows.append(
                self._row(
                    task_id=task_id,
                    meta=meta,
                    outputs_dir=outputs_dir,
                    raw=raw,
                    run_timed_out=run_timed_out,
                )
            )
        return rows

    def _row(
        self,
        *,
        task_id: str,
        meta: dict[str, Any],
        outputs_dir: Path,
        raw: RawOracleResult,
        run_timed_out: bool,
    ) -> VibeTaskResult:
        """Build a single normalized result row from one upstream report."""
        instance_id = meta.get("instance_id", _strip_prefix(task_id))
        model = meta.get("model", "none")
        model_key = meta.get("model_key", _model_key(model))
        source_dataset = meta.get("source_dataset", "susvibes")

        report_path = (
            outputs_dir / model_key / instance_id / _REPORT_FILE
        )
        summary_path = outputs_dir / model_key / _SUMMARY_FILE
        report = _load_json(report_path)

        defaults = self._null_fields()
        if report is None:
            # No per-instance report was written. If the whole run timed out
            # this is an infra timeout; otherwise the oracle never produced a
            # scoreable result for this instance (image/startup issue).
            return self._missing_report_row(
                task_id=task_id,
                model=model,
                source_dataset=source_dataset,
                report_path=report_path,
                summary_path=summary_path,
                raw=raw,
                meta=meta,
                run_timed_out=run_timed_out,
                defaults=defaults,
            )

        func = report.get("func", {}) or {}
        sec = report.get("sec", {}) or {}
        func_status = func.get("status")
        sec_status = sec.get("status")

        # Stale-report guard. Upstream's ``no_patch`` early return writes NO
        # per-instance ``report.json`` (susvibes.tasks.run_evaluation_single),
        # so in a reused task dir a leftover report from a PRIOR attempt would
        # otherwise launder this run's model failure into a pass. The
        # per-model ``summary.json`` is authoritative for ``no_patch`` /
        # ``model_patch_error``: when it lists this instance and the report's
        # own statuses contradict that listing (a fresh apply-failure report
        # legitimately carries ``model_patch_error`` itself), ignore the
        # stale report and emit the same recovered model-failure row as the
        # missing-report path. A missing/unreadable summary keeps trusting
        # the report.
        summary_status = self._summary_failure_status(
            summary_path, instance_id
        )
        if summary_status is not None and summary_status not in (
            func_status,
            sec_status,
        ):
            return self._summary_model_failure_row(
                task_id=task_id,
                model=model,
                source_dataset=source_dataset,
                report_path=report_path,
                raw=raw,
                meta=meta,
                upstream_status=summary_status,
                extra={
                    "report_missing": False,
                    "stale_report_ignored": True,
                    "summary_path": str(summary_path),
                    "resolved_via": (
                        f"summary.{summary_status} "
                        "(stale report.json ignored)"
                    ),
                },
            )

        fields = self._classify(func_status, sec_status, func, sec)
        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model or "none",
            status=fields["status"],
            failure_origin=fields["failure_origin"],
            failure_reason=fields["failure_reason"],
            patch_applied=fields["patch_applied"],
            build_pass=fields["build_pass"],
            functional_pass=fields["functional_pass"],
            security_oracle_pass=fields["security_oracle_pass"],
            known_vuln_present=fields["known_vuln_present"],
            new_vuln_introduced=None,
            oracle_capabilities=self.capabilities,
            raw=self._raw_block(
                func_status=func_status,
                sec_status=sec_status,
                report_path=report_path,
                summary_path=summary_path,
                raw=raw,
            ),
            provenance=self._provenance(meta, raw),
        )
        return derive_task_metrics(result)

    def _classify(
        self,
        func_status: str | None,
        sec_status: str | None,
        func: dict[str, Any],
        sec: dict[str, Any],
    ) -> dict[str, Any]:
        """Attribute an upstream report to normalized status + verdicts.

        SusVibes runs ``func`` then ``sec``; a model/infra terminal status in
        either run is propagated. ``no_patch`` / ``model_patch_error`` are
        model failures (the candidate diff never reached a clean oracle run).
        ``startup_error`` is *also* a model failure: the patch applied but
        broke imports/build so the test runner could not start, which upstream
        counts (in-denominator) as not-correct & not-secure. Only ``timeout``
        is an infra failure (Docker/host).
        """
        statuses = {func_status, sec_status}

        # --- model failures -------------------------------------------
        # no_patch / model_patch_error are submitted, non-correct, non-secure
        # rows in upstream's 186-instance denominator (correct_ratio counts
        # them as failures, not exclusions), so functional/security verdicts
        # are False, not None. known_vuln_present stays None: with no applied
        # patch we cannot assess whether the target vuln is present.
        if _NO_PATCH in statuses:
            return {
                "status": "model_failure",
                "failure_origin": "model",
                "failure_reason": "empty_diff",
                "patch_applied": False,
                "build_pass": None,
                "functional_pass": False,
                "security_oracle_pass": False,
                "known_vuln_present": None,
            }
        if _MODEL_PATCH_ERROR in statuses:
            return {
                "status": "model_failure",
                "failure_origin": "model",
                "failure_reason": "patch_apply_failed",
                "patch_applied": False,
                "build_pass": None,
                "functional_pass": False,
                "security_oracle_pass": False,
                "known_vuln_present": None,
            }

        # --- infra failures -------------------------------------------
        if _TIMEOUT in statuses:
            return {
                "status": "infra_failure",
                "failure_origin": "infra",
                "failure_reason": "oracle_timeout",
                "patch_applied": None,
                "build_pass": None,
                "functional_pass": None,
                "security_oracle_pass": None,
                "known_vuln_present": None,
            }

        # --- model build failure --------------------------------------
        # The patch applied but broke imports/build so the test runner never
        # started. Upstream counts this as a submitted, non-correct,
        # non-secure row (in the 186-instance denominator), not as infra.
        if _STARTUP_ERROR in statuses:
            return {
                "status": "model_failure",
                "failure_origin": "model",
                "failure_reason": "build_failed",
                "patch_applied": True,
                "build_pass": False,
                "functional_pass": False,
                "security_oracle_pass": False,
                "known_vuln_present": True,
            }

        # --- completed (scoreable) ------------------------------------
        functional_pass = _as_bool(func.get("pass"))
        security_pass = _as_bool(sec.get("pass"))
        known_vuln = (
            None if security_pass is None else (not security_pass)
        )
        # The patch reached the full oracle path, so it applied.
        return {
            "status": "completed",
            "failure_origin": "none",
            "failure_reason": None,
            "patch_applied": True,
            "build_pass": None,
            "functional_pass": functional_pass,
            "security_oracle_pass": security_pass,
            "known_vuln_present": known_vuln,
        }

    def _missing_report_row(
        self,
        *,
        task_id: str,
        model: str,
        source_dataset: str,
        report_path: Path,
        summary_path: Path,
        raw: RawOracleResult,
        meta: dict[str, Any],
        run_timed_out: bool,
        defaults: dict[str, Any],
    ) -> VibeTaskResult:
        """Map a missing per-instance report to a normalized result row.

        A run-level timeout stays an infra failure. Otherwise the instance may
        still be a *submitted* model failure that upstream records only in
        ``summary.json`` (``no_patch`` / ``model_patch_error`` write no
        per-instance ``report.json``); recover those as model failures so they
        land in the published in-denominator count. A genuinely absent record
        (in neither list) remains the infra/``image_missing`` fallback.
        """
        instance_id = meta.get("instance_id", _strip_prefix(task_id))

        # Build the verbatim audit ``extra`` once; recovery paths annotate it.
        extra: dict[str, Any] = {
            "report_missing": True,
            "summary_path": str(summary_path),
        }

        if run_timed_out:
            failure_reason = "oracle_timeout"
            upstream_status = _TIMEOUT
        else:
            # Consult the per-model summary: a no_patch / model_patch_error
            # instance has no report.json but IS a submitted, non-correct,
            # non-secure row in upstream's denominator.
            summary_status = self._summary_failure_status(
                summary_path, instance_id
            )
            if summary_status is not None:
                extra["resolved_via"] = f"summary.{summary_status}"
                return self._summary_model_failure_row(
                    task_id=task_id,
                    model=model,
                    source_dataset=source_dataset,
                    report_path=report_path,
                    raw=raw,
                    meta=meta,
                    upstream_status=summary_status,
                    extra=extra,
                )
            # Genuinely no record and the run did not time out: the only
            # remaining infra path (image/startup never produced output).
            failure_reason = "image_missing"
            upstream_status = _STARTUP_ERROR

        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model or "none",
            status="infra_failure",
            failure_origin="infra",
            failure_reason=failure_reason,
            patch_applied=None,
            build_pass=None,
            functional_pass=None,
            security_oracle_pass=None,
            known_vuln_present=None,
            new_vuln_introduced=None,
            oracle_capabilities=self.capabilities,
            raw=RawBlock(
                upstream_status=upstream_status,
                upstream_result_path=str(report_path),
                logs_dir=raw.logs_dir,
                extra=extra,
            ),
            provenance=self._provenance(meta, raw),
        )
        return derive_task_metrics(result)

    def _summary_failure_status(
        self, summary_path: Path, instance_id: str
    ) -> str | None:
        """Look up ``instance_id`` in the summary's model-failure lists.

        Returns ``no_patch`` / ``model_patch_error`` when the per-model
        ``summary.json`` lists the instance under that ``details`` key
        (mirrors upstream ``get_summary``'s precedence), else ``None`` --
        including when the summary itself is missing or unreadable.
        """
        summary = _load_json(summary_path)
        if summary is None:
            return None
        details = summary.get("details", {}) or {}
        if instance_id in (details.get(_NO_PATCH, []) or []):
            return _NO_PATCH
        if instance_id in (details.get(_MODEL_PATCH_ERROR, []) or []):
            return _MODEL_PATCH_ERROR
        return None

    def _summary_model_failure_row(
        self,
        *,
        task_id: str,
        model: str,
        source_dataset: str,
        report_path: Path,
        raw: RawOracleResult,
        meta: dict[str, Any],
        upstream_status: str,
        extra: dict[str, Any],
    ) -> VibeTaskResult:
        """Build the summary-recovered model-failure row.

        Shared by the missing-report recovery and the stale-report guard: a
        summary-listed ``no_patch`` / ``model_patch_error`` instance is a
        submitted, non-correct, non-secure row in upstream's denominator
        (functional/security verdicts False, not None); ``known_vuln_present``
        stays null because no applied patch exists to assess.
        """
        failure_reason = (
            "empty_diff"
            if upstream_status == _NO_PATCH
            else "patch_apply_failed"
        )
        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model or "none",
            status="model_failure",
            failure_origin="model",
            failure_reason=failure_reason,
            patch_applied=False,
            build_pass=None,
            functional_pass=False,
            security_oracle_pass=False,
            known_vuln_present=None,
            new_vuln_introduced=None,
            oracle_capabilities=self.capabilities,
            raw=RawBlock(
                upstream_status=upstream_status,
                upstream_result_path=str(report_path),
                logs_dir=raw.logs_dir,
                extra=extra,
            ),
            provenance=self._provenance(meta, raw),
        )
        return derive_task_metrics(result)

    @staticmethod
    def _null_fields() -> dict[str, Any]:
        """Tri-state verdict defaults (all ``None``)."""
        return {
            "patch_applied": None,
            "functional_pass": None,
            "security_oracle_pass": None,
            "known_vuln_present": None,
        }

    def _raw_block(
        self,
        *,
        func_status: str | None,
        sec_status: str | None,
        report_path: Path,
        summary_path: Path,
        raw: RawOracleResult,
    ) -> RawBlock:
        """Preserve verbatim upstream func/sec statuses + output paths."""
        combined = " / ".join(
            s for s in (func_status, sec_status) if s
        ) or None
        return RawBlock(
            upstream_status=combined,
            upstream_result_path=str(report_path),
            logs_dir=raw.logs_dir,
            extra={
                "func_status": func_status,
                "sec_status": sec_status,
                "summary_path": str(summary_path),
            },
        )

    def _provenance(
        self, meta: dict[str, Any], raw: RawOracleResult
    ) -> ProvenanceBlock:
        """Attach reproduction/audit metadata for one result row."""
        return ProvenanceBlock(
            adapter_name=self.name,
            parser_version=self.parser_version,
            upstream_url=self.env.upstream_url,
            upstream_ref=self.env.upstream_ref,
            upstream_command=list(
                raw.metadata.get("upstream_command", [])
            ),
            upstream_workdir=raw.metadata.get("upstream_workdir"),
            worker_count=raw.metadata.get("max_workers"),
            artifact_sha256=meta.get("artifact_sha256"),
            task_sha256=meta.get("task_sha256"),
        )


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from ``path`` or return ``None`` if absent/bad."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_bool(value: Any) -> bool | None:
    """Coerce an upstream ``pass`` field to a tri-state verdict."""
    if value is None:
        return None
    return bool(value)
