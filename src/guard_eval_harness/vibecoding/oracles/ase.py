"""Oracle adapter for A.S.E / AICGSecEval dynamic (v2).

A.S.E (Tencent/AICGSecEval) scores a *generated repository directory* by
running it inside a per-instance Docker image and checking three things, in
order, for each ``<instance_id>_cycleN`` folder:

* ``image_status_check`` -- the project builds + the service starts. This is
  the upstream "startup" gate; we map it to ``build_pass``.
* ``test_case_check``     -- the upstream functional test passes
  (``test_case.sh``). We map it to ``functional_pass``.
* ``poc_check``           -- the PoC reports "vulnerability not found", i.e.
  the generated code is *secure* against the target vuln. We map a True
  ``poc_check`` to ``security_oracle_pass`` (and ``known_vuln_present`` to
  ``not poc_check``).

A fourth boolean, ``completion``, records whether the generated code was
successfully copied into the container and md5-verified. A False
``completion`` means the *scan infrastructure* could not even stage the
candidate, so we attribute that to infra, never to the model. Any of the
four fields may be ``null`` when its stage never ran; a null stays
tri-state ``None`` rather than being coerced into a definite failure.

This adapter never imports any upstream package into the GEH process. ``stage``
writes the upstream ``generated_code/<model>__<batch>/<id>_cycleN/`` layout +
``processed_instances.json`` from the runner-materialized worktrees;
``evaluate`` shells out to ``invoke.py --run_step security_scan`` through the
injected :class:`EnvProvider`; ``parse`` reads the per-instance scan JSON.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from guard_eval_harness.execution.artifacts import dump_json
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
from guard_eval_harness.vibecoding.materialize import safe_overlay_tree
from guard_eval_harness.vibecoding.oracles.base import OracleAdapter
from guard_eval_harness.vibecoding.registry import oracle_registry
from guard_eval_harness.vibecoding.results import (
    ProvenanceBlock,
    RawBlock,
    VibeTaskResult,
    derive_task_metrics,
)
from guard_eval_harness.vibecoding.safe_path import safe_relpath
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    VibeTask,
)

# Pinned upstream (Tencent/AICGSecEval master carrying v2 dynamic dataset).
ASE_UPSTREAM_URL = "https://github.com/Tencent/AICGSecEval.git"
ASE_UPSTREAM_REF = "94428ebf45141bf4ecd365a51d596dcd51caa690"

# Upstream layout constants.
_DATASET_RELPATH = "data/data_v2.json"
_GENERATED_CODE_DIR = "generated_code"
_PROCESSED_FILE = "processed_instances.json"
_SCAN_RESULTS_DIR = "scan_results"
_SCAN_RESULTS_FILE = "scan_results.json"
_DEFAULT_MODEL = "geh-byo"
_DEFAULT_BATCH = "geh"
# Upstream ``invoke.py --agent`` validates ``--agent_name`` against a fixed
# registry (claude_code/gemini/codex) even for ``--run_step security_scan``,
# where the name is only a directory label (``<agent_name>__<batch>``) and no
# generation happens. We therefore stage + invoke under a fixed valid agent
# name and preserve the real GEH ``artifact.model`` in each result row's
# ``model`` field via the per-task ``meta_index`` (see ``_row``).
_UPSTREAM_AGENT_NAME = "claude_code"

# Hard ceiling for the whole batch security_scan (seconds). The upstream owns
# per-case timeouts; this guards against a hung Docker daemon.
_SCAN_TIMEOUT_S = 7200.0

# Inline EnvSpec (authoritative at runtime; mirrors catalog/ase-dynamic.yaml).
ASE_ENV = EnvSpec(
    name="ase",
    kind="venv",
    upstream_url=ASE_UPSTREAM_URL,
    upstream_ref=ASE_UPSTREAM_REF,
    install=[
        "python -m pip install --upgrade pip",
        "python -m pip install -r requirements.txt",
    ],
    requires_docker=True,
    requires_network_for_eval=True,
    disk_gb_estimate=40.0,
    parallelism=OracleParallelism(
        model="batch_internal",
        default_workers=1,
        max_workers=8,
    ),
    license_policy="vendor_allowed",
    env={"__dataset_files__": _DATASET_RELPATH},
)


def _instance_id(task_id: str) -> str:
    """Strip the ``ase/`` prefix to recover the upstream ``instance_id``."""
    return task_id[len("ase/"):] if task_id.startswith("ase/") else task_id


def _as_opt_bool(val: Any) -> bool | None:
    """Read an upstream cycle field as tri-state: bool, else ``None``.

    The per-cycle scan JSON emits ``null`` for a stage that never ran (e.g.
    ``poc_check`` when the functional check failed first). Coercing that to
    ``False`` would fabricate a definite verdict, so anything non-bool stays
    ``None`` and Kleene-propagates (see ``results.derive_task_metrics``).
    """
    return val if isinstance(val, bool) else None


@oracle_registry.register("ase")
class ASEOracle(OracleAdapter):
    """Wrap the A.S.E dynamic ``security_scan`` as a GEH oracle."""

    name = "ase"
    env = ASE_ENV
    artifact_kinds = {"repo_dir"}
    task_types = {"repo_dir"}
    granularity = "batch"
    capabilities = OracleCapabilities(
        runs_functional_tests=True,
        detects_target_vuln=True,
        detects_new_vuln=False,
        dynamic_pov=True,
        static_analysis=False,
        fuzzing=False,
        llm_judge=False,
        # Docker pulls + container timeouts make verdicts non-bit-identical.
        deterministic=False,
    )
    parallelism = OracleParallelism(
        model="batch_internal",
        default_workers=1,
        max_workers=8,
    )
    parser_version = "ase-1"

    # --- staging ------------------------------------------------------

    def stage(
        self,
        tasks: list[VibeTask],
        artifacts: list[AgentArtifact],
        run_dir: Path,
    ) -> StagedOracleInput:
        """Write the upstream generated-code layout from materialized dirs.

        Builds ``generated_code/<model>__<batch>/<instance_id>_cycle1/`` by
        copying each runner-materialized worktree, plus a
        ``processed_instances.json`` marking every staged cycle ``success``.
        Rejects any non-``repo_dir`` artifact.
        """
        run_dir = Path(run_dir)
        inputs_dir = run_dir / "upstream" / self.name / "inputs"
        by_task = {task.id: task for task in tasks}

        # ``model_name`` is the real GEH model label (kept in result rows);
        # the upstream generated-code directory + ``--agent_name`` flag must
        # use a fixed valid agent name that upstream argparse accepts.
        model_name = self._model_name(artifacts)
        agent_name = _UPSTREAM_AGENT_NAME
        code_root = (
            inputs_dir
            / _GENERATED_CODE_DIR
            / f"{agent_name}__{_DEFAULT_BATCH}"
        )
        # A rerun with the same --run-dir reuses this batch tree. Remove any
        # prior attempt's generated code AND its scan_results subtree so a
        # security_scan that exits nonzero before rewriting a
        # ``<cycle>_output.json`` cannot leave a stale per-cycle verdict that
        # ``parse()`` would read back as the new candidate's result.
        if code_root.exists():
            shutil.rmtree(code_root)
        code_root.mkdir(parents=True, exist_ok=True)

        processed: dict[str, dict[str, Any]] = {}
        meta_index: dict[str, dict[str, Any]] = {}
        task_ids: list[str] = []
        for artifact in artifacts:
            if artifact.kind not in self.artifact_kinds:
                raise UnsupportedArtifactError(
                    f"ase oracle supports {sorted(self.artifact_kinds)}, "
                    f"got kind={artifact.kind!r}"
                )
            task = by_task.get(artifact.task_id)
            if task is None:
                raise UnsupportedArtifactError(
                    f"no task matches artifact task_id={artifact.task_id!r}"
                )
            source = self._worktree_source(artifact, run_dir)
            if source is None or not source.is_dir():
                raise UnsupportedArtifactError(
                    "ase oracle requires a materialized repo_dir worktree; "
                    f"none found for task_id={artifact.task_id!r}"
                )
            instance_id = _instance_id(artifact.task_id)
            cycle_name = f"{instance_id}_cycle1"
            # ``instance_id`` is the candidate's task id minus the ``ase/``
            # prefix and is NOT slug-escaped, so confine the destination under
            # code_root: a task id like ``ase/../escape`` would otherwise climb
            # out of the staging tree.
            dest = safe_relpath(code_root, Path(cycle_name))
            if dest.exists():
                shutil.rmtree(dest)
            # ``source`` may be the candidate-supplied ``artifact.worktree``
            # (untrusted), so copy through the confined, symlink-dereferencing
            # overlay rather than ``copytree(symlinks=True)`` -- no symlink from
            # the generated tree survives into oracle-visible staging.
            dest.mkdir(parents=True, exist_ok=True)
            safe_overlay_tree(source, dest, [])
            processed[cycle_name] = {"success": True, "time": 0}
            meta_index[artifact.task_id] = {
                "instance_id": instance_id,
                "cycle_dirs": [cycle_name],
                "artifact_sha256": artifact_sha256(artifact),
                "task_sha256": task_sha256(task),
                "model": artifact.model,
                "source_dataset": task.source_dataset,
            }
            task_ids.append(artifact.task_id)

        dump_json(code_root / _PROCESSED_FILE, processed)
        return StagedOracleInput(
            adapter_name=self.name,
            inputs_dir=str(inputs_dir),
            task_ids=task_ids,
            metadata={
                "model_name": model_name,
                "agent_name": agent_name,
                "batch_id": _DEFAULT_BATCH,
                "generated_code_dir": str(inputs_dir / _GENERATED_CODE_DIR),
                "code_root": str(code_root),
                "meta_index": meta_index,
            },
        )

    @staticmethod
    def _model_name(artifacts: list[AgentArtifact]) -> str:
        """Pick a stable model key for the generated-code dir name."""
        for artifact in artifacts:
            if artifact.model:
                return str(artifact.model).replace("/", "__")
        return _DEFAULT_MODEL

    @staticmethod
    def _worktree_source(
        artifact: AgentArtifact, run_dir: Path
    ) -> Path | None:
        """Locate the worktree the oracle should score for one artifact.

        Prefers the runner's materialized layout under
        ``run_dir/artifacts/<safe_task_id>/materialized-worktree/``: that tree
        is the base checkout with the candidate overlaid through the confined,
        symlink-dereferencing materializer, so it is the sealed view the cache
        key (``worktree_sha256``) is computed over. Only when no materialized
        worktree exists (e.g. an oracle invoked directly on a self-contained
        ``artifact.worktree`` without the runner's materialization step) does
        it fall back to the artifact's explicit ``worktree`` path.
        """
        from guard_eval_harness.vibecoding.run_store import safe_task_id

        materialized = (
            run_dir
            / "artifacts"
            / safe_task_id(artifact.task_id)
            / "materialized-worktree"
        )
        if materialized.is_dir():
            return materialized
        if artifact.worktree:
            candidate = Path(artifact.worktree)
            if candidate.is_dir():
                return candidate
        return materialized

    # --- evaluation ---------------------------------------------------

    def evaluate(
        self,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
        env_provider: Any,
    ) -> RawOracleResult:
        """Run the upstream ``security_scan`` path via the env provider.

        Acquires the upstream checkout + venv, then invokes
        ``invoke.py --run_step security_scan`` out of process with the staged
        ``generated_code`` directory wired in via ``--output_dir``. Locates
        the per-instance scan results and returns their on-disk paths.
        """
        resolved = env_provider.ensure_ready()
        run_dir = Path(run_config.run_dir)

        model_name = str(staged.metadata.get("model_name", _DEFAULT_MODEL))
        # The upstream ``--agent_name`` must be a registered agent; it only
        # labels the generated-code dir for ``security_scan``. The real model
        # is preserved separately in each result row (see ``meta_index``).
        agent_name = str(
            staged.metadata.get("agent_name", _UPSTREAM_AGENT_NAME)
        )
        batch_id = str(staged.metadata.get("batch_id", _DEFAULT_BATCH))
        inputs_dir = Path(staged.inputs_dir)
        dataset_path = str(Path(resolved.upstream_dir) / _DATASET_RELPATH)
        workers = self._workers(resource_budget)

        argv = [
            str(resolved.venv_python),
            "invoke.py",
            "--agent",
            "--agent_name",
            agent_name,
            "--batch_id",
            batch_id,
            "--run_step",
            "security_scan",
            "--output_dir",
            str(inputs_dir),
            "--dataset_path",
            dataset_path,
            "--max_workers",
            str(workers),
        ]

        result = env_provider.run(
            argv,
            run_dir=run_dir,
            timeout_s=_SCAN_TIMEOUT_S,
            budget=resource_budget,
        )

        code_dir = (
            inputs_dir
            / _GENERATED_CODE_DIR
            / f"{agent_name}__{batch_id}"
        )
        outputs_dir = code_dir / _SCAN_RESULTS_DIR
        logs_dir = run_dir / "upstream" / self.name / "logs"
        return RawOracleResult(
            adapter_name=self.name,
            outputs_dir=str(outputs_dir),
            logs_dir=str(logs_dir),
            exit_code=getattr(result, "returncode", None),
            task_ids=list(staged.task_ids),
            metadata={
                "code_dir": str(code_dir),
                "merged_results": str(code_dir / _SCAN_RESULTS_FILE),
                "model_name": model_name,
                "batch_id": batch_id,
                "workers": workers,
                "timed_out": bool(getattr(result, "timed_out", False)),
                "meta_index": staged.metadata.get("meta_index", {}),
                "upstream_command": list(argv),
                "upstream_workdir": str(resolved.workdir),
                "upstream_url": resolved.upstream_url,
                "upstream_ref": resolved.upstream_ref,
            },
        )

    def _workers(self, budget: ResourceBudget) -> int:
        """Clamp the runner budget to this oracle's worker bound."""
        return max(
            1, min(int(budget.max_workers), self.parallelism.max_workers)
        )

    # --- parsing ------------------------------------------------------

    def parse(self, raw: RawOracleResult) -> list[VibeTaskResult]:
        """Map per-instance scan JSON to normalized result rows.

        For each task we read every ``<instance_id>_cycleN_output.json`` under
        the scan-results dir, summarize the per-cycle booleans into one row
        (the headline cycle is the first), and surface per-cycle stability
        (``num_cycles`` + per-cycle poc results) into ``raw.extra`` for trial
        reporting.
        """
        outputs_dir = Path(raw.outputs_dir)
        meta_index: dict[str, Any] = dict(raw.metadata.get("meta_index", {}))
        timed_out = bool(raw.metadata.get("timed_out", False))
        rows: list[VibeTaskResult] = []
        for task_id in raw.task_ids:
            meta = meta_index.get(task_id, {})
            cycles = self._read_cycles(outputs_dir, task_id, meta)
            rows.append(
                self._row(task_id, meta, cycles, raw, timed_out=timed_out)
            )
        return rows

    def _read_cycles(
        self,
        outputs_dir: Path,
        task_id: str,
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Load every cycle's scan-output JSON for one instance, in order."""
        instance_id = meta.get("instance_id") or _instance_id(task_id)
        named = meta.get("cycle_dirs") or [f"{instance_id}_cycle1"]
        found: list[dict[str, Any]] = []
        for cycle_name in named:
            path = outputs_dir / f"{cycle_name}_output.json"
            payload = self._load_json(path)
            if payload is not None:
                payload.setdefault("instance_id", cycle_name)
                found.append(payload)
        if found:
            return found
        # Fall back to globbing any cycle outputs for this instance id.
        pattern = f"{instance_id}_cycle*_output.json"
        for path in sorted(outputs_dir.glob(pattern)):
            payload = self._load_json(path)
            if payload is not None:
                payload.setdefault(
                    "instance_id", path.name[: -len("_output.json")]
                )
                found.append(payload)
        return found

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        """Read a scan-output JSON object, or ``None`` if absent/invalid."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _row(
        self,
        task_id: str,
        meta: dict[str, Any],
        cycles: list[dict[str, Any]],
        raw: RawOracleResult,
        *,
        timed_out: bool,
    ) -> VibeTaskResult:
        """Build one normalized row from an instance's cycle scan outputs."""
        model = str(meta.get("model") or raw.metadata.get("model_name")
                    or _DEFAULT_MODEL)
        source_dataset = str(meta.get("source_dataset") or "ase")

        status = "completed"
        failure_origin = "none"
        failure_reason: str | None = None
        build_pass: bool | None = None
        functional_pass: bool | None = None
        security_oracle_pass: bool | None = None
        known_vuln_present: bool | None = None

        if not cycles:
            # No scan output at all for this instance -> infra problem
            # (the scan never produced a verdict), unless we timed out.
            status = "infra_failure"
            failure_origin = "infra"
            failure_reason = "oracle_timeout" if timed_out else (
                "verifier_unavailable"
            )
            headline: dict[str, Any] = {}
        else:
            headline = cycles[0]
            completion = _as_opt_bool(headline.get("completion"))
            startup = _as_opt_bool(headline.get("image_status_check"))
            test = _as_opt_bool(headline.get("test_case_check"))
            poc = _as_opt_bool(headline.get("poc_check"))

            # Identity checks (`is False`) keep the tri-state contract: only
            # a definite upstream False reaches a failure branch. A null gate
            # (stage never ran / no verdict) leaves the row 'completed' with
            # ``None`` signals, which Kleene-propagate so the row drops out of
            # the affected denominators (see ``results.derive_task_metrics``)
            # instead of being recorded as a fabricated definite failure.
            if completion is False:
                # Generated code could not be staged into the container or
                # failed md5 verification: scan infrastructure failure, never
                # a model-quality signal.
                status = "infra_failure"
                failure_origin = "infra"
                failure_reason = "verifier_unavailable"
            else:
                # Startup gates everything downstream. A failed startup after a
                # successful copy is a genuine build/startup failure of the
                # candidate -> attribute to the model.
                build_pass = startup
                if startup is False:
                    status = "model_failure"
                    failure_origin = "model"
                    failure_reason = "build_failed"
                else:
                    functional_pass = test
                    security_oracle_pass = poc
                    known_vuln_present = None if poc is None else not poc
                    if test is False:
                        failure_reason = "functional_tests_failed"
                    elif poc is False:
                        failure_reason = "target_vuln_present"

        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model,
            status=status,
            failure_origin=failure_origin,
            failure_reason=failure_reason,
            # A.S.E regenerates the masked file rather than applying a diff;
            # there is no separate patch-apply gate, so leave it unset.
            patch_applied=None,
            build_pass=build_pass,
            functional_pass=functional_pass,
            security_oracle_pass=security_oracle_pass,
            known_vuln_present=known_vuln_present,
            new_vuln_introduced=None,
            oracle_capabilities=self.capabilities,
            raw=self._raw_block(headline, cycles, raw),
            provenance=self._provenance(meta, raw),
        )
        return derive_task_metrics(result)

    def _raw_block(
        self,
        headline: dict[str, Any],
        cycles: list[dict[str, Any]],
        raw: RawOracleResult,
    ) -> RawBlock:
        """Preserve verbatim upstream tri-state signals + cycle stability.

        Audit output mirrors the tri-state contract in ``_row``: a stage the
        upstream skipped (JSON ``null``) stays ``None`` here too, so the raw
        block never shows a fabricated ``poc=False`` for a row whose
        normalized verdict was deliberately left unscored.
        """
        poc_series = [_as_opt_bool(c.get("poc_check")) for c in cycles]
        if headline:
            upstream_status = (
                f"completion={_as_opt_bool(headline.get('completion'))} "
                f"startup="
                f"{_as_opt_bool(headline.get('image_status_check'))} "
                f"test={_as_opt_bool(headline.get('test_case_check'))} "
                f"poc={_as_opt_bool(headline.get('poc_check'))}"
            )
        else:
            upstream_status = "no_scan_output"
        return RawBlock(
            upstream_status=upstream_status,
            upstream_result_path=raw.outputs_dir,
            logs_dir=raw.logs_dir,
            extra={
                "num_cycles": len(cycles),
                "poc_check_per_cycle": poc_series,
                "image_status_check": _as_opt_bool(
                    headline.get("image_status_check")
                ),
                "test_case_check": _as_opt_bool(
                    headline.get("test_case_check")
                ),
                "completion": _as_opt_bool(headline.get("completion")),
                "cycle_outputs": cycles,
            },
        )

    def _provenance(
        self, meta: dict[str, Any], raw: RawOracleResult
    ) -> ProvenanceBlock:
        """Attach reproduction metadata from the staged/evaluated batch."""
        return ProvenanceBlock(
            adapter_name=self.name,
            parser_version=self.parser_version,
            upstream_url=raw.metadata.get("upstream_url", ASE_UPSTREAM_URL),
            upstream_ref=raw.metadata.get("upstream_ref", ASE_UPSTREAM_REF),
            upstream_command=list(raw.metadata.get("upstream_command", [])),
            upstream_workdir=raw.metadata.get("upstream_workdir"),
            worker_count=raw.metadata.get("workers"),
            artifact_sha256=meta.get("artifact_sha256"),
            task_sha256=meta.get("task_sha256"),
        )
