"""In-memory mock oracle adapter + env provider.

This adapter never touches Docker, a venv, or the network. It derives a
deterministic per-task outcome from each artifact (either an explicit
``metadata["mock_outcome"]`` hint or, failing that, the artifact's sha256) and
emits normalized result rows. Together with :class:`MockTaskSource` it lets the
conformance harness, runner, and CLI run end-to-end with zero infrastructure.

``InMemoryEnvProvider`` satisfies the :class:`EnvProvider` protocol in-process:
instead of shelling out, it reads the staged outcome map and writes a results
file the adapter's ``parse`` consumes -- exercising the same stage/evaluate/
parse seam a real out-of-process provider would.
"""

from __future__ import annotations

import json
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
from guard_eval_harness.vibecoding.oracles.base import OracleAdapter
from guard_eval_harness.vibecoding.registry import oracle_registry
from guard_eval_harness.vibecoding.results import (
    VibeTaskResult,
    derive_task_metrics,
)
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    VibeTask,
)

# Recognized explicit outcome hints (via ``AgentArtifact.metadata``).
_OUTCOMES = {"infra", "null_functional", "secure_pass", "model_failure"}

# Name of the staged + emitted JSON files within the run dir.
_STAGED_FILE = "staged_outcomes.json"
_RESULTS_FILE = "results.json"


def _resolve_outcome(artifact: AgentArtifact) -> str:
    """Map an artifact to one of the four mock outcomes.

    Honors an explicit ``metadata["mock_outcome"]`` hint; otherwise derives a
    deterministic pass/fail from the artifact sha256 so reruns are stable.
    """
    hint = artifact.metadata.get("mock_outcome")
    if isinstance(hint, str) and hint in _OUTCOMES:
        return hint
    digest = artifact_sha256(artifact)
    # Even final nibble => secure pass; odd => functional-null (a benign,
    # non-infra, non-model split so the default fixture covers both rates).
    return "secure_pass" if int(digest[-1], 16) % 2 == 0 else "null_functional"


@oracle_registry.register("mock")
class MockOracleAdapter(OracleAdapter):
    """A deterministic, infrastructure-free oracle for conformance tests."""

    name = "mock"
    env = EnvSpec(
        name="mock",
        kind="inline",
        license_policy="vendor_allowed",
        parallelism=OracleParallelism(
            model="batch_internal",
            default_workers=1,
            max_workers=4,
        ),
    )
    artifact_kinds = {"patch"}
    task_types = {"repo_patch"}
    granularity = "batch"
    capabilities = OracleCapabilities(
        runs_functional_tests=True,
        detects_target_vuln=True,
        detects_new_vuln=False,
        dynamic_pov=False,
        static_analysis=False,
        fuzzing=False,
        llm_judge=False,
        deterministic=True,
    )
    parallelism = OracleParallelism(
        model="batch_internal",
        default_workers=1,
        max_workers=4,
    )
    parser_version = "mock-1"

    def stage(
        self,
        tasks: list[VibeTask],
        artifacts: list[AgentArtifact],
        run_dir: Path,
    ) -> StagedOracleInput:
        """Record per-task outcomes; reject non-``patch`` artifacts."""
        inputs_dir = Path(run_dir) / "upstream" / self.name / "inputs"
        by_task = {task.id: task for task in tasks}
        outcomes: dict[str, dict[str, Any]] = {}
        task_ids: list[str] = []
        for artifact in artifacts:
            if artifact.kind not in self.artifact_kinds:
                raise UnsupportedArtifactError(
                    f"mock oracle supports {sorted(self.artifact_kinds)}, "
                    f"got kind={artifact.kind!r}"
                )
            task = by_task.get(artifact.task_id)
            if task is None:
                raise UnsupportedArtifactError(
                    f"no task matches artifact task_id={artifact.task_id!r}"
                )
            outcomes[artifact.task_id] = {
                "outcome": _resolve_outcome(artifact),
                "artifact_sha256": artifact_sha256(artifact),
                "task_sha256": task_sha256(task),
                "model": artifact.model,
                "source_dataset": task.source_dataset,
            }
            task_ids.append(artifact.task_id)
        dump_json(inputs_dir / _STAGED_FILE, {"outcomes": outcomes})
        return StagedOracleInput(
            adapter_name=self.name,
            inputs_dir=str(inputs_dir),
            task_ids=task_ids,
            metadata={"staged_file": _STAGED_FILE},
        )

    def evaluate(
        self,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
        env_provider: Any,
    ) -> RawOracleResult:
        """Delegate to the injected env provider (no in-adapter spawning)."""
        return env_provider.evaluate(
            self.env,
            staged,
            run_config,
            resource_budget,
        )

    def parse(self, raw: RawOracleResult) -> list[VibeTaskResult]:
        """Map the env provider's results file to normalized rows."""
        results_path = Path(raw.outputs_dir) / _RESULTS_FILE
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsupportedArtifactError(
                f"mock results unreadable at {results_path}: {exc}"
            ) from exc
        rows: list[VibeTaskResult] = []
        for task_id in raw.task_ids:
            entry = payload.get(task_id, {})
            rows.append(self._row(task_id, entry, raw))
        return rows

    def _row(
        self,
        task_id: str,
        entry: dict[str, Any],
        raw: RawOracleResult,
    ) -> VibeTaskResult:
        """Build a single normalized result row from one outcome entry."""
        outcome = entry.get("outcome", "null_functional")
        model = entry.get("model", "mock-model")
        source_dataset = entry.get("source_dataset", "mock")

        status = "completed"
        failure_origin = "none"
        failure_reason: str | None = None
        patch_applied: bool | None = True
        build_pass: bool | None = True
        functional_pass: bool | None = None
        security_oracle_pass: bool | None = None
        known_vuln_present: bool | None = None
        upstream_status = outcome

        if outcome == "infra":
            status = "infra_failure"
            failure_origin = "infra"
            failure_reason = "oracle_timeout"
            patch_applied = None
            build_pass = None
        elif outcome == "model_failure":
            status = "model_failure"
            failure_origin = "model"
            failure_reason = "empty_diff"
            patch_applied = False
            build_pass = None
        elif outcome == "secure_pass":
            functional_pass = True
            security_oracle_pass = True
            known_vuln_present = False
        else:  # null_functional
            functional_pass = None
            security_oracle_pass = None
            known_vuln_present = None

        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model,
            status=status,
            failure_origin=failure_origin,
            failure_reason=failure_reason,
            patch_applied=patch_applied,
            build_pass=build_pass,
            functional_pass=functional_pass,
            security_oracle_pass=security_oracle_pass,
            known_vuln_present=known_vuln_present,
            new_vuln_introduced=None,
            oracle_capabilities=self.capabilities,
            raw=self._raw_block(upstream_status, raw),
            provenance=self._provenance(entry, raw),
        )
        return derive_task_metrics(result)

    def _raw_block(self, upstream_status: str, raw: RawOracleResult):
        """Preserve verbatim upstream status + output paths under ``raw``."""
        from guard_eval_harness.vibecoding.results import RawBlock

        return RawBlock(
            upstream_status=upstream_status,
            upstream_result_path=str(
                Path(raw.outputs_dir) / _RESULTS_FILE
            ),
            logs_dir=raw.logs_dir,
        )

    def _provenance(self, entry: dict[str, Any], raw: RawOracleResult):
        """Attach minimal provenance for the mock oracle."""
        from guard_eval_harness.vibecoding.results import ProvenanceBlock

        return ProvenanceBlock(
            adapter_name=self.name,
            parser_version=self.parser_version,
            upstream_command=["python", "-m", "mock.evaluate"],
            artifact_sha256=entry.get("artifact_sha256"),
            task_sha256=entry.get("task_sha256"),
        )


class InMemoryEnvProvider:
    """In-process :class:`EnvProvider`: computes results without subprocess.

    Reads the staged outcome map written by ``MockOracleAdapter.stage`` and
    writes a per-task ``results.json`` into the run dir, returning paths in a
    :class:`RawOracleResult`. This stands in for the real
    :mod:`guard_eval_harness.vibecoding.envs` provider so this stage has zero
    sibling dependencies.
    """

    def evaluate(
        self,
        env: Any,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
    ) -> RawOracleResult:
        """Translate staged outcomes into an on-disk results file."""
        staged_file = staged.metadata.get("staged_file", _STAGED_FILE)
        staged_path = Path(staged.inputs_dir) / staged_file
        staged_payload = json.loads(
            staged_path.read_text(encoding="utf-8")
        )
        outcomes = staged_payload.get("outcomes", {})

        run_dir = Path(run_config.run_dir)
        outputs_dir = run_dir / "upstream" / staged.adapter_name / "outputs"
        logs_dir = run_dir / "upstream" / staged.adapter_name / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {}
        for task_id in staged.task_ids:
            results[task_id] = outcomes.get(task_id, {})

        dump_json(outputs_dir / _RESULTS_FILE, results)
        # Honor the worker budget the runner passed (cosmetic for the mock).
        workers = min(
            resource_budget.max_workers,
            self_max_workers(env),
        )
        return RawOracleResult(
            adapter_name=staged.adapter_name,
            outputs_dir=str(outputs_dir),
            logs_dir=str(logs_dir),
            exit_code=0,
            task_ids=list(staged.task_ids),
            metadata={"workers": workers},
        )


def self_max_workers(env: Any) -> int:
    """Best-effort read of an env spec's max worker bound (default 1)."""
    parallelism = getattr(env, "parallelism", None)
    if parallelism is None:
        return 1
    return int(getattr(parallelism, "max_workers", 1))
