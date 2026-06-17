"""Contract tests for the vibecoding core type foundation."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from guard_eval_harness.vibecoding.artifacts import (
    AgentArtifact,
    artifact_sha256,
    task_sha256,
)
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
    RepoSpec,
    ResourceBudget,
    ResourceEstimate,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)


class SchemaConstructionTest(unittest.TestCase):
    """Valid construction + JSON round-trip of every model."""

    def _round_trip(self, model) -> None:
        data = model.model_dump(mode="json")
        rebuilt = type(model).model_validate(data)
        self.assertEqual(rebuilt.model_dump(mode="json"), data)

    def test_vibe_task_round_trip(self) -> None:
        task = VibeTask(
            id="susvibes/example",
            source_dataset="susvibes",
            task_type="repo_patch",
            instructions="Implement the requested feature.",
            repo=RepoSpec(
                url="https://github.com/example/project",
                base_commit="abc123",
                workdir=".",
            ),
            labels=TaskLabels(cwe=["CWE-22"], cve=[]),
            environment=TaskEnvironmentRef(
                oracle="susvibes", requires_docker=True
            ),
        )
        self.assertEqual(task.id, "susvibes/example")
        self._round_trip(task)

    def test_env_spec_round_trip(self) -> None:
        env = EnvSpec(
            name="susvibes",
            kind="venv",
            upstream_url="https://github.com/LeiLiLab/susvibes",
            upstream_ref="dd28a7e",
            resource_estimate=ResourceEstimate(
                cpu_per_worker=4, memory_gb_per_worker=4.0
            ),
            parallelism=OracleParallelism(
                model="batch_internal", default_workers=4, max_workers=8
            ),
            license_policy="vendor_allowed",
            env={"SEMGREP_APP_TOKEN": "x"},
        )
        self.assertEqual(env.parallelism.max_workers, 8)
        self._round_trip(env)

    def test_capabilities_and_budget_round_trip(self) -> None:
        caps = OracleCapabilities(
            runs_functional_tests=True,
            detects_target_vuln=True,
            dynamic_pov=True,
            deterministic=True,
        )
        budget = ResourceBudget(
            max_workers=4, cpu_cores=16, memory_gb=64.0, disk_gb=500.0
        )
        self._round_trip(caps)
        self._round_trip(budget)

    def test_result_round_trip(self) -> None:
        result = VibeTaskResult(
            task_id="susvibes/example",
            source_dataset="susvibes",
            model="claude-code",
            status="completed",
            functional_pass=True,
            security_oracle_pass=False,
            known_vuln_present=True,
            raw=RawBlock(upstream_status="model_patch_error"),
            provenance=ProvenanceBlock(parser_version="1"),
        )
        self._round_trip(result)


class ForbidExtraTest(unittest.TestCase):
    """extra='forbid' rejects unknown keys across the models."""

    def test_vibe_task_rejects_unknown(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTask(
                id="a", source_dataset="b", nonsense=True
            )  # type: ignore[call-arg]

    def test_env_spec_rejects_unknown(self) -> None:
        with self.assertRaises(ValidationError):
            EnvSpec(name="x", bogus=1)  # type: ignore[call-arg]

    def test_result_rejects_unknown(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTaskResult(
                task_id="a",
                source_dataset="b",
                model="m",
                secure_success=True,  # type: ignore[call-arg]
            )


class EnumRejectionTest(unittest.TestCase):
    """Each Literal enum rejects an out-of-set value."""

    def test_task_type_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTask(
                id="a",
                source_dataset="b",
                task_type="not_a_type",  # type: ignore[arg-type]
            )

    def test_env_kind_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            EnvSpec(name="x", kind="docker")  # type: ignore[arg-type]

    def test_parallelism_model_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            OracleParallelism(model="parallel")  # type: ignore[arg-type]

    def test_license_policy_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            EnvSpec(name="x", license_policy="gpl")  # type: ignore[arg-type]

    def test_status_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTaskResult(
                task_id="a",
                source_dataset="b",
                model="m",
                status="exploded",  # type: ignore[arg-type]
            )

    def test_failure_origin_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTaskResult(
                task_id="a",
                source_dataset="b",
                model="m",
                failure_origin="cosmic_ray",  # type: ignore[arg-type]
            )

    def test_failure_reason_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTaskResult(
                task_id="a",
                source_dataset="b",
                model="m",
                failure_reason="gremlins",  # type: ignore[arg-type]
            )


class ValidatorTest(unittest.TestCase):
    """Custom model validators."""

    def test_empty_id_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTask(id="   ", source_dataset="b")

    def test_empty_source_dataset_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VibeTask(id="a", source_dataset="")

    def test_parallelism_max_lt_default_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            OracleParallelism(default_workers=8, max_workers=4)

    def test_parallelism_max_eq_default_allowed(self) -> None:
        p = OracleParallelism(default_workers=4, max_workers=4)
        self.assertEqual(p.max_workers, 4)


class TriStateVerdictTest(unittest.TestCase):
    """Tri-state verdicts round-trip True/False/None."""

    def test_verdict_round_trip(self) -> None:
        for value in (True, False, None):
            result = VibeTaskResult(
                task_id="a",
                source_dataset="b",
                model="m",
                functional_pass=value,
            )
            data = result.model_dump(mode="json")
            self.assertEqual(data["functional_pass"], value)
            rebuilt = VibeTaskResult.model_validate(data)
            self.assertIs(rebuilt.functional_pass, value)

    def test_verdicts_default_none(self) -> None:
        result = VibeTaskResult(
            task_id="a", source_dataset="b", model="m"
        )
        for field in (
            "patch_applied",
            "build_pass",
            "functional_pass",
            "security_oracle_pass",
            "known_vuln_present",
            "new_vuln_introduced",
            "target_secure_success",
            "strict_secure_success",
        ):
            self.assertIsNone(getattr(result, field), field)


class ArtifactValidatorTest(unittest.TestCase):
    """AgentArtifact kind/payload validation + round-trip."""

    def test_patch_round_trip(self) -> None:
        art = AgentArtifact(
            task_id="t", model="m", kind="patch", patch="diff --git a b"
        )
        data = art.model_dump(mode="json")
        self.assertEqual(AgentArtifact.model_validate(data), art)

    def test_patch_without_patch_raises(self) -> None:
        with self.assertRaises(ValidationError):
            AgentArtifact(task_id="t", model="m", kind="patch")

    def test_full_file_empty_files_raises(self) -> None:
        with self.assertRaises(ValidationError):
            AgentArtifact(
                task_id="t", model="m", kind="full_file", files={}
            )

    def test_full_file_with_files_ok(self) -> None:
        art = AgentArtifact(
            task_id="t",
            model="m",
            kind="full_file",
            files={"a.py": "x = 1"},
        )
        self.assertEqual(art.files["a.py"], "x = 1")

    def test_completion_without_completion_raises(self) -> None:
        with self.assertRaises(ValidationError):
            AgentArtifact(task_id="t", model="m", kind="completion")

    def test_repo_dir_without_worktree_raises(self) -> None:
        with self.assertRaises(ValidationError):
            AgentArtifact(task_id="t", model="m", kind="repo_dir")

    def test_artifact_kind_rejects_out_of_set(self) -> None:
        with self.assertRaises(ValidationError):
            AgentArtifact(
                task_id="t", model="m", kind="zipfile"  # type: ignore
            )


class ArtifactHashTest(unittest.TestCase):
    """artifact_sha256 stability and sensitivity."""

    def _files_artifact(self, files: dict[str, str]) -> AgentArtifact:
        return AgentArtifact(
            task_id="t", model="m", kind="full_file", files=files
        )

    def test_hash_stable_across_reordered_keys(self) -> None:
        a = self._files_artifact({"a.py": "1", "b.py": "2"})
        b = self._files_artifact({"b.py": "2", "a.py": "1"})
        self.assertEqual(artifact_sha256(a), artifact_sha256(b))

    def test_hash_differs_on_payload_change(self) -> None:
        a = self._files_artifact({"a.py": "1"})
        b = self._files_artifact({"a.py": "2"})
        self.assertNotEqual(artifact_sha256(a), artifact_sha256(b))

    def test_task_hash_stable(self) -> None:
        t1 = VibeTask(
            id="x", source_dataset="d", labels=TaskLabels(cwe=["CWE-1"])
        )
        t2 = VibeTask(
            id="x", source_dataset="d", labels=TaskLabels(cwe=["CWE-1"])
        )
        self.assertEqual(task_sha256(t1), task_sha256(t2))


class DeriveMetricsTest(unittest.TestCase):
    """derive_task_metrics null-propagation rules."""

    def _result(self, **kwargs) -> VibeTaskResult:
        base = dict(task_id="a", source_dataset="b", model="m")
        base.update(kwargs)
        return VibeTaskResult(**base)

    def test_target_secure_true(self) -> None:
        r = derive_task_metrics(
            self._result(functional_pass=True, security_oracle_pass=True)
        )
        self.assertIs(r.target_secure_success, True)

    def test_target_secure_false(self) -> None:
        r = derive_task_metrics(
            self._result(functional_pass=True, security_oracle_pass=False)
        )
        self.assertIs(r.target_secure_success, False)

    def test_functional_none_propagates_to_target(self) -> None:
        r = derive_task_metrics(
            self._result(functional_pass=None, security_oracle_pass=True)
        )
        self.assertIsNone(r.target_secure_success)
        self.assertIsNone(r.strict_secure_success)

    def test_new_vuln_none_propagates_to_strict_only(self) -> None:
        r = derive_task_metrics(
            self._result(
                functional_pass=True,
                security_oracle_pass=True,
                new_vuln_introduced=None,
            )
        )
        self.assertIs(r.target_secure_success, True)
        self.assertIsNone(r.strict_secure_success)

    def test_strict_secure_true_when_no_new_vuln(self) -> None:
        r = derive_task_metrics(
            self._result(
                functional_pass=True,
                security_oracle_pass=True,
                new_vuln_introduced=False,
            )
        )
        self.assertIs(r.strict_secure_success, True)

    def test_strict_secure_false_when_new_vuln(self) -> None:
        r = derive_task_metrics(
            self._result(
                functional_pass=True,
                security_oracle_pass=True,
                new_vuln_introduced=True,
            )
        )
        self.assertIs(r.target_secure_success, True)
        self.assertIs(r.strict_secure_success, False)

    # --- Kleene-AND: False dominates AND, even over an unknown (None) gate. ---

    def test_target_secure_false_when_functional_false_security_none(
        self,
    ) -> None:
        # functional already failed -> definite False regardless of unknown sec.
        r = derive_task_metrics(
            self._result(
                functional_pass=False, security_oracle_pass=None
            )
        )
        self.assertIs(r.target_secure_success, False)
        self.assertIs(r.strict_secure_success, False)

    def test_target_secure_false_when_functional_none_security_false(
        self,
    ) -> None:
        # security already failed -> definite False regardless of unknown func.
        r = derive_task_metrics(
            self._result(
                functional_pass=None, security_oracle_pass=False
            )
        )
        self.assertIs(r.target_secure_success, False)
        self.assertIs(r.strict_secure_success, False)

    def test_target_secure_none_when_both_none(self) -> None:
        # Nothing definitely failed, but both gates unknown -> indeterminate.
        r = derive_task_metrics(
            self._result(
                functional_pass=None, security_oracle_pass=None
            )
        )
        self.assertIsNone(r.target_secure_success)
        self.assertIsNone(r.strict_secure_success)

    def test_target_secure_none_when_functional_true_security_none(
        self,
    ) -> None:
        # True AND None -> None (no failure yet, one gate unknown).
        r = derive_task_metrics(
            self._result(
                functional_pass=True, security_oracle_pass=None
            )
        )
        self.assertIsNone(r.target_secure_success)
        self.assertIsNone(r.strict_secure_success)

    def test_strict_secure_false_when_functional_false_rest_none(self) -> None:
        # functional failed -> strict is a definite False even though both
        # security_oracle_pass and new_vuln_introduced are unknown (None).
        r = derive_task_metrics(
            self._result(
                functional_pass=False,
                security_oracle_pass=None,
                new_vuln_introduced=None,
            )
        )
        self.assertIs(r.strict_secure_success, False)

    def test_strict_secure_none_when_new_vuln_none_others_true(self) -> None:
        # func & sec True but new_vuln unknown -> strict stays None.
        r = derive_task_metrics(
            self._result(
                functional_pass=True,
                security_oracle_pass=True,
                new_vuln_introduced=None,
            )
        )
        self.assertIsNone(r.strict_secure_success)


class NoFlatSecureSuccessTest(unittest.TestCase):
    """There must be no flat secure_success attribute or field."""

    def test_no_secure_success_attribute(self) -> None:
        result = VibeTaskResult(
            task_id="a", source_dataset="b", model="m"
        )
        self.assertFalse(hasattr(result, "secure_success"))
        self.assertNotIn("secure_success", VibeTaskResult.model_fields)


if __name__ == "__main__":
    unittest.main()
