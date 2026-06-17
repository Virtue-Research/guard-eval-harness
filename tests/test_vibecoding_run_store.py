"""Tests for the vibecoding run-store layout + sha256 helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.provenance import (
    Provenance,
    build_provenance,
    git_commit,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.run_store import (
    append_result,
    compute_artifact_sha256,
    compute_task_sha256,
    ensure_vibe_run_layout,
    safe_task_id,
    task_artifact_dir,
    upstream_dir,
    write_agent_artifact,
    write_manifest,
    write_results,
    write_run_config,
    write_tasks,
)
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    ResourceBudget,
    VibeTask,
)


class RunLayoutTest(unittest.TestCase):
    """ensure_vibe_run_layout + path helpers."""

    def test_layout_creates_subtrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(Path(tmp) / "run1")
            self.assertTrue((run / "artifacts").is_dir())
            self.assertTrue((run / "upstream").is_dir())

    def test_safe_task_id_replaces_slash(self) -> None:
        self.assertEqual(safe_task_id("susvibes/inst-1"), "susvibes__inst-1")

    def test_task_artifact_dir_uses_safe_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = task_artifact_dir(tmp, "susvibes/inst-1")
            self.assertEqual(d.name, "susvibes__inst-1")
            self.assertEqual(d.parent.name, "artifacts")

    def test_upstream_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = upstream_dir(tmp, "susvibes")
            self.assertEqual(d.name, "susvibes")
            self.assertEqual(d.parent.name, "upstream")


class WriteHelpersTest(unittest.TestCase):
    """run_config/manifest/tasks/results writers."""

    def _task(self) -> VibeTask:
        return VibeTask(id="susvibes/inst-1", source_dataset="susvibes")

    def _result(self) -> VibeTaskResult:
        return VibeTaskResult(
            task_id="susvibes/inst-1",
            source_dataset="susvibes",
            model="m",
            status="completed",
        )

    def test_write_run_config_redacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            path = write_run_config(
                run, {"env": {"SEMGREP_APP_TOKEN": "shh"}, "limit": 3}
            )
            body = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(
                body["env"]["SEMGREP_APP_TOKEN"], "shh"
            )
            self.assertEqual(body["limit"], 3)

    def test_write_manifest_redacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            path = write_manifest(run, {"api_key": "secret", "run": "x"})
            body = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(body["api_key"], "secret")
            self.assertEqual(body["run"], "x")

    def test_write_tasks_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            path = write_tasks(run, [self._task()])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["id"], "susvibes/inst-1")

    def test_write_results_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            path = write_results(run, [self._result(), self._result()])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_append_result_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            append_result(run, self._result())
            path = append_result(run, self._result())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)


class WriteAgentArtifactTest(unittest.TestCase):
    """write_agent_artifact materializes patch/files + artifact.json."""

    def test_patch_artifact_writes_patch_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            art = AgentArtifact(
                task_id="susvibes/inst-1",
                model="m",
                kind="patch",
                patch="diff --git a/x b/x\n",
            )
            refs = write_agent_artifact(run, art)
            self.assertTrue(Path(refs.patch_path).name == "agent.patch")
            self.assertTrue(Path(refs.patch_path).exists())
            self.assertTrue(Path(refs.artifact_json).exists())
            self.assertEqual(
                Path(refs.artifact_json).name, "artifact.json"
            )
            reloaded = json.loads(
                Path(refs.artifact_json).read_text(encoding="utf-8")
            )
            self.assertEqual(reloaded["kind"], "patch")
            self.assertEqual(
                refs.artifact_sha256, compute_artifact_sha256(art)
            )

    def test_full_file_artifact_writes_agent_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            art = AgentArtifact(
                task_id="ds/inst-2",
                model="m",
                kind="full_file",
                files={"pkg/a.py": "x = 1", "b.py": "y = 2"},
            )
            refs = write_agent_artifact(run, art)
            self.assertIsNone(refs.patch_path)
            self.assertTrue(Path(refs.files_dir).name == "agent-files")
            self.assertEqual(
                Path(refs.file_paths["pkg/a.py"]).read_text(
                    encoding="utf-8"
                ),
                "x = 1",
            )

    def test_artifact_dir_uses_safe_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            art = AgentArtifact(
                task_id="ds/with/slashes",
                model="m",
                kind="patch",
                patch="p",
            )
            refs = write_agent_artifact(run, art)
            # artifacts/<safe_task_id>/<candidate>/ : the task dir is the
            # parent of the per-candidate dir.
            art_dir = Path(refs.artifact_dir)
            self.assertEqual(art_dir.parent.name, "ds__with__slashes")
            self.assertTrue(art_dir.name.startswith("m-"))

    def test_duplicate_candidates_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = ensure_vibe_run_layout(tmp)
            base = dict(task_id="ds/inst", model="m", kind="patch")
            a = AgentArtifact(patch="--- a\n+++ b\n@@\n+A\n", **base)
            b = AgentArtifact(patch="--- a\n+++ b\n@@\n+B\n", **base)
            refs_a = write_agent_artifact(run, a)
            refs_b = write_agent_artifact(run, b)
            # Distinct candidates land in distinct dirs (no overwrite); both
            # share the same task-dir parent.
            self.assertNotEqual(refs_a.artifact_dir, refs_b.artifact_dir)
            self.assertEqual(
                Path(refs_a.artifact_dir).parent,
                Path(refs_b.artifact_dir).parent,
            )
            self.assertEqual(
                Path(refs_a.patch_path).read_text(encoding="utf-8"),
                "--- a\n+++ b\n@@\n+A\n",
            )

    def test_rejects_file_path_escaping_run_dir(self) -> None:
        # Prediction artifacts are external input: ``../`` traversal and
        # absolute keys must be rejected before any write.
        for bad in ("../escape.txt", "../../etc/passwd", "/abs/evil.txt"):
            with tempfile.TemporaryDirectory() as tmp:
                run = ensure_vibe_run_layout(tmp)
                art = AgentArtifact(
                    task_id="ds/inst",
                    model="m",
                    kind="full_file",
                    files={bad: "x"},
                )
                with self.assertRaises(ValueError):
                    write_agent_artifact(run, art)
                # Nothing was written outside the run dir.
                self.assertFalse((Path(tmp).parent / "escape.txt").exists())
                self.assertFalse(Path("/abs/evil.txt").exists())


class ArtifactShaTest(unittest.TestCase):
    """compute_artifact_sha256 is stable and metadata-independent."""

    def _patch_artifact(self, **kw) -> AgentArtifact:
        base = dict(task_id="t", model="m", kind="patch", patch="diff")
        base.update(kw)
        return AgentArtifact(**base)

    def test_sha_stable_across_construction(self) -> None:
        a = self._patch_artifact()
        b = self._patch_artifact()
        self.assertEqual(
            compute_artifact_sha256(a), compute_artifact_sha256(b)
        )

    def test_sha_independent_of_metadata(self) -> None:
        a = self._patch_artifact(metadata={"x": 1})
        b = self._patch_artifact(metadata={"x": 2, "y": "z"})
        self.assertEqual(
            compute_artifact_sha256(a), compute_artifact_sha256(b)
        )

    def test_sha_independent_of_worktree(self) -> None:
        a = self._patch_artifact(worktree="/tmp/a")
        b = self._patch_artifact(worktree="/tmp/b")
        self.assertEqual(
            compute_artifact_sha256(a), compute_artifact_sha256(b)
        )

    def test_sha_changes_with_payload(self) -> None:
        a = self._patch_artifact(patch="diff-1")
        b = self._patch_artifact(patch="diff-2")
        self.assertNotEqual(
            compute_artifact_sha256(a), compute_artifact_sha256(b)
        )

    def test_files_sha_order_independent(self) -> None:
        a = AgentArtifact(
            task_id="t",
            model="m",
            kind="full_file",
            files={"a": "1", "b": "2"},
        )
        b = AgentArtifact(
            task_id="t",
            model="m",
            kind="full_file",
            files={"b": "2", "a": "1"},
        )
        self.assertEqual(
            compute_artifact_sha256(a), compute_artifact_sha256(b)
        )

    def test_task_sha_stable(self) -> None:
        t1 = VibeTask(id="x", source_dataset="d")
        t2 = VibeTask(id="x", source_dataset="d")
        self.assertEqual(compute_task_sha256(t1), compute_task_sha256(t2))


class ProvenanceTest(unittest.TestCase):
    """Provenance builder + git_commit + block projection."""

    def test_git_commit_none_on_non_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(git_commit(tmp))

    def test_build_provenance_redacts_env(self) -> None:
        prov = build_provenance(
            geh_version="0.1.0",
            adapter_name="mock",
            parser_version="1",
            oracle_capabilities=OracleCapabilities(
                runs_functional_tests=True
            ),
            resource_budget=ResourceBudget(max_workers=4),
            worker_count=4,
            env={"API_KEY": "secret", "FOO": "bar"},
            artifact_sha256="abc",
            task_sha256="def",
        )
        self.assertIsInstance(prov, Provenance)
        self.assertNotEqual(prov.redacted_env["API_KEY"], "secret")
        self.assertEqual(prov.redacted_env["FOO"], "bar")
        self.assertEqual(prov.resource_budget["max_workers"], 4)

    def test_build_provenance_no_repo_root_means_no_commit(self) -> None:
        prov = build_provenance(adapter_name="mock")
        self.assertIsNone(prov.geh_commit)

    def test_to_block_projection(self) -> None:
        prov = build_provenance(
            adapter_name="mock",
            parser_version="2",
            artifact_sha256="aa",
            task_sha256="bb",
        )
        block = prov.to_block()
        self.assertEqual(block.adapter_name, "mock")
        self.assertEqual(block.parser_version, "2")
        self.assertEqual(block.artifact_sha256, "aa")
        self.assertEqual(block.task_sha256, "bb")


if __name__ == "__main__":
    unittest.main()
