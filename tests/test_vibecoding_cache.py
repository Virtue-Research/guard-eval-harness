"""Tests for the vibecoding oracle-result cache + Materializer."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.cache import (
    CacheKey,
    OracleResultCache,
    _repo_root,
    resolve_cache_dir,
)
from guard_eval_harness.vibecoding.materialize import (
    MaterializeError,
    Materializer,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import RepoSpec, VibeTask


def _key(**overrides) -> CacheKey:
    base = dict(
        task_id="ds/inst-1",
        artifact_sha256="aaa",
        adapter_name="mock",
        adapter_version="1",
        upstream_ref="ref0",
        oracle_config_hash="cfg",
        oracle_capabilities_hash="caps",
        trial_index=0,
        random_seed=None,
        anti_cheat_policy_hash="acp",
    )
    base.update(overrides)
    return CacheKey(**base)


def _result() -> VibeTaskResult:
    return VibeTaskResult(
        task_id="ds/inst-1",
        source_dataset="ds",
        model="m",
        status="completed",
        functional_pass=True,
        security_oracle_pass=False,
    )


class CacheKeyTest(unittest.TestCase):
    """CacheKey.digest determinism + sensitivity."""

    def test_digest_deterministic(self) -> None:
        self.assertEqual(_key().digest(), _key().digest())

    def test_digest_changes_with_artifact_sha(self) -> None:
        self.assertNotEqual(
            _key(artifact_sha256="aaa").digest(),
            _key(artifact_sha256="bbb").digest(),
        )

    def test_digest_changes_with_trial_index(self) -> None:
        self.assertNotEqual(
            _key(trial_index=0).digest(),
            _key(trial_index=1).digest(),
        )

    def test_digest_changes_with_adapter_version(self) -> None:
        self.assertNotEqual(
            _key(adapter_version="1").digest(),
            _key(adapter_version="2").digest(),
        )

    def test_key_rejects_unknown_field(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            CacheKey(  # type: ignore[call-arg]
                task_id="t",
                artifact_sha256="a",
                adapter_name="x",
                adapter_version="1",
                oracle_config_hash="c",
                oracle_capabilities_hash="d",
                anti_cheat_policy_hash="e",
                bogus=True,
            )


class ResolveCacheDirTest(unittest.TestCase):
    """resolve_cache_dir precedence."""

    def setUp(self) -> None:
        self._saved = os.environ.pop("GEH_CACHE_DIR", None)

    def tearDown(self) -> None:
        os.environ.pop("GEH_CACHE_DIR", None)
        if self._saved is not None:
            os.environ["GEH_CACHE_DIR"] = self._saved

    def test_explicit_arg_wins_over_env(self) -> None:
        # An explicit --cache-dir must beat an exported GEH_CACHE_DIR so that
        # task loading (VibeRunner) and evaluation (EnvProvider) resolve the
        # same checkout; EnvProvider uses the same precedence.
        os.environ["GEH_CACHE_DIR"] = "/custom/cache"
        self.assertEqual(
            resolve_cache_dir("/arg/dir", repo_root="/repo"),
            Path("/arg/dir"),
        )

    def test_env_var_over_repo_default(self) -> None:
        os.environ["GEH_CACHE_DIR"] = "/custom/cache"
        self.assertEqual(
            resolve_cache_dir(repo_root="/repo"),
            Path("/custom/cache"),
        )

    def test_arg_over_repo(self) -> None:
        self.assertEqual(
            resolve_cache_dir("/arg/dir", repo_root="/repo"),
            Path("/arg/dir"),
        )

    def test_repo_default(self) -> None:
        self.assertEqual(
            resolve_cache_dir(repo_root="/repo"),
            Path("/repo/.geh"),
        )

    def test_default_walks_to_pyproject_root(self) -> None:
        # With no arg, env, or repo_root, the fallback walks up from the
        # installed module to the nearest dir containing pyproject.toml
        # (the checkout root for an editable install) instead of assuming a
        # fixed parents[N] source layout or the cwd.
        resolved = resolve_cache_dir()
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.name, ".geh")
        self.assertTrue((resolved.parent / "pyproject.toml").exists())

    def test_env_provider_and_oracles_share_the_resolver(self) -> None:
        # The canonical resolver is the single source of truth: EnvProvider
        # and the seccodebench oracle fallback must agree with it under every
        # precedence branch (here: the GEH_CACHE_DIR one). The bare-cache-dir
        # branch is checked against the canonical resolver directly so the
        # coverage holds in the OSS tree, where the internal-only mtsec oracle
        # (whose _default_cache just returns resolve_cache_dir) is not shipped.
        from guard_eval_harness.vibecoding import envs
        from guard_eval_harness.vibecoding.oracles.seccodebench import (
            _default_upstream,
        )

        self.assertIs(envs.resolve_cache_dir, resolve_cache_dir)
        os.environ["GEH_CACHE_DIR"] = "/custom/cache"
        self.assertEqual(resolve_cache_dir(), Path("/custom/cache"))
        self.assertEqual(
            _default_upstream(),
            Path("/custom/cache/upstreams/seccodebench"),
        )

    def test_repo_root_walk_finds_the_harness_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "checkout"
            module = repo / "src" / "guard_eval_harness" / "vibecoding"
            module.mkdir(parents=True)
            (repo / "pyproject.toml").write_text(
                '[project]\nname = "guard-eval-harness"\n', encoding="utf-8"
            )
            self.assertEqual(
                _repo_root(module / "cache.py"), repo.resolve()
            )

    def test_repo_root_walk_refuses_a_foreign_pyproject(self) -> None:
        # Wheel install into an application's local venv: the nearest
        # ancestor pyproject.toml belongs to the consuming app, not the
        # harness, and must not capture the cache root. The walk falls back
        # to the cwd (workspace-local), per the docstring.
        with tempfile.TemporaryDirectory() as td:
            app = Path(td) / "app"
            module = (
                app / ".venv" / "lib" / "python3.12" / "site-packages"
                / "guard_eval_harness" / "vibecoding"
            )
            module.mkdir(parents=True)
            (app / "pyproject.toml").write_text(
                '[project]\nname = "someapp"\n', encoding="utf-8"
            )
            self.assertEqual(_repo_root(module / "cache.py"), Path.cwd())

    def test_oracle_fallbacks_track_the_resolver(self) -> None:
        # No env override: the oracle defaults must equal whatever the
        # canonical resolver yields (no parallel parents[N] reimplementation).
        from guard_eval_harness.vibecoding.oracles.seccodebench import (
            _default_upstream,
        )

        self.assertEqual(
            _default_upstream(),
            resolve_cache_dir() / "upstreams" / "seccodebench",
        )


class OracleResultCacheTest(unittest.TestCase):
    """put/get round-trip + miss handling."""

    def test_put_get_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleResultCache(Path(tmp) / "cache" / "vibecoding")
            key = _key()
            cache.put(key, _result())
            got = cache.get(key)
            self.assertIsNotNone(got)
            self.assertEqual(got.task_id, "ds/inst-1")
            self.assertIs(got.functional_pass, True)
            self.assertIs(got.security_oracle_pass, False)

    def test_miss_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleResultCache(Path(tmp))
            self.assertIsNone(cache.get(_key()))

    def test_different_keys_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleResultCache(Path(tmp))
            cache.put(_key(trial_index=0), _result())
            self.assertIsNone(cache.get(_key(trial_index=1)))

    def test_corrupt_entry_is_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleResultCache(Path(tmp))
            key = _key()
            cache.base_dir.mkdir(parents=True, exist_ok=True)
            (cache.base_dir / f"{key.digest()}.json").write_text(
                "not json", encoding="utf-8"
            )
            self.assertIsNone(cache.get(key))


class MaterializerTest(unittest.TestCase):
    """Materializer.prepare + _hash_tree + _apply_patch."""

    def _materializer(self, root: Path) -> Materializer:
        return Materializer(
            cache_dir=root / "cache", run_dir=root / "run"
        )

    def _task(self) -> VibeTask:
        return VibeTask(
            id="ds/inst-1",
            source_dataset="ds",
            task_type="repo_dir",
            repo=RepoSpec(base_commit="abc123"),
        )

    def test_prepare_none_when_no_worktree_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            art = AgentArtifact(
                task_id="ds/inst-1", model="m", kind="patch", patch="p"
            )
            self.assertIsNone(
                mat.prepare(self._task(), art, need_worktree=False)
            )

    def test_apply_full_files_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            for bad in ("../escape.py", "/abs/evil.py"):
                art = AgentArtifact(
                    task_id="ds/inst-1", model="m", kind="full_file",
                    files={bad: "x"},
                )
                with self.assertRaises(ValueError):
                    mat._apply_full_files(worktree, art, [])
            self.assertFalse((Path(tmp) / "escape.py").exists())

    def test_hash_tree_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            tree = Path(tmp) / "tree"
            (tree / "pkg").mkdir(parents=True)
            (tree / "pkg" / "a.py").write_text("x = 1", encoding="utf-8")
            (tree / "b.py").write_text("y = 2", encoding="utf-8")
            h1 = mat._hash_tree(tree)
            h2 = mat._hash_tree(tree)
            self.assertEqual(h1, h2)

    def test_hash_tree_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            tree = Path(tmp) / "tree"
            tree.mkdir()
            target = tree / "a.py"
            target.write_text("x = 1", encoding="utf-8")
            before = mat._hash_tree(tree)
            target.write_text("x = 2", encoding="utf-8")
            self.assertNotEqual(before, mat._hash_tree(tree))

    def test_hash_tree_skips_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            tree = Path(tmp) / "tree"
            (tree / ".git").mkdir(parents=True)
            (tree / "a.py").write_text("x = 1", encoding="utf-8")
            before = mat._hash_tree(tree)
            (tree / ".git" / "HEAD").write_text("ref", encoding="utf-8")
            self.assertEqual(before, mat._hash_tree(tree))

    def test_apply_patch_applies_simple_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            target = worktree / "hello.txt"
            target.write_text("hello\n", encoding="utf-8")
            # Build a real unified diff with git so it is well-formed.
            subprocess.run(
                ["git", "init", "-q"], cwd=str(worktree), check=True
            )
            subprocess.run(
                ["git", "add", "hello.txt"], cwd=str(worktree), check=True
            )
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "-m", "init"],
                cwd=str(worktree),
                check=True,
            )
            target.write_text("hello world\n", encoding="utf-8")
            diff = subprocess.run(
                ["git", "diff"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            # Reset and apply via the Materializer helper.
            target.write_text("hello\n", encoding="utf-8")
            notes: list[str] = []
            mat._apply_patch(worktree, diff, notes)
            self.assertEqual(
                target.read_text(encoding="utf-8"), "hello world\n"
            )

    def test_apply_patch_empty_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            with self.assertRaises(MaterializeError):
                mat._apply_patch(worktree, "   ", [])

    def test_prepare_full_file_builds_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = self._materializer(Path(tmp))
            art = AgentArtifact(
                task_id="ds/inst-1",
                model="m",
                kind="full_file",
                files={"a.py": "x = 1"},
            )
            wt = mat.prepare(self._task(), art, need_worktree=True)
            self.assertIsNotNone(wt)
            self.assertEqual(wt.applied_kind, "full_file")
            self.assertTrue(
                (Path(wt.worktree_dir) / "a.py").exists()
            )
            self.assertTrue(wt.worktree_sha256)


if __name__ == "__main__":
    unittest.main()
