"""Symlink-confinement tests for the ``repo_dir`` overlay in the Materializer.

``_overlay_repo_dir`` may copy an untrusted generated repo tree into the eval
worktree, so it must never let a symlink (or a symlinked parent directory)
expose host files to the oracle: a link whose real target resolves outside the
source root is dropped, and an in-tree link is collapsed to a plain regular
file. The materialized worktree therefore carries no surviving symlinks.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from guard_eval_harness.vibecoding import materialize as materialize_mod
from guard_eval_harness.vibecoding.materialize import (
    MaterializeError,
    Materializer,
)
from guard_eval_harness.vibecoding.schema import RepoSpec, VibeTask


def _overlay(source: Path, worktree: Path) -> list[str]:
    """Run the overlay from ``source`` into ``worktree``; return the notes."""
    mat = Materializer(
        cache_dir=source.parent / "cache", run_dir=source.parent / "run"
    )
    worktree.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    mat._overlay_repo_dir(worktree, str(source), notes)
    return notes


def _git_repo_with_commit(path: Path) -> str:
    """Init a git repo at ``path`` with one commit; return the commit sha."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("x = 1\n")
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
    _git("add", "-A")
    _git("commit", "-qm", "init")
    return _git("rev-parse", "HEAD")


def test_restore_base_uses_checkout_matching_base_commit(
    tmp_path: Path,
) -> None:
    """A cached checkout whose git history contains the task's base_commit is
    handed to the agent as a sealed worktree (content visible, .git stripped)."""
    cache = tmp_path / "cache"
    sha = _git_repo_with_commit(cache / "upstreams" / "mock")
    task = VibeTask(
        id="mock/inst-1",
        source_dataset="mock",
        repo=RepoSpec(url="https://example.invalid/r", base_commit=sha),
    )
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    wt = mat.restore_base(task)
    assert wt is not None
    assert (wt / "app.py").read_text() == "x = 1\n"
    assert not (wt / ".git").exists()


def test_restore_base_uses_base_commit_not_current_tree(
    tmp_path: Path,
) -> None:
    """The sealed worktree must reflect the repo *at base_commit*, even when
    the cached checkout has moved past it (its env's pinned ref / the upstream
    fix) and carries dirty local edits -- never the checkout's live tree."""
    cache = tmp_path / "cache"
    repo = cache / "upstreams" / "mock"
    base_sha = _git_repo_with_commit(repo)  # app.py == "x = 1\n" at base
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True, capture_output=True, text=True, env=env,
        )

    # Advance HEAD past base_commit (the "upstream fix") and add a new file...
    (repo / "app.py").write_text("x = 999  # upstream fix\n")
    (repo / "fix_note.txt").write_text("patched\n")
    _git("add", "-A")
    _git("commit", "-qm", "upstream fix")
    # ...then dirty the working tree on top, so neither HEAD nor the working
    # tree match base_commit.
    (repo / "app.py").write_text("x = 12345  # uncommitted local edit\n")

    task = VibeTask(
        id="mock/inst-1",
        source_dataset="mock",
        repo=RepoSpec(url="https://example.invalid/r", base_commit=base_sha),
    )
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    wt = mat.restore_base(task)

    assert wt is not None
    # The base revision's content -- not the committed fix, not the dirty edit.
    assert (wt / "app.py").read_text() == "x = 1\n"
    # A file introduced only after base_commit must not leak into the worktree.
    assert not (wt / "fix_note.txt").exists()
    assert not (wt / ".git").exists()


def test_restore_base_accepts_caller_supplied_source_and_ref(
    tmp_path: Path,
) -> None:
    """An oracle that knows its scoring base injects ``(source, ref)``
    explicitly; restore_base seals exactly that revision -- ignoring
    ``task.repo.base_commit`` and any later commit / dirty edit in the source
    repo. This is the path SecureVibeBench uses to hand the agent the PVIC tree
    its patch is scored against, rather than the VIC carried by base_commit."""
    cache = tmp_path / "cache"
    repo = tmp_path / "clone"
    base_sha = _git_repo_with_commit(repo)  # app.py == "x = 1\n" at base
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True, capture_output=True, text=True, env=env,
        )

    # Advance HEAD past the requested ref and dirty the tree, so neither the
    # checkout's HEAD nor its working tree match the ref we inject.
    (repo / "app.py").write_text("x = 2  # newer commit\n")
    _git("commit", "-aqm", "newer")
    (repo / "app.py").write_text("x = 3  # uncommitted edit\n")

    task = VibeTask(
        id="any/inst-1",
        source_dataset="any",
        # A bogus base_commit that must be IGNORED when source/ref are given.
        repo=RepoSpec(url="https://example.invalid/r", base_commit="f" * 40),
    )
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    wt = mat.restore_base(task, source=repo, ref=base_sha)

    assert wt is not None
    # The injected ref's content -- not HEAD, not the dirty edit.
    assert (wt / "app.py").read_text() == "x = 1\n"
    assert not (wt / ".git").exists()


def test_restore_base_rejects_harness_checkout(tmp_path: Path) -> None:
    """A checkout whose history does NOT contain the task base_commit (the
    benchmark harness/evaluator tree) yields None -- never a leaked tree."""
    cache = tmp_path / "cache"
    # A real repo, but the task points at a different (absent) base commit,
    # exactly as a SusVibes harness checkout relates to a per-instance repo.
    _git_repo_with_commit(cache / "upstreams" / "mock")
    task = VibeTask(
        id="mock/inst-1",
        source_dataset="mock",
        repo=RepoSpec(base_commit="0" * 40),
    )
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    assert mat.restore_base(task) is None


def test_restore_base_none_without_base_commit(tmp_path: Path) -> None:
    """No base_commit -> None: correspondence to the task repo cannot be proven."""
    cache = tmp_path / "cache"
    _git_repo_with_commit(cache / "upstreams" / "mock")
    task = VibeTask(id="mock/inst-1", source_dataset="mock")
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    assert mat.restore_base(task) is None


def test_overlay_copies_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "a.txt").write_text("A")
    (source / "pkg" / "b.txt").write_text("B")
    _overlay(source, tmp_path / "wt")
    assert (tmp_path / "wt" / "a.txt").read_text() == "A"
    assert (tmp_path / "wt" / "pkg" / "b.txt").read_text() == "B"


def test_overlay_drops_file_symlink_escaping_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("HOST SECRET")
    source = tmp_path / "src"
    source.mkdir()
    (source / "normal.txt").write_text("ok")
    # A file symlink pointing at a host file outside the source tree.
    os.symlink(outside / "secret.txt", source / "leak.txt")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    assert (worktree / "normal.txt").read_text() == "ok"
    # The escaping link is neither preserved nor dereferenced into the worktree.
    assert not (worktree / "leak.txt").exists()
    assert not (worktree / "leak.txt").is_symlink()


def test_overlay_does_not_descend_symlinked_dir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("HOST SECRET")
    source = tmp_path / "src"
    source.mkdir()
    (source / "keep.txt").write_text("keep")
    # A directory symlink pointing outside the source tree.
    os.symlink(outside, source / "linkdir")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    assert (worktree / "keep.txt").read_text() == "keep"
    # No host file is reachable through the symlinked directory.
    assert not (worktree / "linkdir").exists()
    assert not (worktree / "linkdir" / "secret.txt").exists()


def test_overlay_collapses_in_tree_symlink_to_regular_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "real.txt").write_text("REAL")
    # An in-tree link is allowed but materialized as a plain regular file.
    os.symlink(source / "real.txt", source / "alias.txt")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    alias = worktree / "alias.txt"
    assert alias.read_text() == "REAL"
    assert not alias.is_symlink()


def test_overlay_skips_broken_symlink(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "real.txt").write_text("REAL")
    os.symlink(source / "missing.txt", source / "broken.txt")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    assert (worktree / "real.txt").read_text() == "REAL"
    assert not (worktree / "broken.txt").exists()


def test_overlay_skips_git_and_cache_dirs(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / ".git").mkdir(parents=True)
    (source / ".git" / "config").write_text("x")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "m.pyc").write_text("y")
    (source / "keep.py").write_text("z")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    assert (worktree / "keep.py").read_text() == "z"
    assert not (worktree / ".git").exists()
    assert not (worktree / "__pycache__").exists()


def test_overlay_worktree_has_no_surviving_symlinks(tmp_path: Path) -> None:
    """A tree mixing safe and unsafe links yields a link-free worktree."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET")
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "real.txt").write_text("R")
    os.symlink(source / "sub" / "real.txt", source / "sub" / "alias.txt")
    os.symlink(outside / "secret.txt", source / "escape.txt")
    os.symlink(outside, source / "escdir")

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    links = [p for p in worktree.rglob("*") if p.is_symlink()]
    assert links == []
    assert not (worktree / "escape.txt").exists()
    assert not (worktree / "escdir").exists()
    assert (worktree / "sub" / "real.txt").read_text() == "R"
    assert (worktree / "sub" / "alias.txt").read_text() == "R"


def test_overlay_reproduces_empty_directories(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "empty").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "data" / "f.txt").write_text("x")
    _overlay(source, tmp_path / "wt")
    assert (tmp_path / "wt" / "empty").is_dir()
    assert (tmp_path / "wt" / "data" / "f.txt").read_text() == "x"


def test_overlay_drops_relative_symlink_resolving_outside(
    tmp_path: Path,
) -> None:
    """A *relative* link that resolves out of the source root is dropped too."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("HOST SECRET")
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "ok.txt").write_text("ok")
    # ../../outside/secret.txt from inside source/sub -> escapes the root.
    os.symlink(
        Path("..") / ".." / "outside" / "secret.txt",
        source / "sub" / "rel_leak.txt",
    )

    worktree = tmp_path / "wt"
    _overlay(source, worktree)

    assert (worktree / "sub" / "ok.txt").read_text() == "ok"
    assert not (worktree / "sub" / "rel_leak.txt").exists()
    assert not (worktree / "sub" / "rel_leak.txt").is_symlink()


def _raise_timeout(*args, **kwargs):
    """Stand-in for ``subprocess.run`` that always times out."""
    cmd = args[0] if args else kwargs.get("args", ["x"])
    raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))


def test_overlay_commit_tree_timeout_raises_materialize_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A wedged ``git archive`` becomes a ``MaterializeError``, never a raw
    ``TimeoutExpired`` leaking out of the materializer to halt the batch."""
    repo = tmp_path / "repo"
    sha = _git_repo_with_commit(repo)  # real subprocess BEFORE patching
    mat = Materializer(cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    task = VibeTask(
        id="x/1",
        source_dataset="x",
        repo=RepoSpec(url="https://example.invalid/r", base_commit=sha),
    )
    monkeypatch.setattr(materialize_mod.subprocess, "run", _raise_timeout)
    with pytest.raises(MaterializeError) as excinfo:
        mat.restore_base(task, source=repo, ref=sha)
    assert "timed out" in str(excinfo.value)


def test_apply_patch_timeout_raises_materialize_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A wedged ``git apply`` / ``patch`` becomes a ``MaterializeError``."""
    mat = Materializer(cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    patch = "--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+y\n"
    monkeypatch.setattr(materialize_mod.subprocess, "run", _raise_timeout)
    with pytest.raises(MaterializeError) as excinfo:
        mat._apply_patch(worktree, patch, [])
    assert "timed out" in str(excinfo.value)
    # The temporary patch file is cleaned up even on the timeout path.
    assert not (worktree / ".geh-agent.patch").exists()


def test_probe_timeout_degrades_to_none(
    tmp_path: Path, monkeypatch
) -> None:
    """A hung ``git cat-file`` membership probe cannot confirm the checkout is
    the task repo, so ``restore_base`` degrades to None (in-container path)
    instead of aborting the run."""
    cache = tmp_path / "cache"
    sha = _git_repo_with_commit(cache / "upstreams" / "mock")
    task = VibeTask(
        id="mock/inst-1",
        source_dataset="mock",
        repo=RepoSpec(base_commit=sha),
    )
    mat = Materializer(cache_dir=cache, run_dir=tmp_path / "run")
    monkeypatch.setattr(materialize_mod.subprocess, "run", _raise_timeout)
    assert mat.restore_base(task) is None


def test_hash_tree_includes_regular_file_named_dot_git(
    tmp_path: Path,
) -> None:
    """A regular FILE named ``.git`` (submodule pointer) changes the hash.

    Only directory components are skip-matched: two candidates differing
    solely by a top-level ``.git`` file must not collide on the cache key.
    """
    mat = Materializer(cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "main.py").write_text("x = 1\n")
    without_pointer = mat._hash_tree(worktree)
    (worktree / ".git").write_text("gitdir: ../.git/modules/sub\n")
    with_pointer = mat._hash_tree(worktree)
    assert with_pointer != without_pointer


def test_hash_tree_still_skips_git_directory_contents(
    tmp_path: Path,
) -> None:
    """Files inside a ``.git`` DIRECTORY stay out of the hash."""
    mat = Materializer(cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    worktree = tmp_path / "wt"
    (worktree / ".git").mkdir(parents=True)
    (worktree / "main.py").write_text("x = 1\n")
    (worktree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    first = mat._hash_tree(worktree)
    (worktree / ".git" / "HEAD").write_text("ref: refs/heads/other\n")
    second = mat._hash_tree(worktree)
    assert first == second
