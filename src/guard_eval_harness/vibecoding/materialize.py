"""Materializer: build a worktree from a task + artifact for repo_dir oracles.

Shared by both ``live_agent`` and ``patch_eval`` modes. Given a task (which
names a repo + base commit) and an :class:`AgentArtifact`, it restores the
repo, applies the candidate (patch / full files / repo_dir overlay), and
records deterministic tree hashes for the base and final worktrees.

Adapters that score patches in their own batch executor (e.g. SusVibes) stay
worktree-free; ``prepare`` returns ``None`` when ``need_worktree`` is False.
The hashing is a deterministic git-like tree hash over the worktree so a
caller can fingerprint the materialized state without invoking git.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import Field

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.run_store import safe_task_id
from guard_eval_harness.vibecoding.safe_path import safe_relpath
from guard_eval_harness.vibecoding.schema import VibeModel, VibeTask

# Names skipped when overlaying/hashing a worktree (VCS + caches).
_TREE_HASH_SKIP_DIRS = {".git", "__pycache__"}

# Wall-clock caps for the materializer's subprocesses. ``geh vibe run`` runs
# outside CI (no Actions ``timeout-minutes``), and these children run
# sequentially per task, so a single wedged child -- a corrupt pack object
# hanging ``git archive``, a ``tar`` blocked on a full pipe -- would otherwise
# stall the whole batch with no escape hatch. A timeout becomes a loud
# ``MaterializeError`` instead.
_PROBE_TIMEOUT_S = 60      # `git cat-file -e` commit-membership probe
_ARCHIVE_TIMEOUT_S = 300   # `git archive` + `tar -x` base-tree extraction
_PATCH_TIMEOUT_S = 60      # `git apply` / `patch` candidate application


def safe_overlay_tree(
    source: Path, dest_root: Path, notes: list[str]
) -> int:
    """Copy ``source``'s contents into ``dest_root``, confined and symlink-free.

    Shared by the materializer and any oracle that stages a candidate-supplied
    repo tree (e.g. A.S.E). The source may be an untrusted generated repo, so
    no symlink is ever reproduced into ``dest_root`` (a surviving link could
    point at host files and expose them to the oracle):

    - ``os.walk`` runs with ``followlinks=False`` so a symlinked directory is
      never descended into (which would otherwise reach outside the tree, or
      loop); such directories are also pruned from the walk.
    - VCS history / cache dirs (``.git``/``__pycache__``) are skipped.
    - Each file is copied only if its real target resolves inside ``source``;
      a link (or symlinked parent) escaping it, or a broken link, is dropped
      with a note rather than reproduced.
    - In-tree links are dereferenced and written as plain regular files
      (``shutil.copy2`` defaults to ``follow_symlinks=True``), so ``dest_root``
      never carries a surviving symlink.

    Every written path is additionally confined to ``dest_root`` via
    :func:`safe_relpath`. Returns the number of files copied.
    """
    source_root = source.resolve()
    copied = 0
    skipped_links = 0
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        # Prune VCS/cache dirs and any symlinked subdirectory in place so the
        # walk neither hashes nor follows them. A symlinked directory is never
        # reproduced: if it aliases an in-tree directory, that target is walked
        # at its real path anyway.
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _TREE_HASH_SKIP_DIRS
            and not (current / name).is_symlink()
        ]
        rel_dir = current.relative_to(source)
        if rel_dir != Path("."):
            # Reproduce the (possibly empty) directory, confined to root.
            safe_relpath(dest_root, rel_dir).mkdir(parents=True, exist_ok=True)
        for name in sorted(filenames):
            item = current / name
            rel = rel_dir / name
            # A broken symlink has no target contents to materialize.
            if item.is_symlink() and not item.exists():
                skipped_links += 1
                notes.append(f"skipped broken symlink: {rel.as_posix()}")
                continue
            # The real target must resolve inside the source tree; a link (or a
            # symlinked parent) escaping it would expose host files.
            if not item.resolve().is_relative_to(source_root):
                skipped_links += 1
                notes.append(
                    f"skipped symlink escaping source root: {rel.as_posix()}"
                )
                continue
            # Confine each overlaid entry to the destination root.
            dest = safe_relpath(dest_root, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # follow_symlinks defaults to True, so an in-tree link is
            # dereferenced and written as a plain regular file.
            shutil.copy2(item, dest)
            copied += 1
    note = f"overlaid {copied} file(s) from {source}"
    if skipped_links:
        note += f" ({skipped_links} unsafe symlink(s) skipped)"
    notes.append(note)
    return copied


class MaterializedWorktree(VibeModel):
    """A worktree built for a single task + artifact."""

    task_id: str
    worktree_dir: str
    base_commit: str | None = None
    base_tree_sha256: str | None = None
    artifact_sha256: str | None = None
    worktree_sha256: str
    applied_kind: str
    notes: list[str] = Field(default_factory=list)


class MaterializeError(RuntimeError):
    """Raised when a worktree cannot be materialized from an artifact."""


def _run_or_timeout(
    cmd: list[str], *, timeout: float, what: str, **kwargs
) -> subprocess.CompletedProcess:
    """``subprocess.run`` with a wall-clock cap; timeout -> MaterializeError.

    The binary tar stream piped between ``git archive`` and ``tar`` rules out
    the text-mode :func:`vibecoding.subprocess.run_command` helper, so each
    call gets an explicit ``timeout=`` here and a :class:`subprocess.TimeoutExpired`
    is translated into a :class:`MaterializeError` naming ``what`` timed out --
    callers never see a raw ``TimeoutExpired`` leak out of the materializer.
    """
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise MaterializeError(
            f"{what} timed out after {timeout:g}s"
        ) from exc


class Materializer:
    """Builds materialized worktrees under ``run_dir/artifacts/<task>/``."""

    def __init__(
        self, cache_dir: str | Path, run_dir: str | Path
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.run_dir = Path(run_dir)

    def prepare(
        self,
        task: VibeTask,
        artifact: AgentArtifact,
        need_worktree: bool,
    ) -> MaterializedWorktree | None:
        """Materialize a worktree for ``task`` from ``artifact``.

        Returns ``None`` when ``need_worktree`` is False (the oracle scores
        the raw artifact and never needs a checkout). Otherwise restores the
        repo, applies the candidate, and returns a
        :class:`MaterializedWorktree` with deterministic tree hashes.
        """
        if not need_worktree:
            return None

        worktree = (
            self.run_dir
            / "artifacts"
            / safe_task_id(task.id)
            / "materialized-worktree"
        )
        notes: list[str] = []
        self._restore_repo(task, worktree, notes)
        base_tree = self._hash_tree(worktree)

        if artifact.kind == "patch":
            self._apply_patch(worktree, artifact.patch or "", notes)
        elif artifact.kind in ("full_file", "completion"):
            self._apply_full_files(worktree, artifact, notes)
        elif artifact.kind == "repo_dir":
            self._overlay_repo_dir(worktree, artifact.worktree or "", notes)
        else:
            raise MaterializeError(
                f"cannot materialize artifact kind {artifact.kind!r}"
            )

        final_tree = self._hash_tree(worktree)
        return MaterializedWorktree(
            task_id=task.id,
            worktree_dir=str(worktree),
            base_commit=task.repo.base_commit,
            base_tree_sha256=base_tree,
            artifact_sha256=artifact.metadata.get("artifact_sha256")
            if isinstance(artifact.metadata.get("artifact_sha256"), str)
            else None,
            worktree_sha256=final_tree,
            applied_kind=artifact.kind,
            notes=notes,
        )

    def restore_base(
        self,
        task: VibeTask,
        *,
        source: str | Path | None = None,
        ref: str | None = None,
    ) -> Path | None:
        """Lay down a sealed base checkout for live generation; None if there is
        no checkout that genuinely corresponds to the task repo.

        Two ways the (repo, revision) to seal is resolved:

        - **Caller-supplied** (``source`` + ``ref``): an oracle that knows the
          exact base its scoring uses passes a git checkout dir and the ref to
          extract. SecureVibeBench, for instance, applies candidate patches at
          PVIC (``VIC^``) inside the ARVO container, so its
          :meth:`SecureVibeBenchOracle.live_base` hands the host-side clone +
          the resolved PVIC here -- ``task.repo.base_commit`` (which carries
          VIC) is *not* the patch base and must not be used directly.
        - **Default** (neither given): a cached env checkout whose git history
          actually contains ``task.repo.base_commit`` (see
          :meth:`_task_repo_source`), extracted at ``base_commit``.

        Either way the committed tree *at* ``ref`` is extracted -- never a
        checkout's live working tree, which may sit at the env's pinned ref, a
        later commit, or carry dirty edits (any of which could already contain
        the upstream fix) -- then ``.git`` history + local solution artifacts
        are sealed so a live agent cannot read the upstream fix.

        The default path deliberately does **not** fall back to the dataset's
        upstream *benchmark/evaluator* checkout (``cache_dir/upstreams/
        <dataset>``): for the v0 datasets that tree is the harness + dataset
        metadata (e.g. SusVibes' ``datasets/default/susvibes_dataset.jsonl``),
        **not** the repo named by ``task.repo`` -- the repo under test lives
        inside per-instance Docker images. Handing the harness tree to a live
        agent would base generations on irrelevant/leaky files and invalidate
        the run. When no usable checkout exists, return ``None`` so the
        in-container driver extracts the repo from the eval image instead.
        """
        from guard_eval_harness.vibecoding.sandbox.git_seal import (
            seal_git_history,
        )

        if source is None:
            source = self._task_repo_source(task)
            ref = task.repo.base_commit
        if source is None or not ref:
            return None

        worktree = self.run_dir / "gen" / safe_task_id(task.id)
        if worktree.exists():
            shutil.rmtree(worktree)
        worktree.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []
        self._overlay_commit_tree(worktree, Path(source), ref, notes)
        has_content = any(
            p.is_file() and ".git" not in p.parts
            for p in worktree.rglob("*")
        )
        if not has_content:
            return None
        seal_git_history(worktree)
        return worktree

    def _task_repo_source(self, task: VibeTask) -> Path | None:
        """Cached checkout that genuinely is ``task.repo`` at ``base_commit``.

        A dataset's upstream tree (``cache_dir/upstreams/<dataset>``) is the
        benchmark/evaluator harness, not the repo under test, so it qualifies
        only when its git history actually contains ``task.repo.base_commit``;
        the harness tree will not. Returns ``None`` (so the in-container driver
        handles extraction) when there is no base commit, no checkout, or the
        checkout does not contain that commit.
        """
        base = task.repo.base_commit
        if not base:
            return None
        candidate = self.cache_dir / "upstreams" / task.source_dataset
        if not candidate.is_dir():
            return None
        # Only treat the checkout as the task repo when its history actually
        # contains the task's base commit; the benchmark harness tree will not.
        # A timeout here cannot confirm membership, so degrade to the
        # in-container extraction path (None) rather than wedge the batch.
        try:
            proc = subprocess.run(
                ["git", "-C", str(candidate), "cat-file", "-e",
                 f"{base}^{{commit}}"],
                capture_output=True,
                text=True,
                check=False,
                timeout=_PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return None
        return candidate if proc.returncode == 0 else None

    def _overlay_commit_tree(
        self, worktree: Path, source: Path, commit: str, notes: list[str]
    ) -> None:
        """Overlay ``source``'s committed tree *at ``commit``* into ``worktree``.

        The cached checkout's working tree is never read directly: it may sit
        at the env's pinned ref, at a commit past the task's ``base_commit``,
        or carry local edits -- any of which could already contain the
        upstream fix and would invalidate a live generation. ``git archive``
        serializes the tree recorded at ``commit`` (independent of the
        checkout's ``HEAD`` and of any dirty working-tree state) into a
        scratch dir, which is then overlaid through :meth:`_overlay_repo_dir`
        so the same symlink-confined copy + VCS/cache skipping apply and the
        sealed worktree is byte-identical to a clean ``base_commit`` checkout.

        Raises :class:`MaterializeError` if ``commit`` cannot be archived
        (e.g. its tree objects are absent), so the run fails loudly rather
        than seeding the agent with the wrong tree.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        scratch = Path(
            tempfile.mkdtemp(prefix="geh-base-", dir=str(self.run_dir))
        )
        try:
            archive = _run_or_timeout(
                ["git", "-C", str(source), "archive",
                 "--format=tar", commit],
                timeout=_ARCHIVE_TIMEOUT_S,
                what=f"git archive of {commit} in {source}",
                capture_output=True,
                check=False,
            )
            if archive.returncode != 0:
                raise MaterializeError(
                    f"git archive of {commit} in {source} failed: "
                    f"{archive.stderr.decode('utf-8', 'replace').strip()}"
                )
            extract = _run_or_timeout(
                ["tar", "-x", "-C", str(scratch)],
                timeout=_ARCHIVE_TIMEOUT_S,
                what=f"tar extract of base archive for {commit}",
                input=archive.stdout,
                capture_output=True,
                check=False,
            )
            if extract.returncode != 0:
                raise MaterializeError(
                    f"extracting base-commit archive for {commit} failed: "
                    f"{extract.stderr.decode('utf-8', 'replace').strip()}"
                )
            self._overlay_repo_dir(worktree, str(scratch), notes)
            notes.append(f"restored from base commit {commit}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _restore_repo(
        self, task: VibeTask, worktree: Path, notes: list[str]
    ) -> None:
        """Lay down the starting repo state at ``worktree``.

        Prefers a previously cached checkout under
        ``cache_dir/upstreams/<dataset>``; if none exists, creates an empty
        directory (network clone is out of scope for this layer and is the
        caller's responsibility). The directory is always recreated clean.
        """
        if worktree.exists():
            shutil.rmtree(worktree)
        worktree.mkdir(parents=True, exist_ok=True)

        cached = (
            self.cache_dir / "upstreams" / task.source_dataset
        )
        if cached.is_dir():
            self._overlay_repo_dir(worktree, str(cached), notes)
            notes.append(f"restored from cache: {cached}")
        else:
            notes.append("no cached checkout; starting from empty worktree")

    def _apply_patch(
        self, worktree: Path, patch_text: str, notes: list[str]
    ) -> None:
        """Apply a unified diff to ``worktree`` via ``git apply``/``patch``.

        Tries ``git apply`` first (handles new files / renames), then falls
        back to GNU ``patch``. Raises :class:`MaterializeError` on failure.
        """
        if not patch_text.strip():
            raise MaterializeError("empty patch cannot be applied")
        patch_file = worktree / ".geh-agent.patch"
        patch_file.write_text(patch_text, encoding="utf-8")
        try:
            attempts = (
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                ["patch", "-p1", "-i", str(patch_file)],
            )
            last_err = ""
            for cmd in attempts:
                completed = _run_or_timeout(
                    cmd,
                    timeout=_PATCH_TIMEOUT_S,
                    what=f"{cmd[0]} patch apply",
                    cwd=str(worktree),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode == 0:
                    notes.append(f"applied patch via {cmd[0]}")
                    return
                last_err = completed.stderr.strip() or completed.stdout.strip()
            raise MaterializeError(
                f"patch apply failed: {last_err}"
            )
        finally:
            if patch_file.exists():
                patch_file.unlink()

    def _apply_full_files(
        self,
        worktree: Path,
        artifact: AgentArtifact,
        notes: list[str],
    ) -> None:
        """Write ``files`` (and a ``completion`` if present) into worktree."""
        wrote = 0
        for rel_path, content in artifact.files.items():
            # Confine BYO file keys to the worktree (no ``..``/abs/symlink).
            target = safe_relpath(worktree, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            wrote += 1
        if artifact.completion is not None:
            target = worktree / "completion.txt"
            target.write_text(artifact.completion, encoding="utf-8")
            wrote += 1
        notes.append(f"wrote {wrote} full file(s)")

    def _overlay_repo_dir(
        self, worktree: Path, source_dir: str, notes: list[str]
    ) -> None:
        """Copy a source repo directory tree into ``worktree``.

        Delegates to :func:`safe_overlay_tree`, which skips VCS/cache dirs and
        never lets a symlink survive into the materialized worktree.
        """
        source = Path(source_dir)
        if not source.is_dir():
            raise MaterializeError(
                f"repo_dir source is not a directory: {source_dir}"
            )
        safe_overlay_tree(source, worktree, notes)

    def _hash_tree(self, worktree: str | Path) -> str:
        """Deterministic git-like tree hash over ``worktree``.

        Walks all regular files in sorted relative-path order (skipping
        ``.git``/cache dirs), hashing ``"<relpath>\\0<blob_sha>\\n"`` for each.
        Independent of OS walk order, so identical trees hash identically.
        """
        root = Path(worktree)
        digest = hashlib.sha256()
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            # Match parent-directory parts only: a regular FILE named
            # ``.git`` (e.g. a submodule pointer) must still be hashed.
            and not (
                set(p.relative_to(root).parent.parts) & _TREE_HASH_SKIP_DIRS
            )
        )
        for path in files:
            rel = path.relative_to(root).as_posix()
            blob = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(blob.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()
