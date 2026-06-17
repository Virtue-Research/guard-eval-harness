"""Seal git history + local solution artifacts in a generation worktree.

Before a live agent runs, the sandbox must make the known upstream fix
unreachable. That means two things:

1. Remove (or, on failure, neutralize) the ``.git`` directory so the agent
   cannot ``git log``/``git show`` its way to the upstream solution commit, or
   diff against an unfixed parent to recover the patch.
2. Remove obvious local solution artifacts that benchmarks sometimes leave in
   the checkout -- ``golden_patch``/``security_patch``/``gold.patch`` files,
   ``*.solution`` files, and similar -- so the answer is not sitting on disk.

This is a *generation-side* concern only; oracle evaluation runs against a
fresh materialized worktree and is unaffected. The function records exactly
what it sealed so the result/provenance can attest to it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import Field

from guard_eval_harness.vibecoding.schema import VibeModel

# Directory names that hold VCS history we must seal.
_GIT_DIR_NAMES = (".git",)

# Filename stems (case-insensitive, without extension) that name a known fix.
_SOLUTION_STEMS = (
    "golden_patch",
    "gold_patch",
    "golden",
    "gold",
    "security_patch",
    "fix_patch",
    "reference_patch",
    "solution",
    "answer",
    "expected_patch",
)

# Filename suffixes that mark a file as solution material regardless of stem.
_SOLUTION_SUFFIXES = (
    ".solution",
    ".gold",
    ".golden",
)

# Hard cap so a pathological tree can never make sealing walk forever.
_MAX_SCAN_FILES = 200_000


class GitSealResult(VibeModel):
    """Summary of what :func:`seal_git_history` removed or neutralized."""

    worktree_path: str
    git_sealed: bool = False
    git_seal_method: str | None = None
    removed_git_dirs: list[str] = Field(default_factory=list)
    removed_solution_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def anything_sealed(self) -> bool:
        """True when any git history or solution file was sealed."""
        return bool(
            self.git_sealed
            or self.removed_git_dirs
            or self.removed_solution_files
        )


def _looks_like_solution(path: Path) -> bool:
    """Heuristic: does ``path``'s name name a known-fix artifact?

    Matches on a small allowlist of solution stems (the file stem must equal
    or start with the stem, e.g. ``golden_patch.diff``) plus a few solution
    suffixes (``foo.solution``). Conservative on purpose: it should never trip
    on ordinary source files.
    """
    name = path.name.lower()
    stem = path.stem.lower()
    for suffix in _SOLUTION_SUFFIXES:
        if name.endswith(suffix):
            return True
    for marker in _SOLUTION_STEMS:
        if stem == marker or stem.startswith(marker + "."):
            return True
        if stem.startswith(marker + "_") or stem.startswith(marker + "-"):
            return True
    return False


def _seal_git_dir(git_path: Path, result: GitSealResult) -> None:
    """Remove a ``.git`` dir; on failure, neutralize it in place.

    The primary path is a full ``rmtree``. If that fails (permission, busy
    handle), we fall back to truncating ``HEAD`` and renaming the directory
    so history is unreachable even though some bytes remain on disk.
    """
    rel = str(git_path)
    try:
        if git_path.is_dir():
            shutil.rmtree(git_path)
        else:
            git_path.unlink()
        result.removed_git_dirs.append(rel)
        result.git_sealed = True
        result.git_seal_method = "removed"
        return
    except OSError as exc:
        result.notes.append(f"rmtree failed for {rel}: {exc}")

    # Fallback: neutralize. Renaming away from ``.git`` is enough for git to
    # stop recognizing it as a repository.
    try:
        head = git_path / "HEAD"
        if head.is_file():
            head.write_text("", encoding="utf-8")
        sealed = git_path.with_name(".git.sealed")
        if sealed.exists():
            shutil.rmtree(sealed, ignore_errors=True)
        git_path.rename(sealed)
        result.removed_git_dirs.append(rel)
        result.git_sealed = True
        result.git_seal_method = "neutralized"
    except OSError as exc:
        result.notes.append(f"neutralize failed for {rel}: {exc}")


def seal_git_history(worktree_path: str | Path) -> GitSealResult:
    """Seal ``.git`` history + local solution artifacts under a worktree.

    Walks ``worktree_path`` once and:

    - removes every ``.git`` directory (or neutralizes it if removal fails);
    - removes files whose names look like a known-fix artifact (golden /
      security patches, ``*.solution`` files, etc).

    Returns a :class:`GitSealResult` recording exactly what was sealed. Safe
    to call on a tree with no ``.git`` and no solution files (returns a result
    with ``anything_sealed`` False).
    """
    root = Path(worktree_path)
    result = GitSealResult(worktree_path=str(root))
    if not root.is_dir():
        result.notes.append(f"worktree is not a directory: {root}")
        return result

    # Seal git dirs first (top-down), then scan remaining files. Collect git
    # dirs before mutating so we don't fight the live walk.
    git_dirs: list[Path] = []
    solution_files: list[Path] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        scanned += 1
        if scanned > _MAX_SCAN_FILES:
            result.notes.append(
                f"scan truncated at {_MAX_SCAN_FILES} entries"
            )
            break
        # Skip anything already inside a git dir we will remove wholesale.
        if any(part in _GIT_DIR_NAMES for part in path.relative_to(root).parts):
            if path.name in _GIT_DIR_NAMES and path.is_dir():
                git_dirs.append(path)
            continue
        if path.is_file() and _looks_like_solution(path):
            solution_files.append(path)

    # Catch a top-level ``.git`` (rglob includes it, handled above) and any
    # bare-repo ``.git`` file pointer.
    for name in _GIT_DIR_NAMES:
        candidate = root / name
        if candidate.exists() and candidate not in git_dirs:
            git_dirs.append(candidate)

    for git_path in git_dirs:
        _seal_git_dir(git_path, result)

    for sol in solution_files:
        try:
            sol.unlink()
            result.removed_solution_files.append(str(sol))
        except OSError as exc:
            result.notes.append(f"could not remove {sol}: {exc}")

    if not result.anything_sealed:
        result.notes.append("nothing to seal")
    return result
