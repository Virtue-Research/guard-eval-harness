"""Anti-cheat for live-agent generation.

Anti-cheat is a generation-side core concern. It seals access to known fixes
(via :mod:`git_seal`), controls egress (via :mod:`network`), and watches the
agent run for signs that it reached prohibited information -- a copied golden
patch, a reference to hidden solution material, or an attempt to reach a
denied host.

This module provides the post-generation **detectors** and the policy
identity used for provenance / cache keys. The detectors are deliberately
conservative file/content heuristics (no agent runtime hooks required): they
scan the workspace before and after the run, diff the file set, and flag any
new file whose content looks like an upstream solution patch or that
references hidden solution material.

v0 STATUS: the detectors are implemented and tested but NOT yet invoked by
the runner (enforcement is deferred Stage E work; see the runner's
``_DEFAULT_ANTI_CHEAT_POLICY`` note). Until the runner wires them into live
generation, no code path emits ``status=cheating_detected`` or sets
``anti_cheat_enforced=True`` -- so leaderboards cannot yet require
enforcement. When wired, a flagged row counts as a *model failure*, never a
silent drop (the caller maps ``cheating_flagged`` onto
``status=cheating_detected``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from guard_eval_harness.execution.artifacts import sha256_payload
from guard_eval_harness.vibecoding.schema import VibeModel

# Default policy identity. This is the SINGLE source of truth for the
# anti-cheat policy token: the runner imports it for its cache-key /
# provenance hash, so the two can never drift. The v0 value is the runner's
# long-standing placeholder ("none", detectors not yet enforced); changing it
# invalidates every cached oracle verdict, so bump it only when enforcement
# semantics actually change.
DEFAULT_POLICY_ID = "none"

# Substrings that, when found in a *new* workspace file, indicate the agent
# copied or referenced hidden solution material. Matched case-insensitively.
_SOLUTION_MARKERS = (
    "golden_patch",
    "golden patch",
    "gold_patch",
    "security_patch",
    "reference_patch",
    "reference solution",
    "official fix",
    "upstream fix commit",
    "solution.patch",
    "hidden solution",
    "BEGIN GOLDEN PATCH",
)

# Diff/patch shape markers; a new file that is itself a unified diff is
# suspicious when paired with a solution marker.
_DIFF_MARKERS = ("diff --git ", "--- a/", "+++ b/")

# Files we never treat as content to scan (binaries / caches / our own seal).
_SKIP_DIR_PARTS = {".git", ".git.sealed", "__pycache__"}

# Cap per-file read so a huge artifact can't blow memory.
_MAX_READ_BYTES = 2_000_000


class WorkspaceSnapshot(VibeModel):
    """A before/after fingerprint of a generation workspace.

    Maps each relative file path to its content sha256 so :class:`AntiCheat`
    can identify newly created and modified files after the agent runs.
    """

    root: str
    files: dict[str, str] = Field(default_factory=dict)


class AntiCheatResult(VibeModel):
    """Outcome of the post-generation anti-cheat scan."""

    cheating_flagged: bool = False
    anti_cheat_enforced: bool = False
    findings: list[str] = Field(default_factory=list)
    policy_id: str = DEFAULT_POLICY_ID
    policy_hash: str = ""
    new_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)


def _iter_workspace_files(root: Path):
    """Yield regular files under ``root`` skipping VCS/cache dirs."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Match parent-directory parts only: a regular FILE named ``.git``
        # (e.g. a submodule pointer) must still be snapshotted so anti-cheat
        # sees an agent creating or modifying it.
        parts = set(path.relative_to(root).parent.parts)
        if parts & _SKIP_DIR_PARTS:
            continue
        yield path


def _file_sha256(path: Path) -> str:
    """SHA256 of a file's bytes (best effort; empty string on error)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_text(path: Path) -> str:
    """Read a capped, decoded view of a file (empty on binary/error)."""
    try:
        data = path.read_bytes()[:_MAX_READ_BYTES]
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")


def _scan_content_for_solution(text: str) -> list[str]:
    """Return finding ids for solution markers / copied patches in ``text``."""
    findings: list[str] = []
    lowered = text.lower()
    matched_markers = [
        marker for marker in _SOLUTION_MARKERS if marker.lower() in lowered
    ]
    for marker in matched_markers:
        findings.append(f"solution_marker:{marker}")
    looks_like_diff = any(m in text for m in _DIFF_MARKERS)
    if looks_like_diff and matched_markers:
        findings.append("copied_golden_patch")
    return findings


class AntiCheat:
    """Snapshot-and-scan anti-cheat for a single generation workspace.

    Usage::

        ac = AntiCheat(policy_id="anti_cheat:v0", enforced=True)
        before = ac.scan_workspace_before(worktree)
        ...  # agent runs
        result = ac.scan_workspace_after(worktree, before)

    ``enforced`` records whether sandbox policy + monitoring were active for
    this trial; it flows straight into ``AntiCheatResult.anti_cheat_enforced``.
    """

    def __init__(
        self,
        policy_id: str = DEFAULT_POLICY_ID,
        enforced: bool = True,
        extra_markers: list[str] | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.enforced = enforced
        self._extra_markers = tuple(extra_markers or ())

    # --- snapshots ----------------------------------------------------

    def scan_workspace_before(
        self, worktree_path: str | Path
    ) -> WorkspaceSnapshot:
        """Fingerprint the workspace before the agent runs."""
        return self._snapshot(worktree_path)

    def scan_workspace_after(
        self,
        worktree_path: str | Path,
        before: WorkspaceSnapshot,
    ) -> AntiCheatResult:
        """Diff against ``before`` and scan new/modified files for cheating."""
        after = self._snapshot(worktree_path)
        root = Path(worktree_path)

        new_files = sorted(set(after.files) - set(before.files))
        modified_files = sorted(
            rel
            for rel, digest in after.files.items()
            if rel in before.files and before.files[rel] != digest
        )

        findings: list[str] = []
        for rel in new_files + modified_files:
            text = _read_text(root / rel)
            for fid in self._scan_text(text):
                findings.append(f"{fid}@{rel}")

        findings = sorted(set(findings))
        return AntiCheatResult(
            cheating_flagged=bool(findings),
            anti_cheat_enforced=self.enforced,
            findings=findings,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash(),
            new_files=new_files,
            modified_files=modified_files,
        )

    def scan_text(self, text: str) -> AntiCheatResult:
        """Standalone content check (e.g. over an agent's emitted patch).

        Useful when there is no before/after workspace -- the caller hands the
        candidate text directly and we flag copied-patch / solution markers.
        """
        findings = sorted(set(self._scan_text(text)))
        return AntiCheatResult(
            cheating_flagged=bool(findings),
            anti_cheat_enforced=self.enforced,
            findings=findings,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash(),
        )

    # --- policy identity ----------------------------------------------

    def policy_descriptor(self) -> dict[str, object]:
        """Canonical dict describing this anti-cheat policy."""
        return {
            "anti_cheat_policy_id": self.policy_id,
            "enforced": self.enforced,
            "markers": sorted(set(_SOLUTION_MARKERS) | set(self._extra_markers)),
        }

    def policy_hash(self) -> str:
        """Content hash of the policy for cache/provenance keys."""
        return sha256_payload(self.policy_descriptor())

    # --- internals ----------------------------------------------------

    def _snapshot(self, worktree_path: str | Path) -> WorkspaceSnapshot:
        root = Path(worktree_path)
        files: dict[str, str] = {}
        if root.is_dir():
            for path in _iter_workspace_files(root):
                rel = path.relative_to(root).as_posix()
                files[rel] = _file_sha256(path)
        return WorkspaceSnapshot(root=str(root), files=files)

    def _scan_text(self, text: str) -> list[str]:
        findings = _scan_content_for_solution(text)
        lowered = text.lower()
        for marker in self._extra_markers:
            if marker.lower() in lowered:
                findings.append(f"solution_marker:{marker}")
        return findings


def anti_cheat_policy_id(enforced: bool = True) -> str:
    """The default anti-cheat policy id (stable token for provenance)."""
    return DEFAULT_POLICY_ID if enforced else "anti_cheat:disabled"


def anti_cheat_policy_hash(
    policy_id: str = DEFAULT_POLICY_ID,
    enforced: bool = True,
) -> str:
    """Hash a policy id + enforcement flag for cache keys.

    Mirrors the runner's convention of hashing an
    ``{"anti_cheat_policy_id": ...}`` payload so a default-policy hash is
    consistent across the pipeline; the enforcement flag is folded in so an
    enforced vs unenforced run keys distinctly.
    """
    return sha256_payload(
        {"anti_cheat_policy_id": policy_id, "enforced": enforced}
    )
