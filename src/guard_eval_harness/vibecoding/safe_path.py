"""Confine externally-supplied file keys to an intended directory.

Prediction artifacts (BYO ``full_file`` payloads) and generated repository
trees come from outside GEH, so any relative path they carry must be kept
inside the directory we mean to write under. :func:`safe_relpath` rejects
absolute paths, ``..`` traversal, and any key that resolves outside ``root``
(including via a symlink), returning the validated absolute target.
"""

from __future__ import annotations

from pathlib import Path


def safe_relpath(root: str | Path, rel: str | Path) -> Path:
    """Return ``root/rel`` confined to ``root`` or raise ``ValueError``.

    Rejects an absolute ``rel``, ``..`` parent traversal, and any target that
    resolves outside ``root`` (e.g. through a symlink in an already
    materialized tree). The returned path is absolute and guaranteed to live
    under the resolved ``root``.
    """
    root_resolved = Path(root).resolve()
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError(f"absolute path not allowed: {str(rel)!r}")
    if any(part == ".." for part in rel_path.parts):
        raise ValueError(f"parent traversal not allowed: {str(rel)!r}")
    target = (root_resolved / rel_path).resolve()
    if target != root_resolved and not target.is_relative_to(root_resolved):
        raise ValueError(
            f"path escapes {root_resolved}: {str(rel)!r}"
        )
    return target


def assert_relpath_within(rel: str | Path, *, what: str = "path") -> None:
    """Raise ``ValueError`` if ``rel`` is absolute or contains ``..`` traversal.

    The base-independent half of :func:`safe_relpath` (the symlink-escape check
    needs a materialized tree and runs at stage time). An oracle calls this in
    its per-artifact ``validate`` so a candidate-supplied path that would later
    make ``safe_relpath`` raise becomes a single ``unsupported`` row -- the
    runner demotes ``UnsupportedArtifactError`` per candidate -- instead of a
    stage-time ``ValueError`` that aborts the whole batch.
    """
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError(f"unsafe {what} (absolute): {str(rel)!r}")
    if any(part == ".." for part in rel_path.parts):
        raise ValueError(f"unsafe {what} ('..' traversal): {str(rel)!r}")


__all__ = ["safe_relpath", "assert_relpath_within"]
