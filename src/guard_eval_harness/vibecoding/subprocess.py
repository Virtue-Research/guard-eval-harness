"""Out-of-process command execution for the vibecoding subsystem.

Pure stdlib. ``run_command`` runs an upstream CLI in its own process group
with a hard timeout, captures stdout/stderr (optionally to files), records an
environment fingerprint, and returns a structured :class:`CommandResult`. It
never raises on a nonzero exit code; the only exception it propagates is
``OSError`` (e.g. the binary is missing or not executable).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from guard_eval_harness.execution.artifacts import (
    atomic_text_writer,
    sanitize_payload_for_artifacts,
)
from guard_eval_harness.vibecoding.schema import VibeModel
from pydantic import Field


def which(binary: str) -> str | None:
    """Resolve a binary on PATH (thin ``shutil.which`` wrapper)."""
    return shutil.which(binary)


def _env_fingerprint(env: dict[str, str]) -> str:
    """Fingerprint the resolved environment (keys + redacted values).

    Sensitive values are redacted before hashing so the fingerprint never
    leaks secrets while still changing when a secret value changes (the
    redaction marker is constant, so only key membership is captured here;
    that is the intended, audit-safe behavior).
    """
    redacted = sanitize_payload_for_artifacts(dict(env))
    items = sorted(redacted.items())
    serialized = "\n".join(f"{k}={v}" for k, v in items)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CommandResult(VibeModel):
    """Structured outcome of a single ``run_command`` invocation."""

    argv: list[str]
    cwd: str | None = None
    returncode: int | None = None
    timed_out: bool = False
    duration_s: float = Field(default=0.0, ge=0.0)
    stdout: str = ""
    stderr: str = ""
    stdout_path: str | None = None
    stderr_path: str | None = None
    env_fingerprint: str | None = None
    redacted_env: dict[str, str] = Field(default_factory=dict)


def run_command(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    capture_to_files: str | Path | None = None,
    stdout_name: str = "stdout.log",
    stderr_name: str = "stderr.log",
) -> CommandResult:
    """Run ``argv`` with timeout + process-group kill; never raise on rc.

    The child runs in a new session (``start_new_session=True``) so a timeout
    can kill the entire process group via :func:`os.killpg`. On a nonzero
    exit the result is returned with that ``returncode``; only ``OSError``
    (e.g. missing binary) propagates.
    """
    cwd_str = str(cwd) if cwd is not None else None
    resolved_env = dict(os.environ if env is None else env)
    redacted_env = sanitize_payload_for_artifacts(dict(resolved_env))
    fingerprint = _env_fingerprint(resolved_env)

    started = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603 - argv is a list, no shell
        list(argv),
        cwd=cwd_str,
        env=resolved_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
    duration_s = time.monotonic() - started

    returncode = None if timed_out else proc.returncode

    stdout_path: str | None = None
    stderr_path: str | None = None
    if capture_to_files is not None:
        out_dir = Path(capture_to_files)
        stdout_path = str(out_dir / stdout_name)
        stderr_path = str(out_dir / stderr_name)
        with atomic_text_writer(stdout_path) as handle:
            handle.write(stdout or "")
        with atomic_text_writer(stderr_path) as handle:
            handle.write(stderr or "")

    return CommandResult(
        argv=list(argv),
        cwd=cwd_str,
        returncode=returncode,
        timed_out=timed_out,
        duration_s=duration_s,
        stdout=stdout or "",
        stderr=stderr or "",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env_fingerprint=fingerprint,
        redacted_env=redacted_env,
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGKILL of the child's whole process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


__all__ = ["CommandResult", "run_command", "which"]
