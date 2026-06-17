"""SecureVibeBench task source.

Loads the upstream ARVO-derived task metadata (``data/<ARVO_ID>.json``) into
normalized :class:`VibeTask` records. Each instance is a
``repo_patch`` task scored by the ``securevibebench`` oracle, which runs the
ARVO Docker image (``n132/arvo:<id>-vul``) plus the PoV crash oracle out of
process.

The on-disk layout mirrors the upstream ``data/`` directory:

    {
      "1_szz_info": {"vic": "<sha>", "localid": <int>, "repo_url": "<url>"},
      "2_validate_result": {"PVIC": {"log": {"2_check_repo_cwd": {
          "output": "<repo cwd inside container>"}}}},
      "5_final_description": "<task instructions>"
    }

CWE/CVE labels are not present in the public ARVO metadata; when an instance
carries them (e.g. under ``1_szz_info``) they are surfaced into
:class:`TaskLabels`, otherwise the label lists stay empty.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from guard_eval_harness.execution.artifacts import atomic_text_writer
from guard_eval_harness.vibecoding.envs import EnvProvider
from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.safe_path import safe_relpath
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Upstream ships the per-instance task files inside this archive under
# ``data/``; the checkout's ``data/`` otherwise holds only the archive plus
# ``format_example.json`` (no numeric ``<ARVO_ID>.json`` files), so a fresh
# provision must extract it before any task can load.
_DATASET_ARCHIVE = "full_dataset.zip"


def _is_numeric_json_name(name: str) -> bool:
    """True for a flat ``<digits>.json`` member name (the task-file shape)."""
    return (
        "/" not in name
        and name.endswith(".json")
        and Path(name).stem.isdigit()
    )


def _safe_get(data: Any, *keys: str) -> Any:
    """Walk nested dict keys, returning ``None`` on any miss."""
    cur = data
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _as_list(value: Any) -> list[str]:
    """Coerce a scalar/list label field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@task_source_registry.register("securevibebench")
class SecureVibeBenchTaskSource(TaskSource):
    """Yield ``repo_patch`` tasks from SecureVibeBench's ARVO metadata."""

    name = "securevibebench"

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        """Optionally override the ``data/`` directory (used by fixtures)."""
        self._data_dir = Path(data_dir) if data_dir is not None else None

    def _resolved_data_dir(
        self, cache_dir: str | Path | None = None
    ) -> Path:
        """Explicit override, else the env provider's .geh checkout."""
        if self._data_dir is not None:
            return self._data_dir
        from guard_eval_harness.vibecoding.oracles.securevibebench import (
            SecureVibeBenchOracle,
        )

        provider = EnvProvider(
            SecureVibeBenchOracle.env, cache_dir=cache_dir
        )
        return Path(provider.resolve().upstream_dir) / "data"

    def _ensure_dataset_extracted(self, data_dir: Path) -> None:
        """Materialize the per-instance ``<ARVO_ID>.json`` task files.

        Upstream ships the task files inside ``data/full_dataset.zip``; a
        fresh checkout's ``data/`` carries only that archive (plus
        ``format_example.json``), so the glob in :meth:`load` would find
        nothing. When the archive is present, reconcile every numeric member
        onto disk (extracting any that are missing).

        The archive is authoritative, and reconciliation is complete rather
        than short-circuiting on the first existing file: the sharded runner
        starts several processes that each call :meth:`load`, so a guard that
        returned as soon as one file existed could let a process glob a
        partial set while another was mid-extraction. Each call instead
        guarantees the full member set is present before it returns. Writes
        are per-file atomic (unique temp + ``os.replace``) and skip files
        already on disk, so concurrent extractors and repeat calls are safe
        and cheap. A dir with no archive (a fixture or an explicit override
        already carrying loose files) is a no-op.
        """
        archive = data_dir / _DATASET_ARCHIVE
        if not archive.is_file():
            return
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not _is_numeric_json_name(member):
                    continue
                # Confine each member under data_dir (zip-slip defense; the
                # pinned upstream archive is flat + trusted, but the guard
                # keeps a tampered cache from escaping the checkout).
                try:
                    target = safe_relpath(data_dir, member)
                except ValueError:
                    continue
                if target.exists():
                    continue
                text = zf.read(member).decode("utf-8")
                with atomic_text_writer(target) as handle:
                    handle.write(text)

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[VibeTask]:
        """Return normalized tasks for each ``data/<ARVO_ID>.json`` file."""
        data_dir = self._resolved_data_dir(cache_dir)
        self._ensure_dataset_extracted(data_dir)
        paths = sorted(
            p
            for p in data_dir.glob("*.json")
            if p.stem.isdigit()
        )
        tasks: list[VibeTask] = []
        for path in paths:
            task = self._load_one(path)
            if task is not None:
                tasks.append(task)
            if limit is not None and len(tasks) >= max(0, int(limit)):
                break
        return tasks

    def _load_one(self, path: Path) -> VibeTask | None:
        """Parse a single ARVO metadata JSON into a ``VibeTask``."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        szz = _safe_get(data, "1_szz_info") or {}
        arvo_id = szz.get("localid", path.stem)
        arvo_id_str = str(arvo_id)
        vic = szz.get("vic")
        repo_url = szz.get("repo_url")
        # PVIC (parent of VIC) is computed upstream from a git checkout; it is
        # not stored in the JSON, so we only carry VIC here. The oracle's
        # extract step recomputes PVIC at evaluation time.
        repo_cwd = _safe_get(
            data,
            "2_validate_result",
            "PVIC",
            "log",
            "2_check_repo_cwd",
            "output",
        )
        instructions = _safe_get(data, "5_final_description") or ""

        cwe = _as_list(szz.get("cwe"))
        cve = _as_list(szz.get("cve"))

        # ``VibeTask`` is ``extra="forbid"`` with no free-form metadata field.
        # The ARVO-specific fields the oracle needs are carried structurally:
        #   - arvo_id  -> the task id suffix (``securevibebench/<arvo_id>``)
        #   - vic      -> ``repo.base_commit`` (vuln-introducing commit)
        #   - repo cwd -> ``repo.workdir`` (the repo path inside the container)
        # PVIC (parent of VIC) is recomputed by the oracle's extract step at
        # evaluation time, so it is not stored on the task.
        return VibeTask(
            id=f"securevibebench/{arvo_id_str}",
            source_dataset="securevibebench",
            task_type="repo_patch",
            instructions=str(instructions),
            repo=RepoSpec(
                url=repo_url,
                base_commit=vic,
                workdir=str(repo_cwd) if repo_cwd else ".",
            ),
            labels=TaskLabels(cwe=cwe, cve=cve),
            environment=TaskEnvironmentRef(
                oracle="securevibebench",
                requires_docker=True,
            ),
        )
