"""Oracle-result cache keyed by the doc's reproducibility tuple.

A result is cacheable only when it is environment-independent enough to
replay: the runner caches ``completed`` / ``model_failure`` rows and never
caches ``infra_failure`` / ``unsupported`` rows, nor any row whose
``failure_origin`` is ``infra`` (an environment-dependent verdict, e.g. an
unverifiable submission scored as an upstream-parity fail) -- those must be
retried. This module owns the key derivation and the on-disk store; the
skip/retry policy lives in the runner.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import ValidationError

from guard_eval_harness.execution.artifacts import (
    dump_model,
    sha256_payload,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import VibeModel

# Subdirectory under the resolved cache root that holds vibe oracle results.
VIBE_CACHE_SUBDIR = Path("cache") / "vibecoding"

# Directory name of the cache root under the repo / workspace root.
CACHE_DIR_NAME = ".geh"


# The pyproject ``name`` line that identifies the harness checkout. The walk
# must not adopt just ANY ancestor pyproject.toml: under a wheel install into
# a project-local venv (``<app>/.venv/lib/.../site-packages/...``) the first
# ancestor pyproject.toml belongs to the consuming application, and treating
# it as the repo root would drop ``.geh`` into an unrelated project. Matched
# textually rather than TOML-parsed: tomllib is stdlib only from 3.11 and
# this project supports 3.10.
_HARNESS_PYPROJECT_NAME = re.compile(
    r"^\s*name\s*=\s*[\"']guard-eval-harness[\"']\s*$", re.MULTILINE
)


def _is_harness_root(candidate: Path) -> bool:
    """True when ``candidate/pyproject.toml`` names this project."""
    pyproject = candidate / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_HARNESS_PYPROJECT_NAME.search(text))


def _repo_root(start: str | Path | None = None) -> Path:
    """Locate the harness checkout root: nearest ancestor of ``start``
    (default: this file) whose ``pyproject.toml`` names guard-eval-harness,
    falling back to the cwd.

    The walk finds the checkout root under an editable install. For any
    non-checkout install — including a wheel inside another project's venv,
    whose own pyproject.toml the name check refuses — the fallback keeps the
    cache workspace-local instead of landing under the interpreter prefix or
    an unrelated project.
    """
    here = (Path(start) if start is not None else Path(__file__)).resolve()
    for parent in here.parents:
        if _is_harness_root(parent):
            return parent
    return Path.cwd()


def resolve_cache_dir(
    arg_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve the ``.geh`` cache root.

    Precedence: an explicit ``arg_dir``, then the ``GEH_CACHE_DIR`` env var,
    then ``<repo_root>/.geh`` (``repo_root`` defaults to :func:`_repo_root`).
    This is the single canonical resolver — ``VibeRunner``, ``EnvProvider``,
    and the oracle fallbacks all share it so that task loading and oracle
    evaluation resolve the same cache root instead of diverging.
    """
    if arg_dir is not None:
        return Path(arg_dir).resolve()
    env_value = os.environ.get("GEH_CACHE_DIR")
    if env_value:
        return Path(env_value).resolve()
    base = Path(repo_root) if repo_root is not None else _repo_root()
    return (base / CACHE_DIR_NAME).resolve()


class CacheKey(VibeModel):
    """The reproducibility tuple used to key an oracle result.

    Mirrors the spec's 9-tuple (the field list in the architecture doc):
    task id, artifact content hash, adapter name + version, upstream ref,
    oracle config hash, oracle capabilities hash, trial index, random seed,
    and the anti-cheat policy hash.
    """

    task_id: str
    artifact_sha256: str
    adapter_name: str
    adapter_version: str
    upstream_ref: str | None = None
    oracle_config_hash: str
    oracle_capabilities_hash: str
    trial_index: int = 0
    random_seed: int | None = None
    anti_cheat_policy_hash: str

    def digest(self) -> str:
        """Deterministic digest of the key via canonical sorted JSON."""
        return sha256_payload(self.model_dump(mode="json"))


class OracleResultCache:
    """File-backed cache of :class:`VibeTaskResult` rows.

    Each entry is stored as ``<base_dir>/<digest>.json``. ``get`` returns
    ``None`` on a miss or on any unreadable/invalid entry (treated as a
    miss rather than an error).
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _path(self, key: CacheKey) -> Path:
        return self.base_dir / f"{key.digest()}.json"

    def get(self, key: CacheKey) -> VibeTaskResult | None:
        """Return the cached result for ``key`` or ``None`` on a miss."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = path.read_text(encoding="utf-8")
            return VibeTaskResult.model_validate_json(payload)
        except (OSError, ValueError, ValidationError):
            return None

    def put(self, key: CacheKey, result: VibeTaskResult) -> Path:
        """Store ``result`` under ``key`` and return the written path."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        dump_model(path, result)
        return path
