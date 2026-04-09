"""Disk cache for normalized dataset samples.

Caches ``list[NormalizedSample]`` as JSONL so that subsequent runs
skip the expensive ``datasets.load_dataset()`` + PIL decode path
entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from guard_eval_harness.schemas import NormalizedSample

_log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = (
    Path.home() / ".cache" / "guard-eval-harness" / "samples"
)

# Bump this when the cache format or normalization logic changes.
# Any cached entries written with a different version are ignored.
_CACHE_VERSION = "3"


def sample_cache_base_dir() -> Path:
    """Return the default sample cache directory."""
    return _DEFAULT_CACHE_DIR


def compute_cache_key(parts: dict[str, Any]) -> str:
    """Compute a deterministic cache key from *parts*.

    All keys are preserved in the hash — ``None`` values are
    kept as-is so that ``execution_limit=None`` (full dataset)
    produces a different key from ``execution_limit=1``.
    A cache-format version is always included so that code
    changes that affect normalization invalidate old entries.
    """
    cleaned = dict(sorted(parts.items()))
    cleaned["_cache_version"] = _CACHE_VERSION
    payload = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_entry_dir(
    cache_dir: Path,
    adapter_name: str,
    cache_key: str,
) -> Path:
    return cache_dir / adapter_name / cache_key


def _media_uris(samples: list[NormalizedSample]) -> list[str]:
    """Extract all media URIs referenced by *samples*."""
    uris: list[str] = []
    for sample in samples:
        for message in sample.messages:
            for ref in message.media_refs:
                uris.append(ref.uri)
    return uris


def load_cached_samples(
    cache_dir: Path,
    adapter_name: str,
    cache_key: str,
) -> list[NormalizedSample] | None:
    """Load cached samples from disk.

    Returns ``None`` when the cache is missing, corrupt, or any
    referenced media file no longer exists on disk.
    """
    samples_path = (
        _cache_entry_dir(cache_dir, adapter_name, cache_key)
        / "samples.jsonl"
    )
    if not samples_path.exists():
        return None

    try:
        samples: list[NormalizedSample] = []
        with open(samples_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                samples.append(
                    NormalizedSample.model_validate_json(line)
                )
    except Exception:
        _log.warning(
            "Corrupt sample cache for %s — rebuilding",
            adapter_name,
        )
        return None

    # Verify every referenced media file still exists.
    for uri in _media_uris(samples):
        if not Path(uri).exists():
            _log.info(
                "Cached media file missing (%s) — rebuilding "
                "cache for %s",
                uri,
                adapter_name,
            )
            return None

    return samples


def write_sample_cache(
    cache_dir: Path,
    adapter_name: str,
    cache_key: str,
    samples: list[NormalizedSample],
) -> Path:
    """Persist *samples* as JSONL with an atomic rename.

    Returns the path to the written ``samples.jsonl``.
    """
    entry_dir = _cache_entry_dir(cache_dir, adapter_name, cache_key)
    entry_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = entry_dir / f"_tmp_{uuid.uuid4().hex}.jsonl"
    final_path = entry_dir / "samples.jsonl"
    try:
        with open(tmp_path, "w") as fh:
            for sample in samples:
                fh.write(sample.model_dump_json() + "\n")
        os.replace(str(tmp_path), str(final_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path


def clear_sample_cache(
    cache_dir: Path | None = None,
    adapter_name: str | None = None,
) -> int:
    """Remove cached sample entries.

    If *adapter_name* is given only that adapter's subdirectory is
    removed.  Returns the number of top-level entries removed.
    """
    base = cache_dir or sample_cache_base_dir()
    if not base.exists():
        return 0

    if adapter_name is not None:
        target = (base / adapter_name).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(
                f"adapter_name would escape cache root: "
                f"{adapter_name!r}"
            )
        if not target.exists():
            return 0
        shutil.rmtree(target)
        return 1

    count = 0
    for child in list(base.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            count += 1
    return count
