"""Artifact writing helpers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from pydantic import BaseModel

REDACTED_VALUE = "***REDACTED***"
_SENSITIVE_HEADER_KEYS = {
    "apikey",
    "authorization",
    "cookie",
    "proxyauthorization",
    "setcookie",
    "xapikey",
    "xauthtoken",
}
_SENSITIVE_KEY_SUFFIXES = ("apikey", "password", "secret", "token")


def _compact_key(value: str) -> str:
    """Normalize a key into a compact comparison form."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_sensitive_key_name(compact: str) -> bool:
    """Identify compact key names that should not hit artifacts."""
    if compact in _SENSITIVE_HEADER_KEYS:
        return True
    return compact.endswith(_SENSITIVE_KEY_SUFFIXES)


def _should_redact_key(key: str, parent_keys: tuple[str, ...]) -> bool:
    """Identify sensitive config-like keys that should not hit artifacts."""
    compact = _compact_key(key)
    if compact.endswith("env"):
        return _is_sensitive_key_name(compact[:-3])
    return _is_sensitive_key_name(compact)


def sanitize_payload_for_artifacts(
    payload: Any,
    *,
    parent_keys: tuple[str, ...] = (),
) -> Any:
    """Redact obviously sensitive config values before writing artifacts."""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if _should_redact_key(key, parent_keys):
                sanitized[key] = REDACTED_VALUE if value is not None else None
                continue
            sanitized[key] = sanitize_payload_for_artifacts(
                value,
                parent_keys=parent_keys + (_compact_key(key),),
            )
        return sanitized
    if isinstance(payload, list):
        return [
            sanitize_payload_for_artifacts(value, parent_keys=parent_keys)
            for value in payload
        ]
    return payload


def build_resume_signature_payload(
    payload: Any,
    *,
    parent_keys: tuple[str, ...] = (),
) -> Any:
    """Hash sensitive values while preserving resume-signature shape."""
    if isinstance(payload, dict):
        signature: dict[str, Any] = {}
        for key, value in payload.items():
            if _should_redact_key(key, parent_keys):
                signature[key] = (
                    None
                    if value is None
                    else f"sha256:{sha256_payload(value)}"
                )
                continue
            signature[key] = build_resume_signature_payload(
                value,
                parent_keys=parent_keys + (_compact_key(key),),
            )
        return signature
    if isinstance(payload, list):
        return [
            build_resume_signature_payload(value, parent_keys=parent_keys)
            for value in payload
        ]
    return payload


def ensure_run_layout(run_dir: str | Path) -> Path:
    """Create the stable artifact layout for a run."""
    root = Path(run_dir)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def _atomic_text_writer(path: str | Path) -> Iterator[TextIO]:
    """Yield a temp-file text handle and atomically replace on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            yield handle
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_text_atomic(path: str | Path, payload: str) -> None:
    """Write a UTF-8 text payload atomically."""
    with _atomic_text_writer(path) as handle:
        handle.write(payload)


def dump_json(path: str | Path, payload: Any) -> None:
    """Write JSON with deterministic formatting."""
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True),
    )


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL deterministically."""
    with _atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def dump_model(path: str | Path, model: BaseModel) -> None:
    """Write a pydantic model as deterministic JSON."""
    dump_json(path, model.model_dump(mode="json"))


def sha256_payload(payload: Any) -> str:
    """Fingerprint a JSON-like payload via sorted JSON."""
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_model(model: BaseModel) -> str:
    """Fingerprint a model via sorted JSON."""
    return sha256_payload(model.model_dump(mode="json"))
