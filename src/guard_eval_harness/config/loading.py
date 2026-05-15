"""Helpers for loading and resolving config files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from guard_eval_harness.config.models import (
    ResolvedDatasetConfig,
    ResolvedExecutionConfig,
    ResolvedModelConfig,
    ResolvedOutputConfig,
    ResolvedRunConfig,
)
from guard_eval_harness.models.catalog import resolve_templates

# Matches the ${VAR} or $VAR sigils left behind when os.path.expandvars
# encounters an unset environment variable. We collect these so we can
# surface a single clear error instead of a downstream FileNotFoundError
# whose path contains an unresolved ``${...}`` substring.
_UNRESOLVED_ENV_RE = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*"
)


def _expand_strings(value: Any, *, unresolved: set[str]) -> Any:
    """Expand named templates and env vars; record unresolved ${VAR} sigils.

    Named templates (``$llama_guard_taxonomy`` and friends) are substituted
    via :func:`resolve_templates` so raw user YAMLs get the same expansion
    that catalog-shipped configs already enjoy. Environment variables go
    through :func:`os.path.expandvars`; any sigils that survive expansion
    are collected in ``unresolved`` so the caller can raise a single,
    clear error.
    """
    if isinstance(value, dict):
        return {k: _expand_strings(v, unresolved=unresolved) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_strings(v, unresolved=unresolved) for v in value]
    if isinstance(value, str):
        templated = resolve_templates(value)
        if templated is not value:
            return templated
        expanded = os.path.expandvars(value)
        for match in _UNRESOLVED_ENV_RE.findall(expanded):
            unresolved.add(match.lstrip("$").strip("{}"))
        return expanded
    return value


def _apply_expansions(payload: Any) -> Any:
    """Expand strings in ``payload`` and raise on unresolved env vars."""
    unresolved: set[str] = set()
    expanded = _expand_strings(payload, unresolved=unresolved)
    if unresolved:
        names = ", ".join(sorted(unresolved))
        raise ValueError(
            "config references unset environment variable(s): "
            f"{names}. Set them in your shell (or .env file) before "
            "running, or replace the placeholder with a literal value."
        )
    return expanded


def _resolve_relative_path(
    value: str | None,
    *,
    base_dir: Path,
) -> str | None:
    """Resolve a path relative to the config file directory."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.as_posix()


def _build_config(payload: dict[str, Any], *, base_dir: Path) -> ResolvedRunConfig:
    """Build a resolved config from a raw mapping."""
    dataset_payloads = []
    for raw_dataset in payload.get("datasets", []):
        dataset_mapping = dict(raw_dataset)
        dataset_mapping["path"] = _resolve_relative_path(
            dataset_mapping.get("path"),
            base_dir=base_dir,
        )
        dataset_payloads.append(ResolvedDatasetConfig.model_validate(dataset_mapping))

    output_payload = dict(payload.get("output", {}))
    output_payload["run_dir"] = _resolve_relative_path(
        output_payload.get("run_dir"),
        base_dir=base_dir,
    )

    if not output_payload.get("run_dir"):
        raise ValueError("output.run_dir is required")

    resolved = ResolvedRunConfig(
        version=payload.get("version", 1),
        run_name=payload.get("run_name", "guard-eval-run"),
        threshold=payload.get("threshold", 0.5),
        model=ResolvedModelConfig.model_validate(payload["model"]),
        datasets=dataset_payloads,
        execution=ResolvedExecutionConfig.model_validate(
            payload.get("execution", {})
        ),
        output=ResolvedOutputConfig.model_validate(output_payload),
        warnings=list(payload.get("warnings", [])),
        metadata=dict(payload.get("metadata", {})),
    )
    return resolved


def load_config(payload: dict[str, Any], *, base_dir: str | Path = ".") -> ResolvedRunConfig:
    """Resolve a config payload into stable config models."""
    expanded = _apply_expansions(payload)
    return _build_config(expanded, base_dir=Path(base_dir))


def load_config_from_path(
    path: str | Path,
    *,
    output_dir: str | None = None,
    threshold: float | None = None,
    limit: int | None = None,
) -> ResolvedRunConfig:
    """Load and resolve a YAML config from disk."""
    config_path = Path(path).expanduser().resolve()
    raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("config root must be a mapping")

    payload = _apply_expansions(raw_payload)
    if output_dir is not None:
        payload.setdefault("output", {})
        payload["output"]["run_dir"] = output_dir
    if threshold is not None:
        payload["threshold"] = threshold
    if limit is not None:
        payload.setdefault("execution", {})
        payload["execution"]["limit"] = limit

    return _build_config(payload, base_dir=config_path.parent)
