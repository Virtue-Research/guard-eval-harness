"""Helpers for loading and resolving config files."""

from __future__ import annotations

import os
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


def _expand_env(value: Any) -> Any:
    """Recursively expand environment variables in config payloads."""
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


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
    expanded = _expand_env(payload)
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

    payload = _expand_env(raw_payload)
    if output_dir is not None:
        payload.setdefault("output", {})
        payload["output"]["run_dir"] = output_dir
    if threshold is not None:
        payload["threshold"] = threshold
    if limit is not None:
        payload.setdefault("execution", {})
        payload["execution"]["limit"] = limit

    return _build_config(payload, base_dir=config_path.parent)
