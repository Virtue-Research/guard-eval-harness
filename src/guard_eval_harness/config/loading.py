"""YAML → ResolvedRunConfig loader."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from guard_eval_harness.config.schema import (
    InlinePolicy,
    ResolvedBackendConfig,
    ResolvedDatasetSelection,
    ResolvedModelConfig,
    ResolvedOutputConfig,
    ResolvedRunConfig,
)


_UNRESOLVED_ENV_RE = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*"
)


def _expand_strings(value: Any, *, unresolved: set[str]) -> Any:
    """Expand ${ENV} sigils and collect any that fail to resolve."""
    if isinstance(value, dict):
        return {
            k: _expand_strings(v, unresolved=unresolved)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_strings(item, unresolved=unresolved)
            for item in value
        ]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        for match in _UNRESOLVED_ENV_RE.findall(expanded):
            unresolved.add(match.lstrip("$").strip("{}"))
        return expanded
    return value


def _apply_expansions(payload: Any) -> Any:
    """Expand env vars; raise on any unresolved sigils."""
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
    """Resolve a relative path against the YAML's directory."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.as_posix()


def _build_policy_payload(
    value: Any,
    *,
    where: str,
) -> str | InlinePolicy | None:
    """Coerce a raw policy spec into a registry name or inline policy."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return InlinePolicy.model_validate(value)
    raise ValueError(
        f"{where} must be a registry name (str) or an inline "
        "{name, text, categories?} object"
    )


def _build_model(payload: dict[str, Any]) -> ResolvedModelConfig:
    """Validate the ``model:`` block, including guard-compat checks."""
    if "guard" not in payload:
        raise ValueError("model.guard is required")
    if "backend" not in payload:
        raise ValueError("model.backend is required")
    backend_payload = payload["backend"]
    if not isinstance(backend_payload, dict):
        raise ValueError("model.backend must be a mapping")
    backend = ResolvedBackendConfig.model_validate(backend_payload)

    if "policy" in payload:
        raise ValueError(
            "model.policy is no longer supported. Move `policy:` under "
            "each dataset entry (the policy describes how to score the "
            "dataset, not a property of the guard)."
        )

    output_format = payload.get("output_format")
    guard_args = dict(payload.get("guard_args", {}))

    resolved = ResolvedModelConfig(
        guard=payload["guard"],
        output_format=output_format,
        guard_args=guard_args,
        backend=backend,
    )

    # Compat check: reject output_format on guards that don't advertise
    # support. We do this here (not in the pydantic model) so the schema
    # stays free of registry side effects.
    from guard_eval_harness.guards import get_guard_cls

    guard_cls = get_guard_cls(resolved.guard)
    if resolved.output_format is not None and not getattr(
        guard_cls, "accepts_output_format", False
    ):
        raise ValueError(
            f"guard {resolved.guard!r} does not accept a custom "
            "output_format; remove `model.output_format` from the YAML"
        )
    return resolved


def _build_dataset(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> ResolvedDatasetSelection:
    """Validate one ``datasets[]`` entry."""
    mapping = dict(payload)
    mapping["path"] = _resolve_relative_path(
        mapping.get("path"),
        base_dir=base_dir,
    )
    for key in ("sample_ids", "sample_indices"):
        if key in mapping and mapping[key] is not None:
            mapping[key] = tuple(mapping[key])
    if "policy" in mapping:
        mapping["policy"] = _build_policy_payload(
            mapping["policy"],
            where=f"datasets[{mapping.get('name', '?')!r}].policy",
        )
    if "policy" in mapping and "policy_source" in mapping:
        if mapping["policy"] is not None and mapping["policy_source"]:
            raise ValueError(
                f"datasets[{mapping.get('name', '?')!r}]: cannot set "
                "both `policy` and `policy_source` — `policy` already "
                "resolves to a concrete policy, `policy_source` is "
                "only used when no explicit `policy` is given."
            )
    return ResolvedDatasetSelection.model_validate(mapping)


def _build_output(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> ResolvedOutputConfig:
    """Validate the ``output:`` block, resolving run_dir against base_dir."""
    if "run_dir" not in payload:
        raise ValueError("output.run_dir is required")
    mapping = dict(payload)
    mapping["run_dir"] = _resolve_relative_path(
        mapping["run_dir"],
        base_dir=base_dir,
    )
    return ResolvedOutputConfig.model_validate(mapping)


def load_config(
    payload: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> ResolvedRunConfig:
    """Resolve an in-memory payload into a typed ``ResolvedRunConfig``."""
    expanded = _apply_expansions(payload)
    base = Path(base_dir)

    datasets_raw = expanded.get("datasets") or []
    if not datasets_raw:
        raise ValueError("at least one entry under `datasets` is required")

    datasets = [
        _build_dataset(entry, base_dir=base) for entry in datasets_raw
    ]
    model = _build_model(expanded["model"])
    output = _build_output(expanded.get("output", {}), base_dir=base)

    return ResolvedRunConfig(
        version=expanded.get("version", 2),
        run_name=expanded.get("run_name", "guard-eval-run"),
        threshold=expanded.get("threshold", 0.5),
        model=model,
        datasets=datasets,
        output=output,
        metadata=dict(expanded.get("metadata", {})),
    )


def load_config_from_path(
    path: str | Path,
    *,
    output_dir: str | None = None,
    threshold: float | None = None,
) -> ResolvedRunConfig:
    """Load a YAML file from disk, returning a typed config."""
    config_path = Path(path).expanduser().resolve()
    raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("config root must be a mapping")
    if raw_payload.get("version") != 2:
        raise ValueError(
            "load_config_from_path requires `version: 2`; got "
            f"{raw_payload.get('version')!r}"
        )

    payload = _apply_expansions(raw_payload)
    if output_dir is not None:
        payload.setdefault("output", {})
        payload["output"]["run_dir"] = output_dir
    if threshold is not None:
        payload["threshold"] = threshold

    return load_config(payload, base_dir=config_path.parent)


def detect_config_version(path: str | Path) -> int:
    """Peek at a YAML file's top-level `version:` key. Defaults to 1."""
    raw_payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        return 1
    return int(raw_payload.get("version", 1))
