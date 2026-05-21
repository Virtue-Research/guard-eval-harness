"""Resumable, streaming runner for guard × backend runs."""

import hashlib
import json
import logging
import os
import platform
import re
import socket
import sys
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

from guard_eval_harness import __version__
from guard_eval_harness.backends import BackendConfig, get_backend_cls
from guard_eval_harness.backends.base import (
    Backend,
    ClassifierBackend,
    GenerationBackend,
)
from guard_eval_harness.config import (
    InlinePolicy,
    ResolvedDatasetSelection,
    ResolvedRunConfig,
)
from guard_eval_harness.guards import Guard, get_guard_cls
from guard_eval_harness.guards.llm import LLMGuard
from guard_eval_harness.metrics import compute_binary_metrics
from guard_eval_harness.output_formats import (
    OutputFormat,
    get_output_format,
)
from guard_eval_harness.policies import Policy, get_policy
from guard_eval_harness.registry import (
    dataset_registry,
    ensure_builtin_registrations,
)
from guard_eval_harness.schemas import (
    Message,
    NormalizedPrediction,
    NormalizedSample,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact helpers: atomic JSON write, run layout, payload sanitization
# ---------------------------------------------------------------------------


_REDACTED_VALUE = "***REDACTED***"
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


def _should_redact_key(key: str) -> bool:
    """Identify sensitive config-like keys that should not hit artifacts."""
    compact = _compact_key(key)
    if compact.endswith("env"):
        return _is_sensitive_key_name(compact[:-3])
    return _is_sensitive_key_name(compact)


def _sanitize_payload_for_artifacts(payload: Any) -> Any:
    """Redact obviously sensitive config values before writing artifacts."""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if _should_redact_key(key):
                sanitized[key] = (
                    _REDACTED_VALUE if value is not None else None
                )
                continue
            sanitized[key] = _sanitize_payload_for_artifacts(value)
        return sanitized
    if isinstance(payload, list):
        return [_sanitize_payload_for_artifacts(v) for v in payload]
    return payload


def _ensure_run_layout(run_dir: str | Path) -> Path:
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


def _dump_json(path: str | Path, payload: Any) -> None:
    """Write JSON atomically with deterministic formatting."""
    with _atomic_text_writer(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Public result + entry points
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunResult:
    """Result returned by the runner."""

    run_dir: str
    manifest_path: str
    summary_path: str
    completed_predictions: int
    failed_predictions: int


def run_from_config(
    config: ResolvedRunConfig,
    *,
    recompute_metrics_only: bool = False,
) -> RunResult:
    """Execute a run end-to-end from a resolved config."""
    ensure_builtin_registrations()
    guard, policy, output_format = _build_guard(config)
    backend = _build_backend(config)
    return _execute_run(
        config=config,
        guard=guard,
        policy=policy,
        output_format=output_format,
        backend=backend,
        recompute_metrics_only=recompute_metrics_only,
    )


def run_benchmark(
    *,
    guard: Guard,
    backend: GenerationBackend,
    config: ResolvedRunConfig,
    recompute_metrics_only: bool = False,
) -> RunResult:
    """Programmatic entry point: caller supplies prebuilt guard + backend."""
    ensure_builtin_registrations()
    policy = _resolve_policy(config)
    output_format = _resolve_output_format(config)
    return _execute_run(
        config=config,
        guard=guard,
        policy=policy,
        output_format=output_format,
        backend=backend,
        recompute_metrics_only=recompute_metrics_only,
    )


# ---------------------------------------------------------------------------
# Builders: resolved config → live Guard / Backend / Policy / OutputFormat
# ---------------------------------------------------------------------------


def _resolve_policy(config: ResolvedRunConfig) -> Policy | None:
    """Materialize a Policy from a registry name or inline definition."""
    spec = config.model.policy
    if spec is None:
        return None
    if isinstance(spec, InlinePolicy):
        return Policy(
            name=spec.name,
            text=spec.text,
            categories=tuple(spec.categories),
        )
    return get_policy(spec)


def _resolve_output_format(
    config: ResolvedRunConfig,
) -> OutputFormat | None:
    """Look up an OutputFormat by registry name, if any."""
    name = config.model.output_format
    if name is None:
        return None
    return get_output_format(name)


def _build_guard(
    config: ResolvedRunConfig,
) -> tuple[Guard, Policy | None, OutputFormat | None]:
    """Instantiate the configured Guard with sensible defaults."""
    guard_cls = get_guard_cls(config.model.guard)
    policy = _resolve_policy(config)
    output_format = _resolve_output_format(config)

    kwargs: dict[str, Any] = dict(config.model.guard_args)
    if guard_cls is LLMGuard:
        if policy is not None:
            kwargs.setdefault("default_policy", policy)
        if output_format is not None:
            kwargs.setdefault("default_output_format", output_format)
        guard: Guard = LLMGuard(**kwargs)
    else:
        guard = guard_cls(**kwargs)
    return guard, policy, output_format


def _build_backend(config: ResolvedRunConfig) -> Backend:
    """Instantiate the configured backend (generate or classify)."""
    backend_cls = get_backend_cls(config.model.backend.kind)
    backend_config = BackendConfig(
        kind=config.model.backend.kind,
        model=config.model.backend.name,
        args=dict(config.model.backend.args),
    )
    backend = backend_cls.from_config(backend_config)
    if not isinstance(backend, (GenerationBackend, ClassifierBackend)):
        raise TypeError(
            f"backend {config.model.backend.kind!r} must be a "
            "GenerationBackend or ClassifierBackend"
        )
    return backend


def _invoke_backend(
    guard: "Guard",
    backend: Backend,
    messages: Sequence[Message],
) -> Any:
    """Dispatch one inference call based on the guard's backend_kind."""
    if guard.backend_kind == "classify":
        if not isinstance(backend, ClassifierBackend):
            raise TypeError(
                f"guard {guard.name!r} requires a ClassifierBackend, "
                f"got {type(backend).__name__}"
            )
        results = backend.classify([list(messages)])
        return results[0] if results else {}
    # default: generate
    if not isinstance(backend, GenerationBackend):
        raise TypeError(
            f"guard {guard.name!r} requires a GenerationBackend, "
            f"got {type(backend).__name__}"
        )
    outputs = backend.generate([list(messages)])
    return outputs[0] if outputs else ""


# ---------------------------------------------------------------------------
# Config hash + serializable view (used for the resume invariant)
# ---------------------------------------------------------------------------


def _canonical_config_view(config: ResolvedRunConfig) -> dict[str, Any]:
    """Produce the canonical fingerprintable view of a run config.

    Excludes fields that don't affect predictions (run_name, run_dir,
    resume, overwrite) and resolves inline policies into their full
    text so a renamed policy is detected as a change.
    """
    policy_view: Any
    if config.model.policy is None:
        policy_view = None
    elif isinstance(config.model.policy, InlinePolicy):
        policy_view = {
            "kind": "inline",
            "name": config.model.policy.name,
            "text": config.model.policy.text,
            "categories": list(config.model.policy.categories),
        }
    else:
        resolved = get_policy(config.model.policy)
        policy_view = {
            "kind": "registry",
            "name": resolved.name,
            "text": resolved.text,
            "categories": list(resolved.categories),
        }

    datasets_view = []
    for dataset in config.datasets:
        datasets_view.append(
            {
                "name": dataset.name,
                "adapter": dataset.adapter,
                "path": dataset.path,
                "split": dataset.split,
                "limit": dataset.limit,
                "sample_ids": list(dataset.sample_ids),
                "sample_indices": list(dataset.sample_indices),
                "options": dict(dataset.options),
            }
        )

    return {
        "version": config.version,
        "threshold": config.threshold,
        "model": {
            "guard": config.model.guard,
            "policy": policy_view,
            "output_format": config.model.output_format,
            "guard_args": dict(config.model.guard_args),
            "backend": {
                "kind": config.model.backend.kind,
                "name": config.model.backend.name,
                "args": dict(config.model.backend.args),
            },
        },
        "datasets": datasets_view,
    }


def _config_hash(config: ResolvedRunConfig) -> str:
    """Stable SHA-256 over the canonical config view."""
    view = _canonical_config_view(config)
    payload = json.dumps(view, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sample selection (subset filters)
# ---------------------------------------------------------------------------


def _select_samples(
    samples: Sequence[NormalizedSample],
    selection: ResolvedDatasetSelection,
) -> list[NormalizedSample]:
    """Apply the dataset's subset selector (at most one is set)."""
    if selection.sample_indices:
        out: list[NormalizedSample] = []
        max_index = max(selection.sample_indices)
        if max_index >= len(samples):
            raise IndexError(
                f"dataset {selection.name!r}: sample_indices include "
                f"{max_index} but the dataset has only {len(samples)} "
                "samples"
            )
        for idx in selection.sample_indices:
            out.append(samples[idx])
        return out
    if selection.sample_ids:
        wanted = set(selection.sample_ids)
        out = [s for s in samples if s.id in wanted]
        found = {s.id for s in out}
        missing = sorted(wanted - found)
        if missing:
            preview = ", ".join(missing[:5])
            extra = (
                f" (and {len(missing) - 5} more)"
                if len(missing) > 5
                else ""
            )
            raise ValueError(
                f"dataset {selection.name!r}: sample_ids not found "
                f"in dataset: {preview}{extra}"
            )
        return out
    if selection.limit is not None:
        return list(samples[: selection.limit])
    return list(samples)


# ---------------------------------------------------------------------------
# Streaming predictions writer
# ---------------------------------------------------------------------------


class _PredictionsWriter:
    """Append-only JSONL writer that fsyncs after each record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # `a` keeps any prior content; resume relies on append semantics.
        self._handle = path.open("a", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        """Write one record; fsync so a crash loses at most this line."""
        self._handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
        )
        self._handle.write("\n")
        self._handle.flush()
        try:
            os.fsync(self._handle.fileno())
        except OSError:
            # fsync isn't guaranteed on every filesystem (e.g. tmpfs);
            # the flush above is the best we can do there.
            pass

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()

    def __enter__(self) -> "_PredictionsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    """Read prior predictions; tolerate a trailing partial line."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    truncated = False
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Crash mid-write — drop the partial line and rewrite the
            # file without it so the writer can safely append.
            truncated = True
            break
    if truncated:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    )
                )
                handle.write("\n")
    return records


def _record_to_normalized_prediction(
    record: dict[str, Any],
    *,
    threshold: float,
) -> NormalizedPrediction | None:
    """Reconstruct a NormalizedPrediction from a per-sample JSONL record.

    Returns None for records that represent failed inferences.
    """
    if record.get("error") is not None:
        return None
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return NormalizedPrediction(
        sample_id=record["sample_id"],
        unsafe_score=float(parsed["unsafe_score"]),
        unsafe_label=bool(record["unsafe_label"]),
        threshold=threshold,
        latency_ms=float(record.get("latency_ms") or 0.0),
        predicted_categories=tuple(
            parsed.get("predicted_categories") or ()
        ),
        category_scores=dict(parsed.get("category_scores") or {}),
    )


# ---------------------------------------------------------------------------
# Per-dataset execution
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _resolve_dataset_config(
    selection: ResolvedDatasetSelection,
):
    """Build the v1 ResolvedDatasetConfig that the adapter expects."""
    from guard_eval_harness.config import ResolvedDatasetConfig

    return ResolvedDatasetConfig(
        name=selection.name,
        adapter=selection.adapter,
        path=selection.path,
        split=selection.split,
        options=dict(selection.options),
    )


def _ensure_dataset_manifest(
    dataset_dir: Path,
    selection: ResolvedDatasetSelection,
    samples: Sequence[NormalizedSample],
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Read or write the frozen sample manifest for one dataset."""
    manifest_path = dataset_dir / "dataset-manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_hash = prior.get("config_hash")
        if prior_hash != config_hash:
            raise ValueError(
                f"dataset {selection.name!r}: existing dataset-manifest "
                "was produced by a different config (config_hash "
                f"{prior_hash!r} vs {config_hash!r}). Re-run with "
                "output.overwrite=true to discard the prior run."
            )
        return prior
    manifest = {
        "name": selection.name,
        "adapter": selection.adapter,
        "split": selection.split,
        "config_hash": config_hash,
        "sample_count": len(samples),
        "samples": [
            {"row_index": idx, "sample_id": sample.id}
            for idx, sample in enumerate(samples)
        ],
    }
    _dump_json(manifest_path, manifest)
    return manifest


def _build_record(
    *,
    sample: NormalizedSample,
    row_index: int,
    raw_output: Any,
    parsed_label,
    threshold: float,
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    """Build one streaming JSONL record (per the schema in OSS_REDESIGN §7.F)."""
    if parsed_label is not None and error is None:
        parsed_payload = {
            "unsafe_score": float(parsed_label.unsafe_score),
            "predicted_categories": list(
                parsed_label.predicted_categories
            ),
            "category_scores": dict(parsed_label.category_scores),
            "raw_metadata": dict(parsed_label.raw_metadata),
        }
        unsafe_label = parsed_label.unsafe_score >= threshold
    else:
        parsed_payload = None
        unsafe_label = False
    return {
        "sample_id": sample.id,
        "row_index": row_index,
        "raw_output": raw_output,
        "parsed": parsed_payload,
        "unsafe_label": unsafe_label,
        "ground_truth": {
            "unsafe_label": bool(sample.label.unsafe),
            "categories": list(sample.category_labels),
        },
        "latency_ms": round(latency_ms, 3),
        "error": error,
        "timestamp": _utc_now(),
    }


def _run_one_dataset(
    *,
    config: ResolvedRunConfig,
    selection: ResolvedDatasetSelection,
    guard: Guard,
    policy: Policy | None,
    output_format: OutputFormat | None,
    backend: GenerationBackend,
    root: Path,
    config_hash: str,
    recompute_metrics_only: bool,
) -> tuple[dict[str, Any], int, int]:
    """Execute one dataset; returns (metrics, completed, failed)."""
    dir_name = selection.name.replace("/", "__")
    dataset_dir = root / "datasets" / dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = dataset_dir / "predictions.jsonl"

    dataset_cls = dataset_registry.get(selection.adapter)
    dataset_adapter = dataset_cls.from_config(
        _resolve_dataset_config(selection)
    )
    all_samples = dataset_adapter.load()
    if not all_samples:
        raise ValueError(
            f"dataset {selection.name!r} loaded zero samples"
        )
    samples = _select_samples(all_samples, selection)
    if not samples:
        raise ValueError(
            f"dataset {selection.name!r}: subset selection produced "
            "zero samples"
        )

    _ensure_dataset_manifest(
        dataset_dir,
        selection,
        samples,
        config_hash=config_hash,
    )

    existing_records = _load_existing_records(predictions_path)
    completed_ids = {
        record["sample_id"]
        for record in existing_records
        if record.get("error") is None
    }

    if recompute_metrics_only:
        new_records: list[dict[str, Any]] = []
    else:
        new_records = []
        with _PredictionsWriter(predictions_path) as writer:
            for row_index, sample in enumerate(samples):
                if sample.id in completed_ids:
                    continue
                predict_sample = sample.to_predict_sample()
                messages = guard.build_messages(
                    predict_sample,
                    policy=policy,
                    output_format=output_format,
                )
                started = perf_counter()
                raw_output: Any = None
                parsed_label = None
                error_message: str | None = None
                try:
                    raw_output = _invoke_backend(guard, backend, messages)
                    parsed_label = guard.parse(raw_output)
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    _log.warning(
                        "dataset %s sample %s failed: %s",
                        selection.name,
                        sample.id,
                        error_message,
                    )
                latency_ms = (perf_counter() - started) * 1000.0
                record = _build_record(
                    sample=sample,
                    row_index=row_index,
                    raw_output=raw_output,
                    parsed_label=parsed_label,
                    threshold=config.threshold,
                    latency_ms=latency_ms,
                    error=error_message,
                )
                writer.append(record)
                new_records.append(record)

    all_records = existing_records + new_records
    successful = [
        rec for rec in all_records if rec.get("error") is None
    ]
    failed = [rec for rec in all_records if rec.get("error") is not None]

    predictions = [
        _record_to_normalized_prediction(
            rec, threshold=config.threshold
        )
        for rec in successful
    ]
    predictions = [p for p in predictions if p is not None]
    sample_by_id = {s.id: s for s in samples}
    evaluated_samples = [
        sample_by_id[p.sample_id]
        for p in predictions
        if p.sample_id in sample_by_id
    ]
    if not evaluated_samples:
        raise ValueError(
            f"dataset {selection.name!r}: zero successful predictions"
        )
    metrics = compute_binary_metrics(evaluated_samples, predictions)
    metrics["total_dataset_samples"] = len(samples)
    metrics["evaluated_sample_count"] = len(evaluated_samples)
    metrics["failed_sample_count"] = len(failed)
    _dump_json(dataset_dir / "metrics.json", metrics)
    return metrics, len(successful), len(failed)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _validate_resume_state(
    *,
    root: Path,
    config_hash: str,
    overwrite: bool,
) -> dict[str, Any] | None:
    """Read manifest.json (if any) and enforce the config-hash invariant."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_hash = prior.get("config_hash")
    if prior_hash != config_hash and not overwrite:
        raise ValueError(
            "cannot resume run: existing manifest.json was produced "
            f"by a different config (config_hash {prior_hash!r} vs "
            f"{config_hash!r}). Re-run with output.overwrite=true to "
            "discard the prior run."
        )
    return prior


def _clear_run_dir(root: Path) -> None:
    """Remove all artifacts in the run directory (overwrite path)."""
    import shutil

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _write_manifest(
    *,
    root: Path,
    config: ResolvedRunConfig,
    config_hash: str,
    started_at: str,
    finished_at: str,
    status: str,
    dataset_metrics: dict[str, dict[str, Any]],
    completed_predictions: int,
    failed_predictions: int,
) -> None:
    """Write manifest.json with config snapshot + run state + hash."""
    manifest = {
        "tool_version": __version__,
        "version": 2,
        "run_name": config.run_name,
        "run_dir": root.as_posix(),
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_hash": config_hash,
        "threshold": config.threshold,
        "config": _sanitize_payload_for_artifacts(
            config.model_dump(mode="json")
        ),
        "datasets": [
            {
                "name": selection.name,
                "adapter": selection.adapter,
                "split": selection.split,
                "metrics": dataset_metrics.get(selection.name, {}),
            }
            for selection in config.datasets
        ],
        "totals": {
            "completed_predictions": completed_predictions,
            "failed_predictions": failed_predictions,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "metadata": dict(config.metadata),
    }
    _dump_json(root / "manifest.json", manifest)


def _write_summary(
    root: Path,
    dataset_metrics: dict[str, dict[str, Any]],
) -> None:
    """Write a compact summary.json aggregating per-dataset metrics."""
    summary = {
        "datasets": [
            {
                "name": name,
                "metrics": metrics,
            }
            for name, metrics in dataset_metrics.items()
        ],
    }
    _dump_json(root / "summary.json", summary)


def _execute_run(
    *,
    config: ResolvedRunConfig,
    guard: Guard,
    policy: Policy | None,
    output_format: OutputFormat | None,
    backend: GenerationBackend,
    recompute_metrics_only: bool,
) -> RunResult:
    """Shared execution body for run_from_config and run_benchmark."""
    root = Path(config.output.run_dir)
    if config.output.overwrite:
        _clear_run_dir(root)
    root = _ensure_run_layout(root)

    config_hash = _config_hash(config)
    _validate_resume_state(
        root=root,
        config_hash=config_hash,
        overwrite=config.output.overwrite,
    )

    started_at = _utc_now()
    dataset_metrics: dict[str, dict[str, Any]] = {}
    total_completed = 0
    total_failed = 0

    for selection in config.datasets:
        metrics, completed, failed = _run_one_dataset(
            config=config,
            selection=selection,
            guard=guard,
            policy=policy,
            output_format=output_format,
            backend=backend,
            root=root,
            config_hash=config_hash,
            recompute_metrics_only=recompute_metrics_only,
        )
        dataset_metrics[selection.name] = metrics
        total_completed += completed
        total_failed += failed

    finished_at = _utc_now()
    status = "partial" if total_failed > 0 else "completed"
    _write_manifest(
        root=root,
        config=config,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        dataset_metrics=dataset_metrics,
        completed_predictions=total_completed,
        failed_predictions=total_failed,
    )
    _write_summary(root, dataset_metrics)

    return RunResult(
        run_dir=root.as_posix(),
        manifest_path=(root / "manifest.json").as_posix(),
        summary_path=(root / "summary.json").as_posix(),
        completed_predictions=total_completed,
        failed_predictions=total_failed,
    )


__all__ = [
    "RunResult",
    "run_benchmark",
    "run_from_config",
]
