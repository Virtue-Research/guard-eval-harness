"""Execution runner and batch orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
import json
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from typing import Any

from guard_eval_harness import __version__
from guard_eval_harness.config.models import (
    ResolvedModelConfig,
    ResolvedRunConfig,
)
from guard_eval_harness.judgment import partial_evaluation_judgment
from guard_eval_harness.execution.artifacts import (
    build_resume_signature_payload,
    dump_json,
    dump_jsonl,
    dump_model,
    ensure_run_layout,
    sanitize_payload_for_artifacts,
    sha256_payload,
)
from guard_eval_harness.metrics import compute_binary_metrics
from guard_eval_harness.metrics.binary import (
    validate_binary_prediction_set,
    validate_binary_prediction_set_partial,
)
from guard_eval_harness.metrics.code_vuln import (
    compute_code_vuln_metrics,
)
from guard_eval_harness.registry import ensure_builtin_registrations
from guard_eval_harness.registry import dataset_registry, model_registry
from guard_eval_harness.reports import build_summary, write_html_report
from guard_eval_harness.schemas import RunEnvironment, RunManifest
from guard_eval_harness.schemas import (
    NormalizedPrediction,
    NormalizedSample,
    PredictSample,
)

_log = logging.getLogger(__name__)

# Drop rate thresholds for warnings and errors
_DROP_RATE_WARN = 0.05   # 5% — warn loudly
_DROP_RATE_ERROR = 0.20  # 20% — flag results as unreliable

_AUTO_BATCH_REMOTE_ADAPTERS = {
    "anthropic",
    "http",
    "openai_compatible",
    "openai_moderation",
}
_EXECUTION_DEFAULT_MODEL_ARGS = (
    "concurrency",
    "retries",
    "retry_backoff",
)
_AUTO_BATCH_RETRYABLE_ERRORS = (
    "cuda out of memory",
    "insufficient memory",
    "no available memory",
    "out of memory",
    "resource exhausted",
)


def _effective_model_name(model: ResolvedModelConfig) -> str | None:
    """Resolve the configured runtime model name, if any."""
    model_name = model.model_name
    if model_name:
        return str(model_name)
    for key in ("model", "pretrained"):
        value = model.args.get(key)
        if value:
            return str(value)
    return None


@dataclass(slots=True)
class RunResult:
    """Result returned by the runner."""

    run_dir: str
    manifest_path: str
    summary_path: str


ProgressCallback = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    """Return a UTC timestamp string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload: Any,
) -> None:
    """Invoke one optional progress callback with a JSON-safe payload."""
    if progress_callback is None:
        return
    progress_callback(dict(payload))


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object when it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_runtime_config(config: ResolvedRunConfig) -> ResolvedRunConfig:
    """Inject shared runtime defaults into model and dataset config."""
    model = config.model
    if model.adapter in _AUTO_BATCH_REMOTE_ADAPTERS:
        args = dict(model.args)
        defaults = {
            "concurrency": config.execution.concurrency,
            "retries": config.execution.retries,
            "retry_backoff": config.execution.retry_backoff,
        }
        for key, value in defaults.items():
            args.setdefault(key, value)
        model = model.model_copy(update={"args": args})

    runtime_model_name = _effective_model_name(model)
    datasets = []
    for dataset in config.datasets:
        options = dict(dataset.options)
        if runtime_model_name is not None:
            options.setdefault("model_name", runtime_model_name)
        datasets.append(dataset.model_copy(update={"options": options}))

    return config.model_copy(
        update={
            "model": model,
            "datasets": datasets,
        }
    )


def _find_dataset_payload(
    resolved_config: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    """Locate one dataset config payload by name."""
    for dataset in resolved_config.get("datasets", []):
        if dataset.get("name") == dataset_name:
            return dict(dataset)
    raise ValueError(f"resume signature missing dataset {dataset_name!r}")


def _resume_signature(
    resolved_config: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    """Build a stable resume signature for one dataset run."""
    model = dict(resolved_config.get("model", {}))
    model_args = dict(model.get("args", {}))
    if model.get("adapter") in _AUTO_BATCH_REMOTE_ADAPTERS:
        for key in _EXECUTION_DEFAULT_MODEL_ARGS:
            model_args.pop(key, None)
    model["args"] = model_args
    return {
        "threshold": resolved_config.get("threshold"),
        "limit": resolved_config.get("execution", {}).get("limit"),
        "model": model,
        "dataset": _find_dataset_payload(resolved_config, dataset_name),
    }


def _validate_resume_config(
    *,
    prior_resume_signature: dict[str, Any] | None,
    prior_resolved_config: dict[str, Any] | None,
    current_resume_signature: dict[str, Any],
    current_resolved_config: dict[str, Any],
    dataset_name: str,
) -> None:
    """Reject cached predictions from a mismatched run configuration."""
    if prior_resume_signature is None and prior_resolved_config is None:
        raise ValueError(
            f"cannot resume dataset {dataset_name}: resolved-config.json "
            "is missing from the existing run directory"
        )

    if prior_resume_signature is not None:
        previous = _resume_signature(prior_resume_signature, dataset_name)
        current = _resume_signature(current_resume_signature, dataset_name)
    else:
        previous = _resume_signature(prior_resolved_config, dataset_name)
        current = _resume_signature(current_resolved_config, dataset_name)
    if previous != current:
        raise ValueError(
            f"cannot resume dataset {dataset_name}: existing artifacts do not "
            "match the current model, threshold, dataset, or limit"
        )


def _preflight_resume_validation(
    *,
    root: Path,
    runtime_config: ResolvedRunConfig,
    prior_resume_signature: dict[str, Any] | None,
    prior_resolved_config: dict[str, Any] | None,
    current_resume_signature: dict[str, Any],
    current_resolved_config: dict[str, Any],
) -> None:
    """Validate resumable dataset artifacts before mutating run metadata."""
    if not runtime_config.execution.resume:
        return

    for dataset_config in runtime_config.datasets:
        dir_name = dataset_config.name.replace("/", "__")
        predictions_path = root / "datasets" / dir_name / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        _validate_resume_config(
            prior_resume_signature=prior_resume_signature,
            prior_resolved_config=prior_resolved_config,
            current_resume_signature=current_resume_signature,
            current_resolved_config=current_resolved_config,
            dataset_name=dataset_config.name,
        )


def _dataset_resume_signature(
    *,
    resume_signature_payload: dict[str, Any],
    dataset_name: str,
) -> str:
    """Return the stable resume signature hash for one dataset."""
    return sha256_payload(
        _resume_signature(
            resume_signature_payload,
            dataset_name,
        )
    )


def _load_cached_predictions(path: Path) -> list[NormalizedPrediction]:
    """Load canonical predictions from a prior predictions artifact."""
    if not path.exists():
        return []

    predictions: list[NormalizedPrediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            predictions.append(
                NormalizedPrediction.model_validate(json.loads(stripped))
            )
    return predictions


def _merged_predictions(
    samples: Sequence[NormalizedSample],
    cached_predictions: Sequence[NormalizedPrediction],
    new_predictions: Sequence[NormalizedPrediction],
) -> list[NormalizedPrediction]:
    """Merge cached and fresh predictions in sample order."""
    grouped_predictions: dict[str, list[NormalizedPrediction]] = {}
    for prediction in [*cached_predictions, *new_predictions]:
        grouped_predictions.setdefault(
            prediction.sample_id,
            [],
        ).append(prediction)

    merged: list[NormalizedPrediction] = []
    for sample in samples:
        merged.extend(grouped_predictions.pop(sample.id, []))
    for leftover_predictions in grouped_predictions.values():
        merged.extend(leftover_predictions)
    return merged


def _evaluated_samples(
    dataset_name: str,
    samples: Sequence[NormalizedSample],
    predictions: Sequence[NormalizedPrediction],
    *,
    allow_partial_predictions: bool,
) -> tuple[list[NormalizedSample], list[str]]:
    """Validate one dataset result and return evaluated rows."""
    if allow_partial_predictions:
        missing_prediction_ids = validate_binary_prediction_set_partial(
            samples,
            predictions,
        )
    else:
        missing_prediction_ids = validate_binary_prediction_set(
            samples,
            predictions,
        )
    if not missing_prediction_ids:
        return list(samples), []

    prediction_ids = {prediction.sample_id for prediction in predictions}
    evaluated_samples = [
        sample for sample in samples if sample.id in prediction_ids
    ]
    if not evaluated_samples:
        raise ValueError(
            f"dataset {dataset_name} produced no successful predictions"
        )
    return evaluated_samples, missing_prediction_ids


def _prepare_batch(
    model: Any,
    batch: Sequence[NormalizedSample],
) -> Sequence[NormalizedSample] | list[PredictSample]:
    """Strip ground-truth labels before passing samples to the model.

    Adapters that declare ``requires_ground_truth=True`` (only the mock today)
    receive the full ``NormalizedSample``; every production adapter sees a
    label-free ``PredictSample`` view.
    """
    if model.capabilities.requires_ground_truth:
        return batch
    return [sample.to_predict_sample() for sample in batch]


def _predict_in_fixed_batches(
    model: Any,
    samples: Sequence[NormalizedSample],
    *,
    threshold: float,
    batch_size: int,
    dataset_name: str,
    progress_callback: ProgressCallback | None,
) -> list[NormalizedPrediction]:
    """Run prediction batches with a fixed batch size."""
    predictions: list[NormalizedPrediction] = []
    total_batches = (len(samples) + batch_size - 1) // batch_size
    progress_interval = max(1, total_batches // 10)
    started_at = perf_counter()
    pbar = tqdm(
        total=len(samples),
        desc=dataset_name,
        unit="sample",
    )
    for batch_index, start in enumerate(
        range(0, len(samples), batch_size),
        start=1,
    ):
        batch = samples[start : start + batch_size]
        predictions.extend(
            model.predict_batch(_prepare_batch(model, batch), threshold=threshold)
        )
        pbar.update(len(batch))
        if (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_interval == 0
        ):
            _emit_progress(
                progress_callback,
                event="prediction_progress",
                dataset_name=dataset_name,
                completed_batches=batch_index,
                total_batches=total_batches,
                completed_samples=min(start + len(batch), len(samples)),
                total_samples=len(samples),
                inference_seconds=round(perf_counter() - started_at, 3),
            )
    pbar.close()
    return predictions


def _is_retryable_auto_batch_error(exc: Exception) -> bool:
    """Identify retryable local batching failures."""
    message = str(exc).lower()
    return any(token in message for token in _AUTO_BATCH_RETRYABLE_ERRORS)


def _predict_with_auto_batch_size(
    model: Any,
    samples: Sequence[NormalizedSample],
    *,
    threshold: float,
    dataset_name: str,
    progress_callback: ProgressCallback | None,
) -> list[NormalizedPrediction]:
    """Run prediction batches with adaptive backoff on capacity failures."""
    if not samples:
        return []

    adapter_name = getattr(model.config, "adapter", "")
    if adapter_name in _AUTO_BATCH_REMOTE_ADAPTERS:
        predictions = list(
            model.predict_batch(
                _prepare_batch(model, samples), threshold=threshold
            )
        )
        _emit_progress(
            progress_callback,
            event="prediction_progress",
            dataset_name=dataset_name,
            completed_batches=1,
            total_batches=1,
            completed_samples=len(samples),
            total_samples=len(samples),
            inference_seconds=0.0,
        )
        return predictions

    predictions: list[NormalizedPrediction] = []
    start = 0
    current_batch_size = len(samples)
    started_at = perf_counter()
    completed_batches = 0
    while start < len(samples):
        batch_size = min(current_batch_size, len(samples) - start)
        batch = samples[start : start + batch_size]
        try:
            predictions.extend(
                model.predict_batch(
                    _prepare_batch(model, batch), threshold=threshold
                )
            )
        except Exception as exc:
            if batch_size == 1 or not _is_retryable_auto_batch_error(exc):
                raise
            current_batch_size = max(1, batch_size // 2)
            continue
        start += batch_size
        completed_batches += 1
        _emit_progress(
            progress_callback,
            event="prediction_progress",
            dataset_name=dataset_name,
            completed_batches=completed_batches,
            total_batches=None,
            completed_samples=start,
            total_samples=len(samples),
            inference_seconds=round(perf_counter() - started_at, 3),
        )
    return predictions


def _execute_predictions(
    model: Any,
    samples: Sequence[NormalizedSample],
    *,
    threshold: float,
    batch_size: int | str,
    dataset_name: str,
    progress_callback: ProgressCallback | None,
) -> list[NormalizedPrediction]:
    """Run pending samples through the configured batching strategy."""
    if batch_size == "auto":
        return _predict_with_auto_batch_size(
            model,
            samples,
            threshold=threshold,
            dataset_name=dataset_name,
            progress_callback=progress_callback,
        )
    return _predict_in_fixed_batches(
        model,
        samples,
        threshold=threshold,
        batch_size=batch_size,
        dataset_name=dataset_name,
        progress_callback=progress_callback,
    )


def run_benchmark(
    config: ResolvedRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> RunResult:
    """Execute a benchmark run and write disk artifacts."""
    ensure_builtin_registrations()
    runtime_config = _resolve_runtime_config(config)
    root = Path(runtime_config.output.run_dir)
    allow_existing_run = (
        runtime_config.output.overwrite or runtime_config.execution.resume
    )
    if root.exists() and not allow_existing_run and any(root.iterdir()):
        raise FileExistsError(
            f"run directory already exists and is not empty: {root}"
        )
    root = ensure_run_layout(root)

    started_at = _utc_now()
    runtime_config_payload = runtime_config.model_dump(mode="json")
    resume_signature_payload = build_resume_signature_payload(
        runtime_config_payload
    )
    resolved_config_payload = sanitize_payload_for_artifacts(
        runtime_config_payload
    )
    prior_resume_signature = _load_json(root / "resume-signature.json")
    prior_resolved_config = _load_json(root / "resolved-config.json")
    _preflight_resume_validation(
        root=root,
        runtime_config=runtime_config,
        prior_resume_signature=prior_resume_signature,
        prior_resolved_config=prior_resolved_config,
        current_resume_signature=resume_signature_payload,
        current_resolved_config=resolved_config_payload,
    )
    if prior_resolved_config is None or runtime_config.output.overwrite:
        dump_json(root / "resolved-config.json", resolved_config_payload)
    if prior_resume_signature is None or runtime_config.output.overwrite:
        dump_json(root / "resume-signature.json", resume_signature_payload)

    model_cls = model_registry.get(runtime_config.model.adapter)
    model = model_cls.from_config(runtime_config.model)

    dataset_results = []
    dataset_metadata = []
    written_dataset_dirs: set[str] = set()
    run_warnings = list(runtime_config.warnings)
    partial_run = False

    dataset_count = len(runtime_config.datasets)
    for dataset_index, dataset_config in enumerate(
        runtime_config.datasets,
        start=1,
    ):
        dataset_cls = dataset_registry.get(dataset_config.adapter)
        dataset = dataset_cls.from_config(dataset_config)
        dataset_started_at = perf_counter()
        _emit_progress(
            progress_callback,
            event="dataset_load_started",
            dataset_index=dataset_index,
            dataset_count=dataset_count,
            dataset_name=dataset_config.name,
            dataset_adapter=dataset_config.adapter,
            split=dataset_config.split,
        )
        samples = dataset.load()
        load_seconds = perf_counter() - dataset_started_at
        _emit_progress(
            progress_callback,
            event="dataset_load_completed",
            dataset_index=dataset_index,
            dataset_count=dataset_count,
            dataset_name=dataset_config.name,
            dataset_adapter=dataset_config.adapter,
            split=dataset_config.split,
            sample_count=len(samples),
            load_seconds=round(load_seconds, 3),
        )
        if runtime_config.execution.limit is not None:
            samples = samples[: runtime_config.execution.limit]
        if not samples:
            raise ValueError(
                f"dataset {dataset_config.name} loaded zero samples"
            )

        dir_name = dataset_config.name.replace("/", "__")
        dataset_dir = root / "datasets" / dir_name
        cached_predictions: list[NormalizedPrediction] = []
        predictions_path = dataset_dir / "predictions.jsonl"
        dataset_manifest_path = dataset_dir / "dataset-manifest.json"
        if runtime_config.execution.resume and predictions_path.exists():
            dataset_manifest_payload = _load_json(dataset_manifest_path)
            if dataset_manifest_payload is None:
                raise ValueError(
                    f"cannot resume dataset {dataset_config.name}: "
                    "dataset-manifest.json is missing"
                )
            prior_dataset_signature = dataset_manifest_payload.get(
                "metadata", {}
            ).get("resume_signature_sha256")
            current_dataset_signature = _dataset_resume_signature(
                resume_signature_payload=resume_signature_payload,
                dataset_name=dataset_config.name,
            )
            if prior_dataset_signature != current_dataset_signature:
                raise ValueError(
                    f"cannot resume dataset {dataset_config.name}: "
                    "cached predictions were produced by a different "
                    "dataset/model signature"
                )
            cached_predictions = _load_cached_predictions(predictions_path)
            validate_binary_prediction_set_partial(
                samples,
                cached_predictions,
            )

        cached_prediction_ids = {
            prediction.sample_id for prediction in cached_predictions
        }
        pending_samples = [
            sample
            for sample in samples
            if sample.id not in cached_prediction_ids
        ]
        _emit_progress(
            progress_callback,
            event="prediction_started",
            dataset_name=dataset_config.name,
            sample_count=len(pending_samples),
            batch_size=runtime_config.execution.batch_size,
            total_batches=(
                None
                if runtime_config.execution.batch_size == "auto"
                else (
                    (
                        len(pending_samples)
                        + runtime_config.execution.batch_size
                        - 1
                    )
                    // runtime_config.execution.batch_size
                )
            ),
        )
        inference_started_at = perf_counter()
        new_predictions = _execute_predictions(
            model,
            pending_samples,
            threshold=runtime_config.threshold,
            batch_size=runtime_config.execution.batch_size,
            dataset_name=dataset_config.name,
            progress_callback=progress_callback,
        )
        _emit_progress(
            progress_callback,
            event="prediction_completed",
            dataset_name=dataset_config.name,
            prediction_count=len(new_predictions),
            inference_seconds=round(perf_counter() - inference_started_at, 3),
        )
        predictions = _merged_predictions(
            samples,
            cached_predictions,
            new_predictions,
        )

        evaluated_samples, dropped_sample_ids = _evaluated_samples(
            dataset_config.name,
            samples,
            predictions,
            allow_partial_predictions=model.allow_partial_predictions,
        )
        metadata = dataset.describe(samples)

        try:
            metrics = compute_binary_metrics(
                evaluated_samples,
                predictions,
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid prediction set for dataset {dataset_config.name}: "
                f"{exc}"
            ) from exc

        if metadata.metric_eligibility.get("code_vuln", False):
            metrics["code_vuln"] = compute_code_vuln_metrics(
                evaluated_samples,
                predictions,
            )
            if dropped_sample_ids:
                metrics["code_vuln"]["dropped_sample_count"] = len(
                    dropped_sample_ids
                )
                metrics["code_vuln"]["total_dataset_samples"] = len(samples)
        metadata_payload = dict(metadata.metadata)
        metadata_payload["evaluated_sample_count"] = len(
            evaluated_samples,
        )
        metadata_payload["cached_prediction_count"] = len(
            cached_predictions,
        )
        metadata_payload["executed_sample_count"] = len(
            pending_samples,
        )
        metadata_payload["resume_enabled"] = runtime_config.execution.resume
        metadata_payload["resume_signature_sha256"] = _dataset_resume_signature(
            resume_signature_payload=resume_signature_payload,
            dataset_name=dataset_config.name,
        )
        if dropped_sample_ids:
            partial_run = True
            dropped_count = len(dropped_sample_ids)
            total_count = len(samples)
            drop_rate = dropped_count / total_count if total_count else 0
            evaluation_judgment = partial_evaluation_judgment(
                dropped_count=dropped_count,
                total_count=total_count,
            )
            warning = (
                f"dropped {dropped_count} of {total_count} samples "
                f"({drop_rate:.2%}) from dataset "
                f"{dataset_config.name} after adapter errors"
            )
            run_warnings.append(warning)
            if drop_rate >= _DROP_RATE_ERROR:
                _log.error(
                    "WARNING:HIGH DROP RATE: %s — %.1f%% of samples dropped "
                    "(%d/%d). Results for this dataset are UNRELIABLE. "
                    "This is likely caused by API rate limiting (429) "
                    "or request errors. Consider re-running with lower "
                    "concurrency.",
                    dataset_config.name,
                    drop_rate * 100,
                    dropped_count,
                    total_count,
                )
            elif drop_rate >= _DROP_RATE_WARN:
                _log.warning(
                    "WARNING:%s: %.1f%% of samples dropped (%d/%d). "
                    "Results may be affected.",
                    dataset_config.name,
                    drop_rate * 100,
                    dropped_count,
                    total_count,
                )
            metadata_payload["dropped_sample_count"] = dropped_count
            metadata_payload["drop_rate"] = drop_rate
            metadata_payload["evaluation_judgment"] = evaluation_judgment
            dropped_preview = dropped_sample_ids[:25]
            metadata_payload["dropped_sample_ids"] = dropped_preview
            if len(dropped_sample_ids) > len(dropped_preview):
                metadata_payload["dropped_sample_ids_truncated"] = True
        metadata = metadata.model_copy(update={"metadata": metadata_payload})
        dataset_metadata.append(metadata)
        dataset_results.append(
            {
                "metadata": metadata.model_dump(mode="json"),
                "metrics": metrics,
            }
        )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        written_dataset_dirs.add(dir_name)
        dump_jsonl(
            dataset_dir / "predictions.jsonl",
            [prediction.model_dump(mode="json") for prediction in predictions],
        )
        dump_json(dataset_dir / "metrics.json", metrics)
        dump_model(dataset_dir / "dataset-manifest.json", metadata)
        _emit_progress(
            progress_callback,
            event="dataset_completed",
            dataset_name=dataset_config.name,
            total_sample_count=len(samples),
            evaluated_sample_count=len(evaluated_samples),
            accuracy=metrics.get("accuracy"),
            auroc=metrics.get("auroc"),
            auprc=metrics.get("auprc"),
            dataset_seconds=round(perf_counter() - dataset_started_at, 3),
        )

    if runtime_config.output.overwrite or runtime_config.execution.resume:
        datasets_dir = root / "datasets"
        if datasets_dir.is_dir():
            written_lower = {name.casefold() for name in written_dataset_dirs}
            for child in datasets_dir.iterdir():
                if (
                    child.is_dir()
                    and child.name.casefold() not in written_lower
                ):
                    shutil.rmtree(child)

    finished_at = _utc_now()
    manifest = RunManifest(
        tool_version=__version__,
        run_name=runtime_config.run_name,
        run_dir=root.as_posix(),
        status="partial" if partial_run else "completed",
        started_at=started_at,
        finished_at=finished_at,
        resolved_config_sha256=sha256_payload(resolved_config_payload),
        model=runtime_config.model,
        execution=runtime_config.execution,
        output=runtime_config.output,
        threshold=runtime_config.threshold,
        datasets=dataset_metadata,
        adapter_capabilities=model.capabilities,
        environment=RunEnvironment(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            hostname=socket.gethostname(),
        ),
        warnings=run_warnings,
        metadata=runtime_config.metadata,
    )

    summary = build_summary(
        manifest.model_dump(mode="json"),
        dataset_results,
    )
    sanitized_manifest = sanitize_payload_for_artifacts(
        manifest.model_dump(mode="json")
    )
    dump_json(root / "resolved-config.json", resolved_config_payload)
    dump_json(root / "resume-signature.json", resume_signature_payload)
    dump_json(root / "manifest.json", sanitized_manifest)
    dump_json(root / "summary.json", summary)
    write_html_report(root, summary)

    return RunResult(
        run_dir=root.as_posix(),
        manifest_path=(root / "manifest.json").as_posix(),
        summary_path=(root / "summary.json").as_posix(),
    )
