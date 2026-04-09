"""Reusable benchmark preset definitions and runners."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from guard_eval_harness.config import load_config
from guard_eval_harness.execution import run_benchmark
from guard_eval_harness.models.catalog import ModelProfile
from guard_eval_harness.models.templates import MD_JUDGE_PROMPT

DATASET_SCOPE_ALL = "all"
DATASET_SCOPE_CORE12 = "core12"
DATASET_SCOPE_NON_CORE12 = "non_core12"
DatasetScope = Literal["all", "core12", "non_core12"]


@dataclass(frozen=True, slots=True)
class PresetDataset:
    """One dataset entry inside a benchmark preset."""

    name: str
    adapter: str
    split: str = "test"
    options: dict[str, Any] = field(default_factory=dict)
    is_core12: bool = False

    def as_config(self) -> dict[str, Any]:
        """Render the dataset as a run-config payload fragment."""
        payload: dict[str, Any] = {
            "name": self.name,
            "adapter": self.adapter,
        }
        if self.split != "test":
            payload["split"] = self.split
        if self.options:
            payload["options"] = dict(self.options)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkPreset:
    """A reproducible benchmark suite definition."""

    name: str
    description: str
    default_output_root: str
    datasets: tuple[PresetDataset, ...]


@dataclass(frozen=True, slots=True)
class PresetRunResult:
    """One dataset-level result from a preset run."""

    dataset: str
    status: str
    run_dir: str
    elapsed_seconds: float
    summary_path: str = ""
    sample_count: int | None = None
    accuracy: float | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Render the result into a JSON-friendly mapping."""
        return asdict(self)


_PRESET_21X31 = BenchmarkPreset(
    name="21x31",
    description=(
        "31-dataset guard-model benchmark suite used for the 21-model "
        "reproduction wave."
    ),
    default_output_root="out/presets/21x31",
    datasets=(
        PresetDataset(
            name="advbench_behaviors",
            adapter="advbench_behaviors",
            is_core12=True,
        ),
        PresetDataset(
            name="advbench_strings",
            adapter="advbench_strings",
        ),
        PresetDataset(
            name="beaver_tails_330k",
            adapter="beaver_tails_330k",
            is_core12=True,
        ),
        PresetDataset(
            name="cat_qa",
            adapter="cat_qa",
            split="en",
        ),
        PresetDataset(
            name="civil_comments",
            adapter="civil_comments",
            split="validation",
        ),
        PresetDataset(
            name="convabuse",
            adapter="convabuse",
            is_core12=True,
        ),
        PresetDataset(
            name="do_anything_now_questions",
            adapter="do_anything_now_questions",
        ),
        PresetDataset(
            name="do_not_answer",
            adapter="do_not_answer",
            is_core12=True,
        ),
        PresetDataset(
            name="harmful_q",
            adapter="harmful_q",
        ),
        PresetDataset(
            name="harmful_qa",
            adapter="harmful_qa",
            is_core12=True,
        ),
        PresetDataset(
            name="harmful_qa_questions",
            adapter="harmful_qa_questions",
            split="train",
        ),
        PresetDataset(
            name="harmbench_behaviors",
            adapter="harmbench_behaviors",
            is_core12=True,
        ),
        PresetDataset(
            name="hatecheck",
            adapter="hatecheck",
        ),
        PresetDataset(
            name="hatemoji_check",
            adapter="hatemoji_check",
            is_core12=True,
        ),
        PresetDataset(
            name="i_cona",
            adapter="i_cona",
        ),
        PresetDataset(
            name="i_controversial",
            adapter="i_controversial",
        ),
        PresetDataset(
            name="i_malicious_instructions",
            adapter="i_malicious_instructions",
        ),
        PresetDataset(
            name="i_physical_safety",
            adapter="i_physical_safety",
        ),
        PresetDataset(
            name="jbb_behaviors",
            adapter="jbb_behaviors",
            is_core12=True,
        ),
        PresetDataset(
            name="malicious_instruct",
            adapter="malicious_instruct",
        ),
        PresetDataset(
            name="mitre",
            adapter="mitre",
        ),
        PresetDataset(
            name="niche_hazard_qa",
            adapter="niche_hazard_qa",
            split="train",
        ),
        PresetDataset(
            name="aegis_ai_content_safety_dataset_2",
            adapter="aegis_ai_content_safety_dataset_2",
            split="validation",
            options={"task": "prompt_harm"},
        ),
        PresetDataset(
            name="pku_safe_rlhf",
            adapter="pku_safe_rlhf",
            is_core12=True,
        ),
        PresetDataset(
            name="real_toxicity_prompts",
            adapter="real_toxicity_prompts",
            split="train",
        ),
        PresetDataset(
            name="safe_text",
            adapter="safe_text",
        ),
        PresetDataset(
            name="simple_safety_tests",
            adapter="simple_safety_tests",
        ),
        PresetDataset(
            name="strong_reject_instructions",
            adapter="strong_reject_instructions",
            is_core12=True,
        ),
        PresetDataset(
            name="tdc_red_teaming",
            adapter="tdc_red_teaming",
        ),
        PresetDataset(
            name="toxic_chat",
            adapter="toxic_chat",
            is_core12=True,
        ),
        PresetDataset(
            name="xstest",
            adapter="xstest",
            is_core12=True,
        ),
    ),
)

_PRESETS = {
    "21x31": _PRESET_21X31,
    "benchmark-21x31": _PRESET_21X31,
}


def _slugify(value: str) -> str:
    """Convert a display name into a stable filesystem slug."""
    slug_chars = [
        character.lower() if character.isalnum() else "-" for character in value
    ]
    slug = "".join(slug_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _api_profile(
    *,
    name: str,
    model_name: str,
    max_tokens: int,
) -> ModelProfile:
    """Build an OpenRouter-backed API model profile."""
    return ModelProfile(
        name=name,
        adapter="openai_compatible",
        model_name=model_name,
        catalog_name=name,
        args={
            "root_url": "https://openrouter.ai/api",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "concurrency": 4,
            "retries": 2,
            "retry_backoff": 1.0,
            "prompt_template": MD_JUDGE_PROMPT,
        },
    )


def _preset_model_profiles() -> dict[str, ModelProfile]:
    """Return package-owned model profiles for preset execution."""
    return {
        "GPT-4o-mini": _api_profile(
            name="GPT-4o-mini",
            model_name="openai/gpt-4o-mini",
            max_tokens=8,
        ),
        "GPT-4.1-mini": _api_profile(
            name="GPT-4.1-mini",
            model_name="openai/gpt-4.1-mini",
            max_tokens=16,
        ),
    }


def _summary_metrics(
    summary_path: Path,
) -> tuple[int | None, float | None]:
    """Read sample count and accuracy from a run summary."""
    if not summary_path.exists():
        return None, None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    datasets = summary.get("datasets", [])
    if not datasets:
        return None, None
    metrics = datasets[0].get("metrics", {})
    count = metrics.get("count")
    accuracy = metrics.get("accuracy")
    normalized_count = count if isinstance(count, int) else None
    normalized_accuracy = (
        float(accuracy) if isinstance(accuracy, (int, float)) else None
    )
    return normalized_count, normalized_accuracy


def _summary_has_samples(summary_path: Path) -> bool:
    """Return whether an existing summary reflects a real run."""
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "completed":
        return False
    count, _ = _summary_metrics(summary_path)
    return bool(count and count > 0)


def list_presets() -> list[str]:
    """Return the canonical preset names available through the harness."""
    return sorted({preset.name for preset in _PRESETS.values()})


def get_preset(name: str) -> BenchmarkPreset:
    """Return one benchmark preset by name or alias."""
    try:
        return _PRESETS[name]
    except KeyError as exc:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Unknown preset {name!r}. Available: {available}"
        ) from exc


def list_preset_models(preset_name: str) -> list[str]:
    """Return the supported model names for one preset."""
    get_preset(preset_name)
    return sorted(_preset_model_profiles())


def list_preset_datasets(
    preset_name: str,
    *,
    dataset_scope: DatasetScope = DATASET_SCOPE_ALL,
) -> list[str]:
    """Return the dataset names for one preset scope."""
    return [
        dataset.name
        for dataset in _select_datasets(
            preset_name,
            dataset_scope=dataset_scope,
        )
    ]


def resolve_model_config(
    model_name: str,
    *,
    device: int,
) -> dict[str, Any]:
    """Resolve a preset model name into a harness model config."""
    profiles = _preset_model_profiles()
    try:
        profile = profiles[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {available}"
        ) from exc

    args = dict(profile.args)
    if profile.adapter == "hf":
        args["device"] = device
    return {
        "adapter": profile.adapter,
        "model_name": profile.model_name,
        "args": args,
    }


def _select_datasets(
    preset_name: str,
    *,
    dataset_scope: DatasetScope,
) -> list[PresetDataset]:
    """Return the datasets included for a preset scope."""
    preset = get_preset(preset_name)
    if dataset_scope == DATASET_SCOPE_ALL:
        return list(preset.datasets)
    if dataset_scope == DATASET_SCOPE_CORE12:
        return [dataset for dataset in preset.datasets if dataset.is_core12]
    if dataset_scope == DATASET_SCOPE_NON_CORE12:
        return [dataset for dataset in preset.datasets if not dataset.is_core12]
    raise ValueError(f"unsupported dataset scope: {dataset_scope}")


def build_run_payload(
    *,
    preset_name: str,
    model_name: str,
    dataset: PresetDataset,
    device: int,
    batch_size: int,
    output_root: str | Path | None = None,
    concurrency_override: int | None = None,
) -> dict[str, Any]:
    """Build one run payload for one preset/model/dataset combination."""
    preset = get_preset(preset_name)
    model_config = resolve_model_config(model_name, device=device)
    if concurrency_override is not None:
        model_args = dict(model_config.get("args", {}))
        model_args["concurrency"] = concurrency_override
        model_config["args"] = model_args

    run_root = (
        Path(output_root)
        if output_root is not None
        else Path(preset.default_output_root)
    )
    return {
        "version": 1,
        "run_name": f"bench-{preset.name}-{_slugify(model_name)}",
        "threshold": 0.5,
        "model": model_config,
        "datasets": [dataset.as_config()],
        "output": {
            "run_dir": (
                run_root / _slugify(model_name) / dataset.name
            ).as_posix(),
            "overwrite": True,
        },
        "execution": {
            "batch_size": batch_size,
            "concurrency": 1,
            "retries": 0,
        },
        "metadata": {
            "preset": preset.name,
            "preset_dataset_scope": "single-dataset",
        },
    }


def run_preset(
    *,
    preset_name: str,
    model_name: str,
    device: int,
    batch_size: int,
    dataset_scope: DatasetScope = DATASET_SCOPE_ALL,
    output_root: str | Path | None = None,
    concurrency_override: int | None = None,
    skip_existing: bool = True,
    base_dir: str | Path = ".",
    on_result: Callable[[PresetRunResult, int, int], None] | None = None,
) -> list[PresetRunResult]:
    """Run a benchmark preset sequentially and return dataset-level results."""
    results: list[PresetRunResult] = []
    datasets = _select_datasets(
        preset_name,
        dataset_scope=dataset_scope,
    )
    total = len(datasets)

    for index, dataset in enumerate(datasets, start=1):
        payload = build_run_payload(
            preset_name=preset_name,
            model_name=model_name,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            output_root=output_root,
            concurrency_override=concurrency_override,
        )
        started = time.perf_counter()
        try:
            config = load_config(payload, base_dir=base_dir)
        except Exception as exc:
            result = PresetRunResult(
                dataset=dataset.name,
                status="fail",
                run_dir=str(payload["output"]["run_dir"]),
                error=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )
            results.append(result)
            if on_result is not None:
                on_result(result, index, total)
            continue

        run_dir = Path(config.output.run_dir)
        summary_path = run_dir / "summary.json"
        if skip_existing and _summary_has_samples(summary_path):
            sample_count, accuracy = _summary_metrics(summary_path)
            result = PresetRunResult(
                dataset=dataset.name,
                status="skipped",
                run_dir=run_dir.as_posix(),
                summary_path=summary_path.as_posix(),
                sample_count=sample_count,
                accuracy=accuracy,
                elapsed_seconds=0.0,
            )
            results.append(result)
            if on_result is not None:
                on_result(result, index, total)
            continue

        try:
            run_result = run_benchmark(config)
        except Exception as exc:
            result = PresetRunResult(
                dataset=dataset.name,
                status="fail",
                run_dir=run_dir.as_posix(),
                error=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )
            results.append(result)
            if on_result is not None:
                on_result(result, index, total)
            continue

        resolved_summary_path = Path(run_result.summary_path)
        sample_count, accuracy = _summary_metrics(resolved_summary_path)
        result = PresetRunResult(
            dataset=dataset.name,
            status="ok",
            run_dir=run_result.run_dir,
            summary_path=run_result.summary_path,
            sample_count=sample_count,
            accuracy=accuracy,
            elapsed_seconds=time.perf_counter() - started,
        )
        results.append(result)
        if on_result is not None:
            on_result(result, index, total)

    return results


def summarize_results(
    results: list[PresetRunResult],
) -> dict[str, Any]:
    """Summarize preset results into a JSON-friendly payload."""
    counts = {
        "ok": 0,
        "skipped": 0,
        "fail": 0,
        "total": len(results),
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "counts": counts,
        "results": [result.as_dict() for result in results],
    }
