"""Source-backed CircleGuardBench Public dataset adapter."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_REVISION = "501301bd466f59387d8554d43b8790d8a87a7a17"
_SUPPORTED_TASKS = {"prompt_harm", "default_response_harm"}


@dataset_registry.register("circleguardbench_public")
class CircleGuardBenchPublicDataset(SourceBackedDatasetAdapter):
    """Load prompt-harm or default-response views of CircleGuardBench."""

    display_name = "CircleGuardBench Public"
    source_uri = (
        "https://huggingface.co/datasets/whitecircle-ai/"
        "circleguardbench_public"
    )
    license_name = "White Circle Responsible Use License v1.0"
    languages = ("en",)
    metadata_fields_to_preserve = (
        "harm_category",
        "source",
        "intent",
        "behavioral_policies",
        "low_level_harm_cluster_id",
        "prompt_hash",
        "variant",
        "task",
        "target_role",
    )
    label_mapping_note = (
        "CircleGuardBench Public supports prompt_harm and "
        "default_response_harm tasks. prompt_harm emits both original and "
        "jailbreaked user prompts with the source prompt verdict."
    )
    supported_splits = ("train",)

    def _task(self) -> str:
        """Return the configured CircleGuardBench task."""
        task = str(self.config.options.get("task", "prompt_harm")).strip().lower()
        if task not in _SUPPORTED_TASKS:
            supported = ", ".join(sorted(_SUPPORTED_TASKS))
            raise ValueError(
                f"circleguardbench_public supports options.task values: "
                f"{supported}"
            )
        return task

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load and normalize CircleGuardBench rows for the chosen task."""
        task = self._task()
        rows = load_hf_rows(
            "whitecircle-ai/circleguardbench_public",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if task == "prompt_harm":
                normalized_rows.extend(self._prompt_harm_rows(row))
            else:
                normalized = self._default_response_row(row)
                if normalized is not None:
                    normalized_rows.append(normalized)
        return normalized_rows

    def _prompt_harm_rows(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Build prompt-harm samples from the original and jailbreak prompts."""
        unsafe = str(row.get("prompt_verdict", "")).strip().lower() == "unsafe"
        base_metadata = self._base_metadata(row, task="prompt_harm")
        normalized_rows: list[dict[str, Any]] = []
        for variant, prompt in (
            ("original", row.get("prompt")),
            ("jailbreaked", row.get("jailbreaked_prompt")),
        ):
            if prompt is None or not str(prompt).strip():
                continue
            normalized_rows.append(
                {
                    "id": f"circleguardbench-{row['prompt_hash']}-{variant}",
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "unsafe": unsafe,
                    **base_metadata,
                    "variant": variant,
                    "target_role": "user",
                }
            )
        return normalized_rows

    def _default_response_row(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build one response-harm sample from the default answer."""
        answer = row.get("default_answer")
        verdict = row.get("default_answer_verdict")
        if answer is None or not str(answer).strip() or verdict is None:
            return None
        return {
            "id": f"circleguardbench-{row['prompt_hash']}-default-answer",
            "messages": [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(answer)},
            ],
            "unsafe": str(verdict).strip().lower() == "unsafe",
            **self._base_metadata(row, task="default_response_harm"),
            "variant": "default_answer",
            "target_role": "assistant",
        }

    def _base_metadata(self, row: dict[str, Any], *, task: str) -> dict[str, Any]:
        """Return the shared preserved metadata for one row."""
        return {
            "harm_category": row.get("harm_category"),
            "source": row.get("source"),
            "intent": row.get("intent"),
            "behavioral_policies": row.get("behavioral_policies"),
            "low_level_harm_cluster_id": row.get("low_level_harm_cluster_id"),
            "prompt_hash": row.get("prompt_hash"),
            "prompt_verdict": row.get("prompt_verdict"),
            "default_answer_verdict": row.get("default_answer_verdict"),
            "task": task,
        }
