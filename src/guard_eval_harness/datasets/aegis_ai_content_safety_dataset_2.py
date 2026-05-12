"""Source-backed NVIDIA Aegis AI Content Safety Dataset 2.0 adapter."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
_SUPPORTED_TASKS = {"prompt_harm", "response_harm"}


@dataset_registry.register("aegis_ai_content_safety_dataset_2")
class AegisAIContentSafetyDataset2(SourceBackedDatasetAdapter):
    """Load prompt-harm or response-harm views of the Aegis 2.0 dataset."""

    display_name = "NVIDIA Aegis AI Content Safety Dataset 2.0"
    source_uri = (
        "https://huggingface.co/datasets/"
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    metadata_fields_to_preserve = (
        "prompt_label_source",
        "response_label_source",
        "response_was_empty",
        "task",
        "target_role",
    )
    label_mapping_note = (
        "Aegis 2.0 supports prompt_harm and response_harm tasks using the "
        "dataset prompt_label and response_label fields"
    )
    supported_splits = ("train", "validation", "test")

    def _task(self) -> str:
        """Return the configured Aegis task."""
        task = str(self.config.options.get("task", "prompt_harm")).strip().lower()
        if task not in _SUPPORTED_TASKS:
            supported = ", ".join(sorted(_SUPPORTED_TASKS))
            raise ValueError(
                "aegis_ai_content_safety_dataset_2 supports options.task "
                f"values: {supported}"
            )
        return task

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load and normalize the configured Aegis split."""
        task = self._task()
        rows = load_hf_rows(
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if task == "prompt_harm":
                normalized = self._prompt_harm_row(row)
            else:
                normalized = self._response_harm_row(row)
            if normalized is not None:
                normalized_rows.append(normalized)
        return normalized_rows

    def _prompt_harm_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Build a prompt-harm sample from one Aegis row."""
        prompt = str(row.get("prompt", "")).strip()
        label = row.get("prompt_label")
        if not prompt or label is None:
            return None
        return {
            "id": f"aegis2-{row['id']}-prompt",
            "messages": [{"role": "user", "content": prompt}],
            "unsafe": str(label).strip().lower() == "unsafe",
            "violated_categories": row.get("violated_categories"),
            "prompt_label_source": row.get("prompt_label_source"),
            "response_label_source": row.get("response_label_source"),
            "task": "prompt_harm",
            "target_role": "user",
        }

    def _response_harm_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Build a response-harm sample from one Aegis row."""
        prompt = str(row.get("prompt", "")).strip()
        response = row.get("response")
        label = row.get("response_label")
        if not prompt or response is None or label is None:
            return None
        response_text = str(response)
        return {
            "id": f"aegis2-{row['id']}-response",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text},
            ],
            "unsafe": str(label).strip().lower() == "unsafe",
            "violated_categories": row.get("violated_categories"),
            "prompt_label_source": row.get("prompt_label_source"),
            "response_label_source": row.get("response_label_source"),
            "response_was_empty": response_text == "",
            "task": "response_harm",
            "target_role": "assistant",
        }
