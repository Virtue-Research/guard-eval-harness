"""Source-backed NVIDIA Nemotron 3.5 Content Safety Dataset (text) adapter.

Loads the **text-only** rows of ``nvidia/Nemotron-3.5-Content-Safety-Dataset``
for Level-1 content-safety classification. The upstream set mixes text-only
rows with image-grounded rows and also carries a topic-following subset; this
adapter keeps only ``task_type == "safety"`` rows that have no ``image_path``
(image samples are handled separately, if at all). It is the Aegis-v3 lineage
multilingual extension of the Aegis 2.0 dataset and is scored the same way:
``prompt_harm`` uses ``input_label`` and ``response_harm`` uses
``response_label``.
"""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_REVISION = "841f5023b8db12b484180c6121bb10fcf400d1c3"
_SUPPORTED_TASKS = {"prompt_harm", "response_harm"}


@dataset_registry.register("nemotron_content_safety_text")
class NemotronContentSafetyTextDataset(SourceBackedDatasetAdapter):
    """Load prompt-harm or response-harm views of the Nemotron 3.5 text set."""

    display_name = "NVIDIA Nemotron 3.5 Content Safety Dataset (text)"
    source_uri = (
        "https://huggingface.co/datasets/"
        "nvidia/Nemotron-3.5-Content-Safety-Dataset"
    )
    license_name = "CC BY 4.0"
    languages = (
        "en",
        "fr",
        "de",
        "es",
        "it",
        "nl",
        "ar",
        "hi",
        "ja",
        "ko",
        "th",
        "zh",
    )
    metadata_fields_to_preserve = (
        "violated_categories",
        "language",
        "dataset_source",
        "provenance",
        "row_id",
        "task_type",
        "input_label_source",
        "response_label_source",
        "task",
        "target_role",
    )
    label_mapping_note = (
        "Nemotron 3.5 content-safety supports prompt_harm and response_harm "
        "tasks using the dataset input_label and response_label fields; only "
        "text-only rows (no image_path) with task_type == 'safety' are kept"
    )
    supported_splits = ("train", "validation", "test")

    def _task(self) -> str:
        """Return the configured task."""
        task = str(self.config.options.get("task", "prompt_harm")).strip().lower()
        if task not in _SUPPORTED_TASKS:
            supported = ", ".join(sorted(_SUPPORTED_TASKS))
            raise ValueError(
                "nemotron_content_safety_text supports options.task "
                f"values: {supported}"
            )
        return task

    @staticmethod
    def _is_text_safety_row(row: dict[str, Any]) -> bool:
        """Keep text-only safety rows; drop image-grounded / topic-following."""
        if row.get("image_path"):
            return False
        task_type = str(row.get("task_type", "")).strip().lower()
        return task_type == "safety"

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load and normalize the configured split's text-only safety rows."""
        task = self._task()
        rows = load_hf_rows(
            "nvidia/Nemotron-3.5-Content-Safety-Dataset",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if not self._is_text_safety_row(row):
                continue
            if task == "prompt_harm":
                normalized = self._prompt_harm_row(row)
            else:
                normalized = self._response_harm_row(row)
            if normalized is not None:
                normalized_rows.append(normalized)
        return normalized_rows

    def _prompt_harm_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Build a prompt-harm sample from one Nemotron text row."""
        prompt = str(row.get("prompt", "")).strip()
        label = row.get("input_label")
        if not prompt or label is None:
            return None
        return {
            "id": f"nemotron35-{row.get('row_id')}-prompt",
            "messages": [{"role": "user", "content": prompt}],
            "unsafe": str(label).strip().lower() == "unsafe",
            "category_labels": row.get("violated_categories"),
            "violated_categories": row.get("violated_categories"),
            "language": row.get("language"),
            "dataset_source": row.get("dataset_source"),
            "provenance": row.get("provenance"),
            "row_id": row.get("row_id"),
            "task_type": row.get("task_type"),
            "input_label_source": row.get("input_label_source"),
            "response_label_source": row.get("response_label_source"),
            "task": "prompt_harm",
            "target_role": "user",
        }

    def _response_harm_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Build a response-harm sample from one Nemotron text row."""
        prompt = str(row.get("prompt", "")).strip()
        response = row.get("response")
        label = row.get("response_label")
        if not prompt or response is None or label is None:
            return None
        response_text = str(response)
        if not response_text.strip():
            return None
        return {
            "id": f"nemotron35-{row.get('row_id')}-response",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text},
            ],
            "unsafe": str(label).strip().lower() == "unsafe",
            "category_labels": row.get("violated_categories"),
            "violated_categories": row.get("violated_categories"),
            "language": row.get("language"),
            "dataset_source": row.get("dataset_source"),
            "provenance": row.get("provenance"),
            "row_id": row.get("row_id"),
            "task_type": row.get("task_type"),
            "input_label_source": row.get("input_label_source"),
            "response_label_source": row.get("response_label_source"),
            "task": "response_harm",
            "target_role": "assistant",
        }
