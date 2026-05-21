"""Source-backed WildGuardMix dataset adapter."""

from typing import Any

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
_SPLIT_MAP = {
    "train": "wildguardtrain",
    "test": "wildguardtest",
}
_SUPPORTED_TASKS = {"prompt_harm", "response_harm"}


@dataset_registry.register("wildguardmix")
class WildGuardMixDataset(DatasetAdapter):
    """Load WildGuardMix prompt-harm or response-harm examples."""

    display_name = "WildGuardMix"
    source_uri = "https://huggingface.co/datasets/allenai/wildguardmix"
    license_name = "ODC-By"
    languages = ("en",)
    metadata_fields_to_preserve = (
        "adversarial",
        "subcategory",
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
        "prompt_harm_agreement",
        "response_harm_agreement",
        "response_refusal_agreement",
        "task",
        "target_role",
    )
    label_mapping_note = (
        "WildGuardMix supports two binary tasks via options.task: "
        "'prompt_harm' marks unsafe user prompts, and 'response_harm' marks "
        "unsafe assistant responses"
    )
    supported_splits = ("train", "test")

    def _task(self) -> str:
        """Return the configured WildGuardMix task."""
        task = str(self.config.options.get("task", "prompt_harm")).strip().lower()
        if task not in _SUPPORTED_TASKS:
            supported = ", ".join(sorted(_SUPPORTED_TASKS))
            raise ValueError(
                f"wildguardmix supports options.task values: {supported}"
            )
        return task

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load and normalize the configured WildGuardMix split."""
        task = self._task()
        rows = load_hf_rows(
            "allenai/wildguardmix",
            subset=_SPLIT_MAP[self.config.split],
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows, start=1):
            normalized = self._normalize_task_row(
                row,
                row_index=row_index,
                task=task,
            )
            if normalized is not None:
                normalized_rows.append(normalized)
        return normalized_rows

    def _normalize_task_row(
        self,
        row: dict[str, Any],
        *,
        row_index: int,
        task: str,
    ) -> dict[str, Any] | None:
        """Map one WildGuardMix record into the requested task view."""
        if task == "prompt_harm":
            label = row.get("prompt_harm_label")
            if label is None:
                return None
            return {
                "id": (
                    f"wildguardmix-{self.config.split}-prompt-harm-"
                    f"{row_index:05d}"
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": row["prompt"],
                    }
                ],
                "unsafe": str(label).strip().lower() == "harmful",
                "adversarial": bool(row.get("adversarial", False)),
                "subcategory": row.get("subcategory"),
                "prompt_harm_label": label,
                "response_harm_label": row.get("response_harm_label"),
                "response_refusal_label": row.get("response_refusal_label"),
                "prompt_harm_agreement": row.get("prompt_harm_agreement"),
                "response_harm_agreement": row.get("response_harm_agreement"),
                "response_refusal_agreement": row.get(
                    "response_refusal_agreement"
                ),
                "task": task,
                "target_role": "user",
            }

        label = row.get("response_harm_label")
        response = row.get("response")
        if label is None or response is None or not str(response).strip():
            return None
        return {
            "id": (
                f"wildguardmix-{self.config.split}-response-harm-"
                f"{row_index:05d}"
            ),
            "messages": [
                {
                    "role": "user",
                    "content": row["prompt"],
                },
                {
                    "role": "assistant",
                    "content": response,
                },
            ],
            "unsafe": str(label).strip().lower() == "harmful",
            "adversarial": bool(row.get("adversarial", False)),
            "subcategory": row.get("subcategory"),
            "prompt_harm_label": row.get("prompt_harm_label"),
            "response_harm_label": label,
            "response_refusal_label": row.get("response_refusal_label"),
            "prompt_harm_agreement": row.get("prompt_harm_agreement"),
            "response_harm_agreement": row.get("response_harm_agreement"),
            "response_refusal_agreement": row.get(
                "response_refusal_agreement"
            ),
            "task": task,
            "target_role": "assistant",
        }
