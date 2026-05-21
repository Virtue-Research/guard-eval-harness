"""Source-backed ProsocialDialog dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


def _label_balanced_rows(
    rows: list[dict[str, object]],
    *,
    execution_limit: int | None,
) -> list[dict[str, object]]:
    """Balance small smoke slices across unsafe and safe."""
    if execution_limit is None:
        return rows

    unsafe = [r for r in rows if r.get("unsafe")]
    safe = [r for r in rows if not r.get("unsafe")]
    unsafe_limit = execution_limit - (execution_limit // 2)
    safe_limit = execution_limit // 2
    return (unsafe[:unsafe_limit] + safe[:safe_limit])[
        :execution_limit
    ]


@dataset_registry.register("prosocial_dialog")
class ProsocialDialogDataset(DatasetAdapter):
    """Load the ProsocialDialog dataset from Hugging Face."""

    display_name = "ProsocialDialog"
    source_uri = (
        "https://huggingface.co/datasets/allenai/prosocial-dialog"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("dialog_safety", "prosocial")
    metadata_fields_to_preserve = (
        "safety_label",
        "target_role",
    )
    label_mapping_note = (
        "all non-__casual__ safety_label values map to unsafe"
    )
    supported_splits = ("train", "validation", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load ProsocialDialog rows from Hugging Face."""
        rows = load_hf_rows(
            "allenai/prosocial-dialog",
            split=self.config.split,
            revision=_REVISION,
            limit=(
                self.execution_limit * 4
                if self.execution_limit
                else None
            ),
        )
        raw_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            safety = str(
                row.get("safety_label", "")
            ).strip()
            is_unsafe = safety != "__casual__" and safety != ""
            context = str(row.get("context", "")).strip()
            response = str(row.get("response", "")).strip()
            messages: list[dict[str, str]] = []
            if context:
                messages.append(
                    {"role": "user", "content": context}
                )
            if response:
                messages.append(
                    {"role": "assistant", "content": response}
                )
            if not messages:
                continue
            raw_rows.append(
                {
                    "id": (
                        f"prosocial-{self.config.split}-{index}"
                    ),
                    "messages": messages,
                    "unsafe": is_unsafe,
                    "safety_label": safety,
                    "target_role": "user",
                }
            )
        return _label_balanced_rows(
            raw_rows, execution_limit=self.execution_limit
        )
