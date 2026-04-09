"""Source-backed DICES 350 dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
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


@dataset_registry.register("dices_350")
class DICES350Dataset(SourceBackedDatasetAdapter):
    """Load the DICES 350 dataset from Hugging Face."""

    display_name = "DICES 350"
    source_uri = (
        "https://huggingface.co/datasets/google/dices_dataset"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("dialog_safety",)
    metadata_fields_to_preserve = ("target_role",)
    label_mapping_note = (
        "Q_overall ClassLabel 0=No (unsafe), 1=Yes (safe), "
        "2=Unsure (excluded)"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load DICES 350 conversations from Hugging Face."""
        rows = load_hf_rows(
            "google/dices_dataset",
            subset="350",
            split="train",
            revision=_REVISION,
            limit=(
                self.execution_limit * 4
                if self.execution_limit
                else None
            ),
        )
        # Q_overall is ClassLabel: 0=No, 1=Yes, 2=Unsure
        _UNSAFE_IDS = {0}  # No = unsafe
        _SAFE_IDS = {1}  # Yes = safe
        raw_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            q_overall = row.get("Q_overall")
            if isinstance(q_overall, int):
                if q_overall not in _UNSAFE_IDS | _SAFE_IDS:
                    continue  # skip Unsure
                is_unsafe = q_overall in _UNSAFE_IDS
            else:
                safety = str(q_overall).strip()
                if safety == "Unsure":
                    continue
                is_unsafe = safety == "No"
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
                    "id": f"dices-350-{index}",
                    "messages": messages,
                    "unsafe": is_unsafe,
                    "target_role": "conversation",
                }
            )
        return _label_balanced_rows(
            raw_rows, execution_limit=self.execution_limit
        )
