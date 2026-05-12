"""Source-backed DICES 990 dataset adapter."""

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


@dataset_registry.register("dices_990")
class DICES990Dataset(SourceBackedDatasetAdapter):
    """Load the DICES 990 dataset from Hugging Face."""

    display_name = "DICES 990"
    source_uri = (
        "https://huggingface.co/datasets/google/dices_dataset"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("dialog_safety",)
    metadata_fields_to_preserve = ("target_role",)
    label_mapping_note = (
        "degree_of_harm ClassLabel: 3=Extreme and 4=Moderate map "
        "to unsafe; 1=Benign maps to safe; 0 and 2 excluded"
    )
    supported_splits = ("train",)

    # ClassLabel: 0=No annotation, 1=Benign, 2=Debatable,
    #             3=Extreme, 4=Moderate
    _UNSAFE_IDS = {3, 4}  # Extreme, Moderate
    _SAFE_IDS = {1}  # Benign
    _DEGREE_NAMES = {
        0: "No annotation",
        1: "Benign",
        2: "Debatable",
        3: "Extreme",
        4: "Moderate",
    }

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load DICES 990 conversations from Hugging Face."""
        rows = load_hf_rows(
            "google/dices_dataset",
            subset="990",
            split="train",
            revision=_REVISION,
            limit=(
                self.execution_limit * 4
                if self.execution_limit
                else None
            ),
        )
        raw_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            degree_raw = row.get("degree_of_harm")
            if isinstance(degree_raw, int):
                if (
                    degree_raw not in self._UNSAFE_IDS
                    and degree_raw not in self._SAFE_IDS
                ):
                    continue
                is_unsafe = degree_raw in self._UNSAFE_IDS
                degree = self._DEGREE_NAMES.get(
                    degree_raw, str(degree_raw)
                )
            else:
                degree = str(degree_raw).strip()
                if degree in {
                    "No annotation",
                    "Debatable",
                }:
                    continue
                is_unsafe = degree in {
                    "Moderate",
                    "Extreme",
                }
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
                    "id": f"dices-990-{index}",
                    "messages": messages,
                    "unsafe": is_unsafe,
                    "degree_of_harm": degree,
                    "target_role": "conversation",
                }
            )
        return _label_balanced_rows(
            raw_rows, execution_limit=self.execution_limit
        )
