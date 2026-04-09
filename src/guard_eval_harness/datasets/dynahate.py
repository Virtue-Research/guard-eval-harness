"""Source-backed DynaHate dataset adapter."""

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
    """Balance small smoke slices across hateful and non-hateful."""
    if execution_limit is None:
        return rows

    hateful = [r for r in rows if r.get("unsafe")]
    safe = [r for r in rows if not r.get("unsafe")]
    hateful_limit = execution_limit - (execution_limit // 2)
    safe_limit = execution_limit // 2
    return (hateful[:hateful_limit] + safe[:safe_limit])[
        :execution_limit
    ]


@dataset_registry.register("dynahate")
class DynaHateDataset(SourceBackedDatasetAdapter):
    """Load the DynaHate hate-speech dataset from Hugging Face."""

    display_name = "DynaHate"
    source_uri = (
        "https://huggingface.co/datasets/aps/dynahate"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("hate",)
    metadata_fields_to_preserve = ("round", "target")
    label_mapping_note = (
        "ClassLabel 0=nothate (safe), 1=hate (unsafe)"
    )
    supported_splits = ("train", "test")

    # ClassLabel: 0=nothate, 1=hate
    _LABEL_NAMES = {0: "nothate", 1: "hate"}

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load DynaHate rows from Hugging Face."""
        rows = load_hf_rows(
            "aps/dynahate",
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
            raw_label = row["label"]
            if isinstance(raw_label, int):
                is_unsafe = raw_label == 1
                label_str = self._LABEL_NAMES.get(
                    raw_label, str(raw_label)
                )
            else:
                label_str = str(raw_label)
                is_unsafe = label_str == "hate"
            raw_rows.append(
                {
                    "id": (
                        f"dynahate-{self.config.split}"
                        f"-{index}"
                    ),
                    "prompt": row["text"],
                    "unsafe": is_unsafe,
                    "label": label_str,
                    "round": str(
                        row.get("round.base", "")
                    ),
                    "target": row.get("target", ""),
                }
            )
        return _label_balanced_rows(
            raw_rows, execution_limit=self.execution_limit
        )
