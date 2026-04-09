"""Source-backed HateCheck dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/paul-rottger/hatecheck-data/raw/"
    "d84302298ceeb34f4b0dde2d8a1f4932c2caa612"
)
_FILENAME = "test_suite_cases.csv"


def _label_balanced_rows(
    rows: list[dict[str, object]],
    *,
    execution_limit: int | None,
) -> list[dict[str, object]]:
    """Balance small smoke slices across hateful and non-hateful rows."""
    if execution_limit is None:
        return rows

    hateful_rows = [row for row in rows if row["label_gold"] == "hateful"]
    safe_rows = [row for row in rows if row["label_gold"] != "hateful"]
    hateful_limit = execution_limit - (execution_limit // 2)
    safe_limit = execution_limit // 2
    balanced_rows = hateful_rows[:hateful_limit] + safe_rows[:safe_limit]
    return balanced_rows[:execution_limit]


@dataset_registry.register("hatecheck")
class HateCheckDataset(SourceBackedDatasetAdapter):
    """Load the HateCheck statement suite."""

    display_name = "HateCheck"
    source_uri = "https://github.com/paul-rottger/hatecheck-data"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("hate",)
    metadata_fields_to_preserve = ("target_ident",)
    label_mapping_note = (
        "rows labeled hateful map to unsafe and non-hateful rows map to safe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream HateCheck CSV file."""
        rows = load_csv_rows(
            alias="hatecheck",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        rows = _label_balanced_rows(rows, execution_limit=self.execution_limit)
        return [
            {
                "id": str(row["case_id"]),
                "prompt": row["test_case"],
                "unsafe": row["label_gold"] == "hateful",
                "target_ident": row.get("target_ident"),
            }
            for row in rows
        ]
