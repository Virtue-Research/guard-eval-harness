"""Source-backed Hatemoji Check dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/HannahKirk/Hatemoji/raw/"
    "a626f63216158748d072e581efc00267031f4404/HatemojiCheck"
)
_FILENAME = "test.csv"


@dataset_registry.register("hatemoji_check")
class HatemojiCheckDataset(DatasetAdapter):
    """Load the Hatemoji Check benchmark from its published CSV."""

    display_name = "Hatemoji Check"
    source_uri = "https://github.com/HannahKirk/Hatemoji"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("hate",)
    metadata_fields_to_preserve = ("target", "functionality", "set")
    label_mapping_note = (
        "Hatemoji label_gold is mapped directly after filtering unrealistic rows"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the filtered Hatemoji test rows."""
        rows = load_csv_rows(
            alias="hatemoji_check",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        filtered_rows = [
            row for row in rows if int(row["unrealistic_flags"] or "0") == 0
        ]
        return [
            {
                "id": str(row["case_id"]),
                "prompt": row["text"],
                "unsafe": bool(int(row["label_gold"])),
                "target": row["target"],
                "functionality": row["functionality"],
                "set": row["set"],
            }
            for row in filtered_rows
        ]
