"""Source-backed XSTest dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/paul-rottger/exaggerated-safety/raw/"
    "d21c3bc28b35ed3bf54344037af9760ffae6527e"
)
_FILENAME = "xstest_v2_prompts.csv"


@dataset_registry.register("xstest")
class XSTestDataset(DatasetAdapter):
    """Load the full XSTest v2 prompt suite."""

    display_name = "XSTest"
    source_uri = "https://github.com/paul-rottger/exaggerated-safety"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("generic",)
    metadata_fields_to_preserve = ("type", "focus", "note")
    label_mapping_note = (
        "XSTest rows with a 'contrast' type are treated as unsafe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load and normalize the upstream XSTest CSV file."""
        rows = load_csv_rows(
            alias="xstest",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        return [
            {
                "id": str(row["id_v2"]),
                "prompt": row["prompt"],
                "unsafe": "contrast" in row["type"],
                "type": row["type"],
                "focus": row["focus"],
                "note": row["note"],
            }
            for row in rows
        ]
