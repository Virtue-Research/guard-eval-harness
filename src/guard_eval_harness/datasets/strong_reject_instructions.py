"""Source-backed StrongREJECT Instructions dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/alexandrasouly/strongreject/raw/"
    "2db90fc5a283653f71ad1de86cfa0211d7bb90c6/strongreject_dataset"
)
_FILENAME = "strongreject_dataset.csv"


@dataset_registry.register("strong_reject_instructions")
class StrongRejectInstructionsDataset(DatasetAdapter):
    """Load StrongREJECT prompts from the published CSV source."""

    display_name = "StrongREJECT Instructions"
    source_uri = "https://github.com/alexandrasouly/strongreject"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Illegal Goods and Services",
        "Non-Violent Crimes",
        "Harassment and Discrimination",
        "Disinformation and Deception",
        "Violence",
        "Illicit Sexual Content",
    )
    metadata_fields_to_preserve = ("category",)
    label_mapping_note = "custom StrongREJECT prompts are all unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the StrongREJECT custom-source subset."""
        rows = load_csv_rows(
            alias="strong_reject_instructions",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        filtered_rows = [row for row in rows if row["source"] == "custom"]
        return [
            {
                "id": f"strongreject-{index}",
                "prompt": row["forbidden_prompt"],
                "unsafe": True,
                "category": row["category"],
            }
            for index, row in enumerate(filtered_rows, start=1)
        ]
