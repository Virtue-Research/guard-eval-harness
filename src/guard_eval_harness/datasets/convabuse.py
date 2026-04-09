"""Source-backed ConvAbuse dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/amandacurry/convabuse/raw/"
    "c0a9469f48956e868276c605d499010ea3c7c0d0/2_splits"
)
_FILENAMES = {
    "train": "ConvAbuseEMNLPtrain.csv",
    "val": "ConvAbuseEMNLPvalid.csv",
    "test": "ConvAbuseEMNLPtest.csv",
}


@dataset_registry.register("convabuse")
class ConvAbuseDataset(SourceBackedDatasetAdapter):
    """Load the published ConvAbuse CSV splits."""

    display_name = "ConvAbuse"
    source_uri = "https://github.com/amandacurry/convabuse"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = (
        "Sexism",
        "Sexual Harassment",
        "Racism",
        "Transphobia",
        "Ableism",
        "General",
        "Homophobia",
        "Intelligence",
    )
    metadata_fields_to_preserve = ("bot", "target_role")
    label_mapping_note = (
        "abuse is determined by a majority vote over available annotators"
    )
    supported_splits = ("train", "val", "test")

    def _row_is_abusive(self, row: dict[str, str]) -> bool:
        """Collapse multi-annotator labels into one unsafe value."""
        annotations: list[bool] = []
        for index in range(1, 9):
            key_base = f"Annotator{index}_is_abuse"
            if not any(row.get(f"{key_base}.{value}") == "1" for value in range(-3, 2)):
                continue
            annotations.append(
                any(
                    row.get(f"{key_base}.{value}") == "1"
                    for value in (-1, -2, -3)
                )
            )
        if not annotations:
            raise ValueError("ConvAbuse row has no annotator votes")
        return sum(annotations) / len(annotations) > 0.5

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured ConvAbuse split."""
        filename = _FILENAMES[self.config.split]
        rows = load_csv_rows(
            alias="convabuse",
            url=f"{_BASE_URL}/{filename}",
            filename=filename,
        )
        return [
            {
                "id": str(row["example_id"]),
                "messages": [
                    {"role": "assistant", "content": row["prev_agent"]},
                    {"role": "user", "content": row["prev_user"]},
                    {"role": "assistant", "content": row["agent"]},
                    {"role": "user", "content": row["user"]},
                ],
                "unsafe": self._row_is_abusive(row),
                "bot": row["bot"],
                "target_role": "user",
            }
            for row in rows
        ]
