"""Source-backed Implicit Hate (stage 1) dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "02f0834f16f2c0bbf38654378f29c26402e47d15"


@dataset_registry.register("implicit_hate")
class ImplicitHateDataset(SourceBackedDatasetAdapter):
    """Load the Implicit Hate stage-1 classification dataset."""

    display_name = "Implicit Hate"
    source_uri = (
        "https://huggingface.co/datasets/tasksource/implicit-hate-stg1"
    )
    languages = ("en",)
    categories = (
        "Explicit Hate",
        "Implicit Hate",
    )
    metadata_fields_to_preserve = ("class",)
    label_mapping_note = (
        "explicit_hate and implicit_hate map to unsafe; "
        "not_hate maps to safe"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the implicit hate stage-1 split."""
        rows = load_hf_rows(
            "tasksource/implicit-hate-stg1",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"implicit-hate-{index}",
                "prompt": row["post"],
                "unsafe": row["class"] != "not_hate",
                "class": row.get("class"),
            }
            for index, row in enumerate(rows, start=1)
            if str(row.get("post", "")).strip()
        ]
