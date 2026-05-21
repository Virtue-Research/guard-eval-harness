"""Source-backed OLID dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "37262ae493ed725b68b93aab1632b8207760dc8f"


@dataset_registry.register("olid")
class OlidDataset(DatasetAdapter):
    """Load the Offensive Language Identification Dataset."""

    display_name = "OLID"
    source_uri = (
        "https://huggingface.co/datasets/christophsonntag/OLID"
    )
    languages = ("en",)
    categories = ("Offensive",)
    metadata_fields_to_preserve = ("subtask_b", "subtask_c")
    label_mapping_note = (
        "subtask_a OFF maps to unsafe and NOT maps to safe"
    )
    supported_splits = ("train", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured OLID split from Hugging Face."""
        rows = load_hf_rows(
            "christophsonntag/OLID",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"olid-{row['id']}",
                "prompt": row["tweet"],
                "unsafe": row["subtask_a"] == "OFF",
                "subtask_b": row.get("subtask_b"),
                "subtask_c": row.get("subtask_c"),
            }
            for row in rows
        ]
