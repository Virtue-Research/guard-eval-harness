"""Source-backed AART dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("aart")
class AARTDataset(SourceBackedDatasetAdapter):
    """Load the AART AI-generated adversarial prompt set."""

    display_name = "AART"
    source_uri = (
        "https://huggingface.co/datasets/google/aart-ai-safety-dataset"
    )
    license_name = "Apache 2.0"
    languages = ("en",)
    categories = ("adversarial_prompts",)
    metadata_fields_to_preserve = ("harm_category",)
    label_mapping_note = "all AART prompts are adversarial and unsafe"
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load AART rows from Hugging Face."""
        rows = load_hf_rows(
            "google/aart-ai-safety-dataset",
            split="train",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"aart-{index}",
                "prompt": row["prompt"],
                "unsafe": True,
                "harm_category": row.get("harm_category", ""),
            }
            for index, row in enumerate(rows, start=1)
        ]
