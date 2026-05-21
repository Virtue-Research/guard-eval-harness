"""Source-backed SimpleSafetyTests dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5"


@dataset_registry.register("simple_safety_tests")
class SimpleSafetyTestsDataset(DatasetAdapter):
    """Load the SimpleSafetyTests prompt set."""

    display_name = "SimpleSafetyTests"
    source_uri = "https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = (
        "Suicide, Self-Harm, and Eating Disorders",
        "Physical Harm",
        "Illegal and Highly Regulated Items",
        "Scams and Fraud",
        "Child Abuse",
    )
    label_mapping_note = "all SimpleSafetyTests prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the SimpleSafetyTests test split from Hugging Face."""
        rows = load_hf_rows(
            "Bertievidgen/SimpleSafetyTests",
            split="test",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": str(row["id"]),
                "prompt": row["prompt"],
                "unsafe": True,
            }
            for row in rows
        ]
