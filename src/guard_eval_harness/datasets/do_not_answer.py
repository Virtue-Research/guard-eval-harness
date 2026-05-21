"""Source-backed DoNotAnswer dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "74e74f2e4507ef256fe536f78a776f4a1ff67955"


@dataset_registry.register("do_not_answer")
class DoNotAnswerDataset(DatasetAdapter):
    """Load the public DoNotAnswer Hugging Face dataset."""

    display_name = "DoNotAnswer"
    source_uri = "https://huggingface.co/datasets/LibrAI/do-not-answer"
    license_name = "Apache 2.0"
    languages = ("en",)
    categories = (
        "Mental Health",
        "Self-Harm",
        "Disinformation",
        "Spam",
        "Illegal Activities",
        "Unethical Actions",
        "Discrimination",
        "Hate",
        "Privacy",
    )
    metadata_fields_to_preserve = (
        "risk_area",
        "types_of_harm",
        "specific_harms",
    )
    label_mapping_note = "all DoNotAnswer questions are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream train split as the benchmark test set."""
        rows = load_hf_rows(
            "LibrAI/do-not-answer",
            split="train",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": str(row["id"]),
                "prompt": row["question"],
                "unsafe": True,
                "risk_area": row["risk_area"],
                "types_of_harm": row["types_of_harm"],
                "specific_harms": row["specific_harms"],
            }
            for row in rows
        ]
