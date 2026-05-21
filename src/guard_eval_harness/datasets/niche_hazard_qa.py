"""Source-backed NicheHazardQA dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "114a05b168027c3190dc4e2f3b558fe48e6092c8"


@dataset_registry.register("niche_hazard_qa")
class NicheHazardQADataset(DatasetAdapter):
    """Load the NicheHazardQA harmful question set."""

    display_name = "NicheHazardQA"
    source_uri = "https://huggingface.co/datasets/SoftMINER-Group/NicheHazardQA"
    license_name = "Apache 2.0"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = ("category",)
    categories = (
        "Advanced Technology to Create Weapons",
        "Conspiracy Theories and Paranoia",
        "Control the Thoughts and Emotions of Learners",
        "Cruelty and Violence",
        "Fake News and Propaganda",
        "Hate Speech and Discrimination",
    )
    label_mapping_note = "all NicheHazardQA prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the NicheHazardQA training split from Hugging Face."""
        rows = load_hf_rows(
            "SoftMINER-Group/NicheHazardQA",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"niche-hazard-qa-{index}",
                "prompt": row["Question"],
                "unsafe": True,
                "category": row.get("Domain"),
            }
            for index, row in enumerate(rows, start=1)
        ]
