"""Source-backed TechHazardQA dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "d100fa1222a7179e0f492813129054dc19195408"


@dataset_registry.register("tech_hazard_qa")
class TechHazardQADataset(SourceBackedDatasetAdapter):
    """Load the TechHazardQA harmful question set."""

    display_name = "TechHazardQA"
    source_uri = "https://huggingface.co/datasets/SoftMINER-Group/TechHazardQA"
    license_name = "Apache 2.0"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = ("category",)
    categories = (
        "Bio Technology, Biology, Genetic Engineering",
        "Chemical Weapon, Biological and Chemical Weapons",
        "Cyber Security",
        "Finance, Banking",
        "Nuclear Technology, Bio Nuclear Terrorism, Terrorism",
        "Public Healthcare System, Pharmacology",
        "Social Media",
    )
    label_mapping_note = "all TechHazardQA prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the TechHazardQA training split from Hugging Face."""
        rows = load_hf_rows(
            "SoftMINER-Group/TechHazardQA",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"tech-hazard-qa-{index}",
                "prompt": row["Question"],
                "unsafe": True,
                "category": row.get("Topic"),
            }
            for index, row in enumerate(rows, start=1)
        ]
