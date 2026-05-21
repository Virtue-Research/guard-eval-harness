"""Source-backed SALAD-Bench dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "70a1ddce6be3f0239b2fe7c70d0d96cb3be1fcad"


@dataset_registry.register("salad_bench")
class SaladBenchDataset(DatasetAdapter):
    """Load the SALAD-Bench base safety question set."""

    display_name = "SALAD-Bench"
    source_uri = (
        "https://huggingface.co/datasets/OpenSafetyLab/Salad-Data"
    )
    license_name = "Apache 2.0"
    languages = ("en",)
    categories = (
        "Representation & Toxicity",
        "Unfair Discrimination",
        "Information & Safety",
        "Malicious Use",
        "Human Autonomy & Integrity",
        "Socioeconomic Harms",
    )
    metadata_fields_to_preserve = (
        "1-category",
        "2-category",
        "3-category",
        "source",
    )
    label_mapping_note = (
        "all SALAD-Bench base set questions are unsafe"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the SALAD-Bench base_set from Hugging Face."""
        rows = load_hf_rows(
            "OpenSafetyLab/Salad-Data",
            subset="base_set",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"salad-{row.get('qid', index)}",
                "prompt": row["question"],
                "unsafe": True,
                "1-category": row.get("1-category"),
                "2-category": row.get("2-category"),
                "3-category": row.get("3-category"),
                "source": row.get("source"),
            }
            for index, row in enumerate(rows, start=1)
        ]
