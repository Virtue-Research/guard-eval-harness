"""Source-backed HarmEval dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "d1f62d8c8a42b919d7466fcd1896dc9cdb8134c3"


@dataset_registry.register("harm_eval")
class HarmEvalDataset(DatasetAdapter):
    """Load the HarmEval harmful question set."""

    display_name = "HarmEval"
    source_uri = "https://huggingface.co/datasets/SoftMINER-Group/HarmEval"
    license_name = "Apache 2.0"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = ("category",)
    categories = (
        "Adult Content",
        "Child Abuse Content",
        "Economic Harm",
        "Fraud/Deception",
        "Hate/Harass/Violence",
        "Illegal Activity",
        "Malware",
        "Physical Harm",
        "Political Campaigning",
        "Privacy Violation Activity",
        "Tailored Financial Advice",
    )
    label_mapping_note = "all HarmEval prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the HarmEval training split from Hugging Face."""
        rows = load_hf_rows(
            "SoftMINER-Group/HarmEval",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"harm-eval-{index}",
                "prompt": row["Question"],
                "unsafe": True,
                "category": row.get("Topic"),
            }
            for index, row in enumerate(rows, start=1)
        ]
