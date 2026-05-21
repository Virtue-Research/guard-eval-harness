"""Source-backed CatQA dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "321e068b391b7d1aeeb882ac468356a9513c9167"


@dataset_registry.register("cat_qa")
class CatQADataset(DatasetAdapter):
    """Load the English CatQA harmful question set."""

    display_name = "CatQA"
    source_uri = "https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA"
    license_name = "Apache 2.0"
    languages = ("en",)
    supported_splits = ("en",)
    categories = (
        "Illegal Activity",
        "Child Abuse",
        "Hate/Harass/Violence",
        "Malware Viruses",
        "Physical Harm",
        "Economic Harm",
        "Fraud/Deception",
        "Adult Content",
        "Political Campaigning",
        "Privacy Violation Activity",
        "Tailored Financial Advice",
    )
    label_mapping_note = "all CatQA prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the English CatQA split from Hugging Face."""
        rows = load_hf_rows(
            "declare-lab/CategoricalHarmfulQA",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"cat-qa-{index}",
                "prompt": row["Question"],
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
