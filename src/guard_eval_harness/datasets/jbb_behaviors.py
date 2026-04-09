"""Source-backed JBB Behaviors dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"


@dataset_registry.register("jbb_behaviors")
class JBBBehaviorsDataset(SourceBackedDatasetAdapter):
    """Load JailbreakBench behaviors from Hugging Face."""

    display_name = "JBB Behaviors"
    source_uri = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Disinformation",
        "Economic harm",
        "Expert advice",
        "Fraud/Deception",
        "Government decision-making",
        "Harassment/Discrimination",
        "Malware/Hacking",
        "Physical harm",
        "Privacy",
        "Sexual/Adult content",
    )
    metadata_fields_to_preserve = ("behavior", "category", "source")
    label_mapping_note = (
        "harmful split rows are unsafe and benign split rows are safe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load and merge the benign and harmful splits."""
        benign_rows = load_hf_rows(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="benign",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        harmful_rows = load_hf_rows(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="harmful",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, object]] = []
        for prefix, unsafe, rows in (
            ("safe", False, benign_rows),
            ("unsafe", True, harmful_rows),
        ):
            normalized_rows.extend(
                {
                    "id": f"{prefix}_{row['Index']}",
                    "prompt": row["Goal"],
                    "unsafe": unsafe,
                    "behavior": row["Behavior"],
                    "category": row["Category"],
                    "source": row["Source"],
                }
                for row in rows
            )
        return normalized_rows
