"""Source-backed JailbreakBench dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("jailbreakbench")
class JailbreakBenchDataset(SourceBackedDatasetAdapter):
    """Load the JailbreakBench behavior corpus from HF."""

    display_name = "JailbreakBench"
    source_uri = (
        "https://huggingface.co/datasets/"
        "JailbreakBench/JBB-Behaviors"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = ("jailbreak",)
    metadata_fields_to_preserve = ("category",)
    label_mapping_note = (
        "all JailbreakBench behaviors are unsafe prompts"
    )
    supported_splits = ("harmful",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load JailbreakBench behaviors from Hugging Face."""
        rows = load_hf_rows(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="harmful",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"jailbreakbench-{index}",
                "prompt": row.get("Goal", row.get("Behavior", "")),
                "unsafe": True,
                "category": row.get("Category", ""),
            }
            for index, row in enumerate(rows, start=1)
        ]
