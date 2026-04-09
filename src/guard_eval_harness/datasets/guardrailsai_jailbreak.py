"""Source-backed GuardrailsAI detect-jailbreak dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("guardrailsai_jailbreak")
class GuardrailsAIJailbreakDataset(SourceBackedDatasetAdapter):
    """Load the GuardrailsAI jailbreak detection dataset."""

    display_name = "GuardrailsAI Detect-Jailbreak"
    source_uri = (
        "https://huggingface.co/datasets/"
        "GuardrailsAI/detect-jailbreak"
    )
    languages = ("en",)
    categories = ("Jailbreak",)
    metadata_fields_to_preserve = ("source", "tags")
    label_mapping_note = (
        "is_jailbreak True maps to unsafe; False maps to "
        "safe"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load GuardrailsAI jailbreak detection rows."""
        rows = load_hf_rows(
            "GuardrailsAI/detect-jailbreak",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"guardrailsai-jailbreak-{index:05d}",
                "prompt": row["prompt"],
                "unsafe": bool(row["is_jailbreak"]),
                "source": row.get("source"),
                "tags": row.get("tags"),
            }
            for index, row in enumerate(rows, start=1)
            if str(row.get("prompt", "")).strip()
        ]
