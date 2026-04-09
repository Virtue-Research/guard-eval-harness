"""Source-backed Hate Speech Offensive dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "adc5fb774614827695774f2dbe0ea8122f6a92b4"


@dataset_registry.register("hate_speech_offensive")
class HateSpeechOffensiveDataset(SourceBackedDatasetAdapter):
    """Load the Davidson et al. Hate Speech and Offensive Language
    dataset."""

    display_name = "Hate Speech Offensive"
    source_uri = (
        "https://huggingface.co/datasets/hate_speech_offensive"
    )
    languages = ("en",)
    categories = (
        "Hate Speech",
        "Offensive Language",
    )
    metadata_fields_to_preserve = (
        "class",
        "hate_speech_count",
        "offensive_language_count",
        "neither_count",
    )
    label_mapping_note = (
        "class 0 (hate speech) and class 1 (offensive language) "
        "map to unsafe; class 2 (neither) maps to safe"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the hate speech offensive language dataset."""
        rows = load_hf_rows(
            "hate_speech_offensive",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"hso-{index}",
                "prompt": row["tweet"],
                "unsafe": row["class"] in {0, 1},
                "class": row["class"],
                "hate_speech_count": row.get(
                    "hate_speech_count"
                ),
                "offensive_language_count": row.get(
                    "offensive_language_count"
                ),
                "neither_count": row.get("neither_count"),
            }
            for index, row in enumerate(rows, start=1)
            if str(row.get("tweet", "")).strip()
        ]
