"""Source-backed ETHOS hate speech dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "3816a4232f05ebec284ffb76a2b6565d1c858e2a"


@dataset_registry.register("ethos")
class EthosDataset(SourceBackedDatasetAdapter):
    """Load the ETHOS binary hate speech detection dataset."""

    display_name = "ETHOS"
    source_uri = "https://huggingface.co/datasets/SetFit/ethos"
    license_name = "AGPL-3.0"
    languages = ("en",)
    categories = ("Hate Speech",)
    label_mapping_note = (
        "label 1 maps to unsafe (hate speech) and "
        "label 0 maps to safe (no hate speech)"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the ETHOS binary split from Hugging Face."""
        rows = load_hf_rows(
            "SetFit/ethos",
            subset="binary",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"ethos-{index}",
                "prompt": row["text"],
                "unsafe": row["label"] in {1, "1", True},
            }
            for index, row in enumerate(rows, start=1)
        ]
