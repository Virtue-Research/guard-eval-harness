"""Source-backed Toxic Chat dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_SPLIT_MAP = {"train": "train", "test": "test"}
_REVISION = "29df8e4dba60e1f4af4b4075c0705c5b313548a8"


@dataset_registry.register("toxic_chat")
class ToxicChatDataset(DatasetAdapter):
    """Load the Toxic Chat dataset from Hugging Face."""

    display_name = "Toxic Chat"
    source_uri = "https://huggingface.co/datasets/lmsys/toxic-chat"
    license_name = "CC BY-NC 4.0"
    languages = ("en",)
    categories = ("Offensive", "Hate", "Other")
    metadata_fields_to_preserve = ("human_annotation", "jailbreaking")
    label_mapping_note = "toxicity is mapped directly to the unsafe label"
    supported_splits = ("train", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured Toxic Chat split."""
        rows = load_hf_rows(
            "lmsys/toxic-chat",
            subset="toxicchat0124",
            split=_SPLIT_MAP[self.config.split],
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": str(row["conv_id"]),
                "prompt": row["user_input"],
                "unsafe": row["toxicity"] in {1, "1", True},
                "human_annotation": row["human_annotation"],
                "jailbreaking": row["jailbreaking"],
            }
            for row in rows
        ]
