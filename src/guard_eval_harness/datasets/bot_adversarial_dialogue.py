"""Source-backed Bot-Adversarial Dialogue dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("bot_adversarial_dialogue")
class BotAdversarialDialogueDataset(SourceBackedDatasetAdapter):
    """Load the Bot-Adversarial Dialogue dataset from Hugging Face."""

    display_name = "Bot-Adversarial Dialogue"
    source_uri = (
        "https://huggingface.co/datasets/"
        "facebook/bot_adversarial_dialogue"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = ("dialog_safety",)
    metadata_fields_to_preserve = ("dialogue_id", "target_role")
    label_mapping_note = (
        "ClassLabel 1 (__notok__) maps to unsafe, 0 (__ok__) to safe"
    )
    supported_splits = ("train", "valid", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load dialog turns from Hugging Face."""
        rows = load_hf_rows(
            "facebook/bot_adversarial_dialogue",
            subset="dialogue_datasets",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        # labels is a ClassLabel: 0=__ok__, 1=__notok__
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            raw_label = row.get("labels", 0)
            if isinstance(raw_label, int):
                is_unsafe = raw_label == 1
            else:
                is_unsafe = str(raw_label).strip().lower() in {
                    "__notok__",
                    "1",
                }
            text = str(row.get("text", ""))
            dialogue_id = row.get("dialogue_id", index)
            normalized.append(
                {
                    "id": f"bad-{self.config.split}-{index}",
                    "prompt": text,
                    "unsafe": is_unsafe,
                    "dialogue_id": str(dialogue_id),
                    "target_role": "user",
                }
            )
        return normalized
