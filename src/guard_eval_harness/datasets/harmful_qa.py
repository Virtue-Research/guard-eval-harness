"""Source-backed HarmfulQA dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"


@dataset_registry.register("harmful_qa")
class HarmfulQADataset(SourceBackedDatasetAdapter):
    """Load HarmfulQA conversations from Hugging Face."""

    display_name = "HarmfulQA"
    source_uri = "https://huggingface.co/datasets/declare-lab/HarmfulQA"
    license_name = "Apache 2.0"
    languages = ("en",)
    metadata_fields_to_preserve = ("topic", "subtopic", "target_role")
    label_mapping_note = (
        "blue conversations are safe and red conversations are unsafe"
    )

    def _normalize_conversation(
        self,
        conversation: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Convert HarmfulQA turns into the harness message schema."""
        messages: list[dict[str, str]] = []
        for turn in conversation:
            content = str(turn["value"]).strip()
            if not content:
                continue
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": content})
        return messages

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load blue and red conversation variants as one benchmark split."""
        rows = load_hf_rows(
            "declare-lab/HarmfulQA",
            split="train",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, object]] = []
        for row in rows:
            for color, unsafe in (("blue_conversations", False), ("red_conversations", True)):
                conversations = row[color]
                for key, conversation in conversations.items():
                    if conversation is None:
                        continue
                    messages = self._normalize_conversation(conversation)
                    if not messages:
                        continue
                    normalized_rows.append(
                        {
                            "id": f"{row['id']}-{color}-{key}",
                            "messages": messages,
                            "unsafe": unsafe,
                            "topic": row["topic"],
                            "subtopic": row["subtopic"],
                            "target_role": "assistant",
                        }
                    )
        return normalized_rows
