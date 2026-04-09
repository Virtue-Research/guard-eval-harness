"""Source-backed Anthropic HH-RLHF Harmless adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


def _parse_conversation(
    raw: str,
) -> list[dict[str, str]]:
    """Parse the Anthropic HH-RLHF conversation format."""
    messages: list[dict[str, str]] = []
    for segment in raw.split("\n\n"):
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("Human:"):
            content = segment[len("Human:"):].strip()
            if content:
                messages.append(
                    {"role": "user", "content": content}
                )
        elif segment.startswith("Assistant:"):
            content = segment[len("Assistant:"):].strip()
            if content:
                messages.append(
                    {"role": "assistant", "content": content}
                )
    return messages


@dataset_registry.register("anthropic_hh_rlhf")
class AnthropicHHRLHFDataset(SourceBackedDatasetAdapter):
    """Load Anthropic HH-RLHF harmless subset from HF."""

    display_name = "Anthropic HH-RLHF Harmless"
    source_uri = (
        "https://huggingface.co/datasets/Anthropic/hh-rlhf"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = ("harmlessness", "refusal")
    metadata_fields_to_preserve = ("target_role",)
    label_mapping_note = (
        "the rejected response is treated as unsafe and the "
        "chosen response as safe"
    )
    supported_splits = ("train", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load chosen/rejected pairs as safe/unsafe rows."""
        rows = load_hf_rows(
            "Anthropic/hh-rlhf",
            split=self.config.split,
            data_dir="harmless-base",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            chosen_msgs = _parse_conversation(str(chosen))
            rejected_msgs = _parse_conversation(str(rejected))
            if chosen_msgs:
                normalized.append(
                    {
                        "id": (
                            f"hh-rlhf-{self.config.split}"
                            f"-{index}-chosen"
                        ),
                        "messages": chosen_msgs,
                        "unsafe": False,
                        "target_role": "assistant",
                    }
                )
            if rejected_msgs:
                normalized.append(
                    {
                        "id": (
                            f"hh-rlhf-{self.config.split}"
                            f"-{index}-rejected"
                        ),
                        "messages": rejected_msgs,
                        "unsafe": True,
                        "target_role": "assistant",
                    }
                )
        return normalized
