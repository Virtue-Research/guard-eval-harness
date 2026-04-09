"""Source-backed DecodingTrust Stereotypes dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("decodingtrust_stereotypes")
class DecodingTrustStereotypesDataset(SourceBackedDatasetAdapter):
    """Load the DecodingTrust stereotype bias evaluation set.

    Each row presents a stereotypical statement and asks
    the model to agree or disagree.  All statements are
    stereotypical (unsafe).  The ``sys_prompt_type_tag``
    indicates the system-prompt scenario: ``benign``,
    ``untargeted``, or ``targeted``.
    """

    display_name = "DecodingTrust Stereotypes"
    source_uri = (
        "https://huggingface.co/datasets/"
        "AI-Secure/DecodingTrust"
    )
    languages = ("en",)
    categories = (
        "Stereotype Bias",
    )
    metadata_fields_to_preserve = (
        "stereotype_topic_tag",
        "demographic_group_tag",
        "sys_prompt_type_tag",
    )
    label_mapping_note = (
        "all DecodingTrust stereotype prompts are unsafe "
        "(the model should disagree with the statement)"
    )
    supported_splits = ("stereotype",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the DecodingTrust stereotype subset."""
        rows = load_hf_rows(
            "AI-Secure/DecodingTrust",
            subset="stereotype",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            prompt_data = row.get("prompt", {})
            if isinstance(prompt_data, dict):
                text = str(
                    prompt_data.get("text", "")
                ).strip()
                topic = prompt_data.get(
                    "stereotype_topic_tag"
                )
                group = prompt_data.get(
                    "demographic_group_tag"
                )
                sys_type = prompt_data.get(
                    "sys_prompt_type_tag"
                )
            else:
                text = str(prompt_data).strip()
                topic = None
                group = None
                sys_type = None

            if not text:
                continue

            sys_filter = self.config.options.get(
                "sys_prompt_type"
            )
            if (
                sys_filter is not None
                and sys_type != sys_filter
            ):
                continue

            normalized.append(
                {
                    "id": (
                        f"decodingtrust-stereo-{index:05d}"
                    ),
                    "prompt": text,
                    "unsafe": True,
                    "stereotype_topic_tag": topic,
                    "demographic_group_tag": group,
                    "sys_prompt_type_tag": sys_type,
                }
            )
        return normalized
