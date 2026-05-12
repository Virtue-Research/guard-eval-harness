"""Source-backed OpenAI Moderation evaluation dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "84e5cf3bcd6acb3dfc70b6760451645872218a3e"
_CATEGORY_COLUMNS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")

_CATEGORY_LABELS = {
    "S": "Sexual",
    "H": "Hate",
    "V": "Violence",
    "HR": "Harassment",
    "SH": "Self-Harm",
    "S3": "Sexual/Minors",
    "H2": "Hate/Threatening",
    "V2": "Violence/Graphic",
}


@dataset_registry.register("openai_moderation_eval")
class OpenAIModerationEvalDataset(SourceBackedDatasetAdapter):
    """Load the full OpenAI Moderation API evaluation dataset."""

    display_name = "OpenAI Moderation Eval"
    source_uri = (
        "https://huggingface.co/datasets/"
        "mmathys/openai-moderation-api-evaluation"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = tuple(_CATEGORY_LABELS.values())
    metadata_fields_to_preserve = ()
    label_mapping_note = (
        "a prompt is unsafe if any moderation category column "
        "is 1; rows where all category columns are null are "
        "skipped as unlabeled"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the full OpenAI moderation eval set."""
        rows = load_hf_rows(
            "mmathys/openai-moderation-api-evaluation",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            labels = {
                col: row.get(col)
                for col in _CATEGORY_COLUMNS
            }
            known = {
                col: val
                for col, val in labels.items()
                if val is not None
            }
            if not known:
                continue
            normalized.append(
                {
                    "id": f"openai-mod-eval-{index}",
                    "prompt": prompt,
                    "unsafe": any(
                        v in {1, "1", True}
                        for v in known.values()
                    ),
                    **labels,
                }
            )
        return normalized
