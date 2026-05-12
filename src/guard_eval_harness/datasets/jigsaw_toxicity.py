"""Source-backed Jigsaw Toxicity dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "dd6e43f4b089378aa82439da98150c400675e863"
_LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


@dataset_registry.register("jigsaw_toxicity")
class JigsawToxicityDataset(SourceBackedDatasetAdapter):
    """Load the Jigsaw Toxic Comment Classification dataset."""

    display_name = "Jigsaw Toxicity"
    source_uri = (
        "https://huggingface.co/datasets/"
        "Arsive/toxicity_classification_jigsaw"
    )
    license_name = "CC0 1.0"
    languages = ("en",)
    categories = (
        "Toxic",
        "Severe Toxic",
        "Obscene",
        "Threat",
        "Insult",
        "Identity Hate",
    )
    metadata_fields_to_preserve = ()
    label_mapping_note = (
        "a comment is unsafe if any of the six toxicity "
        "label columns is 1; test rows with label -1 are "
        "skipped as unlabeled"
    )
    supported_splits = ("train", "validation", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured Jigsaw toxicity split."""
        rows = load_hf_rows(
            "Arsive/toxicity_classification_jigsaw",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            text = str(row.get("comment_text", "")).strip()
            if not text:
                continue
            label_vals = [row.get(col) for col in _LABEL_COLUMNS]
            if any(v == -1 for v in label_vals):
                continue
            normalized.append(
                {
                    "id": str(
                        row.get("id", f"jigsaw-{index}")
                    ),
                    "prompt": text,
                    "unsafe": any(
                        v in {1, "1", True}
                        for v in label_vals
                    ),
                    **{
                        col: row.get(col)
                        for col in _LABEL_COLUMNS
                    },
                }
            )
        return normalized
