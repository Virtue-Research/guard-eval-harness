"""Source-backed HateXplain dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


def _majority_label(
    annotator_labels: list[int],
) -> str:
    """Compute majority vote over annotator label IDs."""
    if not annotator_labels:
        return "normal"
    counts: dict[int, int] = {}
    for label in annotator_labels:
        counts[label] = counts.get(label, 0) + 1
    return {0: "hatespeech", 1: "normal", 2: "offensive"}.get(
        max(counts, key=counts.get),  # type: ignore[arg-type]
        "normal",
    )


def _label_balanced_rows(
    rows: list[dict[str, object]],
    *,
    execution_limit: int | None,
) -> list[dict[str, object]]:
    """Balance small smoke slices across unsafe and safe."""
    if execution_limit is None:
        return rows

    unsafe = [r for r in rows if r.get("unsafe")]
    safe = [r for r in rows if not r.get("unsafe")]
    unsafe_limit = execution_limit - (execution_limit // 2)
    safe_limit = execution_limit // 2
    return (unsafe[:unsafe_limit] + safe[:safe_limit])[
        :execution_limit
    ]


@dataset_registry.register("hatexplain")
class HateXplainDataset(DatasetAdapter):
    """Load the HateXplain hate-speech dataset from HF."""

    display_name = "HateXplain"
    source_uri = (
        "https://huggingface.co/datasets/hatexplain"
    )
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = ("hate", "offensive")
    metadata_fields_to_preserve = ("majority_label",)
    label_mapping_note = (
        "majority-vote hatespeech or offensive maps to unsafe"
    )
    supported_splits = ("train", "validation", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load HateXplain rows from Hugging Face."""
        rows = load_hf_rows(
            "hatexplain",
            split=self.config.split,
            revision=_REVISION,
            limit=(
                self.execution_limit * 4
                if self.execution_limit
                else None
            ),
        )
        raw_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            tokens = row.get("post_tokens", [])
            text = " ".join(str(t) for t in tokens)
            if not text.strip():
                continue
            annotators = row.get("annotators", {})
            if isinstance(annotators, dict):
                # HF columnar format: {"label": [...], ...}
                annotator_labels = annotators.get(
                    "label", []
                )
            elif isinstance(annotators, list):
                # List-of-dicts format
                annotator_labels = [
                    a.get("label", 1) for a in annotators
                ]
            else:
                annotator_labels = []
            label = _majority_label(annotator_labels)
            is_unsafe = label in {"hatespeech", "offensive"}
            raw_rows.append(
                {
                    "id": (
                        f"hatexplain-{self.config.split}"
                        f"-{index}"
                    ),
                    "prompt": text,
                    "unsafe": is_unsafe,
                    "majority_label": label,
                }
            )
        return _label_balanced_rows(
            raw_rows, execution_limit=self.execution_limit
        )
