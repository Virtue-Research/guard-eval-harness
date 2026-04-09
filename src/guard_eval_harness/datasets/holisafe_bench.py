"""HoliSafe-Bench multimodal safety benchmark adapter."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import (
    NormalizedSample,
    UnsafeLabel,
)

_REVISION = "main"

_RISK_TYPE_SAFE = {"SSS"}


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


def _is_unsafe_by_risk_type(risk_type: str) -> bool:
    """Determine ground-truth unsafe label from risk type.

    The three-character risk type encodes
    ``[image][query][final_input]`` safety.  Only ``SSS``
    (safe image + safe query + safe combined) is considered
    safe; all other combinations are unsafe.
    """
    return risk_type.upper() not in _RISK_TYPE_SAFE


@dataset_registry.register("holisafe_bench")
class HoliSafeBenchDataset(
    SourceBackedMultimodalDatasetAdapter,
):
    """Load HoliSafe-Bench as text+image safety samples."""

    display_name = "HoliSafe-Bench"
    source_uri = (
        "https://huggingface.co/datasets/"
        "etri-vilab/holisafe-bench"
    )
    license_name = "CC-BY-NC-4.0"
    version = _REVISION
    access_mode = "gated"
    languages = ("en",)
    supported_splits = ("test",)
    categories = (
        "hate",
        "violence",
        "illegal_activity",
        "specialized_advice",
        "self_harm",
        "privacy",
        "sexual_content",
    )
    metadata_fields_to_preserve = (
        "id",
        "category",
        "subcategory",
        "type",
        "image_safe",
        "image_safety_label",
    )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize HoliSafe-Bench."""
        self._ensure_supported_split()
        self._source_metadata = (
            self._build_source_metadata()
        )
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "holisafe_bench",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_hf_rows(
            "etri-vilab/holisafe-bench",
            split=self.config.split,
            revision=_REVISION,
        )
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("image")
            if image is None:
                continue
            image_ref = self.resolve_image(image)
            category = _clean_text(row.get("category"))
            subcategory = _clean_text(
                row.get("subcategory")
            )
            query = _clean_text(row.get("query"))
            risk_type = _clean_text(row.get("type"))
            unsafe = (
                _is_unsafe_by_risk_type(risk_type)
                if risk_type
                else True
            )
            sample_id = self._make_sample_id(
                {
                    "id": row.get("id"),
                    "category": category,
                    "subcategory": subcategory,
                    "image_sha256": image_ref.sha256,
                },
                row_index,
            )
            normalized.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=[
                        self.build_multimodal_message(
                            text=query,
                            image_ref=image_ref,
                        )
                    ],
                    label=UnsafeLabel(unsafe=unsafe),
                    category_labels=(
                        (category,)
                        if category
                        else ()
                    ),
                    metadata={
                        "id": row.get("id"),
                        "category": row.get("category"),
                        "subcategory": row.get(
                            "subcategory"
                        ),
                        "type": row.get("type"),
                        "image_safe": row.get(
                            "image_safe"
                        ),
                        "image_safety_label": row.get(
                            "image_safety_label"
                        ),
                    },
                )
            )
        return normalized
