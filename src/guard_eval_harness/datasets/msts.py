"""MSTS multimodal safety benchmark adapter."""

from typing import Any

from guard_eval_harness.datasets.image_base import (
    ImageDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REVISION = "9630ca3cb6251749ca52fee51e68a21943b23067"
_LANGUAGE_SPLITS = (
    "english",
    "german",
    "russian",
    "chinese",
    "hindi",
    "spanish",
    "italian",
    "french",
    "korean",
    "arabic",
    "farsi",
)


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


@dataset_registry.register("msts")
class MSTSDataset(ImageDatasetAdapter):
    """Load MSTS as text+image unsafe samples."""

    display_name = "MSTS"
    source_uri = "https://huggingface.co/datasets/felfri/MSTS"
    license_name = "CC-BY-4.0"
    version = _REVISION
    access_mode = "public"
    languages = _LANGUAGE_SPLITS
    supported_splits = _LANGUAGE_SPLITS
    metadata_fields_to_preserve = (
        "case_id",
        "case_text",
        "prompt_text",
        "prompt_type",
        "hazard_category",
        "hazard_subcategory",
        "hazard_subsubcategory",
        "unsafe_image_id",
        "unsafe_image_url",
        "unsafe_image_license",
        "unsafe_image_cw",
    )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize one MSTS language split."""
        self._ensure_supported_split()
        self._source_metadata = self._build_source_metadata()
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "msts",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_hf_rows(
            "felfri/MSTS",
            split=self.config.split,
            revision=_REVISION,
        )
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("unsafe_image")
            if image is None:
                continue
            image_ref = self.resolve_image(image)
            hazard_category = _clean_text(row.get("hazard_category"))
            prompt_text = _clean_text(row.get("prompt_text")) or _clean_text(
                row.get("case_text")
            )
            sample_id = self._make_sample_id(
                {
                    "case_id": row.get("case_id"),
                    "split": self.config.split,
                    "hazard_category": hazard_category,
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
                            text=prompt_text,
                            image_ref=image_ref,
                        )
                    ],
                    label=UnsafeLabel(unsafe=True),
                    category_labels=(
                        (hazard_category,) if hazard_category else ()
                    ),
                    metadata={
                        "case_id": row.get("case_id"),
                        "case_text": row.get("case_text"),
                        "prompt_text": row.get("prompt_text"),
                        "prompt_type": row.get("prompt_type"),
                        "hazard_category": row.get("hazard_category"),
                        "hazard_subcategory": row.get("hazard_subcategory"),
                        "hazard_subsubcategory": row.get(
                            "hazard_subsubcategory"
                        ),
                        "unsafe_image_id": row.get("unsafe_image_id"),
                        "unsafe_image_url": row.get("unsafe_image_url"),
                        "unsafe_image_license": row.get("unsafe_image_license"),
                        "unsafe_image_cw": row.get("unsafe_image_cw"),
                    },
                )
            )
        return normalized
