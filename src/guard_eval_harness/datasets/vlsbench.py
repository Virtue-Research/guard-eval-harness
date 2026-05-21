"""VLSBench multimodal safety benchmark adapter."""

from typing import Any

from guard_eval_harness.datasets.image_base import (
    ImageDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REVISION = "b56f6f6aad102fdb53f46e35fec96836bbe13001"


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


@dataset_registry.register("vlsbench")
class VLSBenchDataset(ImageDatasetAdapter):
    """Load VLSBench as text+image unsafe samples."""

    display_name = "VLSBench"
    source_uri = "https://huggingface.co/datasets/Foreshhh/vlsbench"
    license_name = "Apache-2.0"
    version = _REVISION
    access_mode = "public"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = (
        "instruction_id",
        "instruction",
        "image_path",
        "category",
        "sub_category",
        "source",
        "image_description",
        "safety_reason",
    )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize VLSBench."""
        self._ensure_supported_split()
        self._source_metadata = self._build_source_metadata()
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "vlsbench",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_hf_rows(
            "Foreshhh/vlsbench",
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
            sample_id = self._make_sample_id(
                {
                    "instruction_id": row.get("instruction_id"),
                    "category": category,
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
                            text=_clean_text(row.get("instruction")),
                            image_ref=image_ref,
                        )
                    ],
                    label=UnsafeLabel(unsafe=True),
                    category_labels=(category,) if category else (),
                    metadata={
                        "instruction_id": row.get("instruction_id"),
                        "instruction": row.get("instruction"),
                        "image_path": row.get("image_path"),
                        "category": row.get("category"),
                        "sub_category": row.get("sub_category"),
                        "source": row.get("source"),
                        "image_description": row.get("image_description"),
                        "safety_reason": row.get("safety_reason"),
                    },
                )
            )
        return normalized
