"""Parveshiiii AI-vs-Real dataset adapter."""

from __future__ import annotations

import logging

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REPO = "Parveshiiii/AI-vs-Real"
_REVISION = "bce7ac5b95c36c5013389341b94c75aa44882165"
_log = logging.getLogger(__name__)


@dataset_registry.register("ai_vs_real")
class AIVsRealDataset(SourceBackedMultimodalDatasetAdapter):
    """Load AI-generated vs real images as GenAI moderation samples."""

    display_name = "AI-vs-Real"
    source_uri = f"https://huggingface.co/datasets/{_REPO}"
    license_name = "MIT"
    categories = ("genai/deepfakes",)
    supported_splits = ("train",)
    version = _REVISION
    metadata_fields_to_preserve = ()

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured split."""
        self._ensure_supported_split()
        self._source_metadata = {
            **self._build_source_metadata(),
            "label_mapping": "binary_label=0 -> unsafe AI-generated image; binary_label=1 -> safe real image.",
            "hf_repo_id": _REPO,
            "hf_revision": _REVISION,
        }
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "ai_vs_real",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_hf_rows(
            _REPO,
            split=self.config.split,
            revision=_REVISION,
            verification_mode="no_checks",
            image_decode=False,
        )
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("image")
            if image is None:
                continue
            binary_label = int(row.get("binary_label", 1))
            image_ref = self.resolve_image(image)
            unsafe = binary_label == 0
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "binary_label": binary_label,
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
                        self.build_multimodal_message(image_ref=image_ref)
                    ],
                    label=UnsafeLabel(unsafe=unsafe),
                    category_labels=("genai/deepfakes",) if unsafe else (),
                    metadata={
                        "image_sha256": image_ref.sha256,
                    },
                )
            )
            if len(normalized) % 500 == 0:
                _log.info(
                    "Materialized %d ai_vs_real samples",
                    len(normalized),
                )
        _log.info("Materialized %d ai_vs_real samples", len(normalized))
        return normalized
