"""Yiting Self-harm_Dataset adapter."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REPO = "yiting/Self-harm_Dataset"
_REVISION = "7b73824c8d63f76e1f258ab2b176a3d7507cc418"


@dataset_registry.register("self_harm_image_dataset")
class SelfHarmImageDataset(SourceBackedMultimodalDatasetAdapter):
    """Load the public self-harm image classification dataset."""

    display_name = "Self-harm Image Dataset"
    source_uri = f"https://huggingface.co/datasets/{_REPO}"
    license_name = "Unknown"
    categories = ("blood/gore",)
    supported_splits = ("train",)
    version = _REVISION
    metadata_fields_to_preserve = ("index", "image_fname", "label")

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured split."""
        self._ensure_supported_split()
        self._source_metadata = {
            **self._build_source_metadata(),
            "label_mapping": "label=1 -> unsafe self-harm image; label=0 -> safe/benign.",
            "hf_repo_id": _REPO,
            "hf_revision": _REVISION,
        }
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "self_harm_image_dataset",
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
        )
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            encoded_image = row.get("image")
            if not isinstance(encoded_image, str) or not encoded_image.strip():
                continue
            label = int(row.get("label", 0))
            image_ref = self.resolve_image(_decode_base64_image(encoded_image))
            unsafe = label == 1
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "image_fname": row.get("image_fname"),
                    "label": label,
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
                    category_labels=("blood/gore",) if unsafe else (),
                    metadata={
                        "index": row.get("index"),
                        "image_fname": row.get("image_fname"),
                        "label": label,
                        "image_sha256": image_ref.sha256,
                    },
                )
            )
        return normalized


def _decode_base64_image(encoded: str) -> Any:
    """Decode a base64 image string into a PIL image."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "self_harm_image_dataset requires Pillow for image decoding"
        ) from exc

    payload = encoded.strip()
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    payload += "=" * (-len(payload) % 4)
    image_bytes = base64.b64decode(payload)
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB").copy()
