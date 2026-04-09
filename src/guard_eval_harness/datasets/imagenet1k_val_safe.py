"""ImageNet 1K validation set adapter (all-safe image-only)."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.datasets.sample_cache import (
    compute_cache_key,
    load_cached_samples,
    sample_cache_base_dir,
    write_sample_cache,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel

_log = logging.getLogger(__name__)

_REPO = "mrm8488/ImageNet1K-val"
_REVISION = "main"


@dataset_registry.register("imagenet1k_val_safe")
class ImageNet1KValSafeDataset(MultimodalDatasetAdapter):
    """Load ImageNet 1K validation as image-only safe samples."""

    display_name = "ImageNet 1K (Safe)"
    source_uri = f"https://huggingface.co/datasets/{_REPO}"
    license_name = "ImageNet License"
    languages: tuple[str, ...] = ()
    supported_splits = ("train",)

    def load(self) -> list[NormalizedSample]:
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"imagenet1k_val_safe supports splits: {supported}"
            )
        self._source_metadata = {
            "display_name": self.display_name,
            "source_uri": self.source_uri,
            "license": self.license_name,
        }

        use_cache = not self.config.options.get(
            "no_sample_cache"
        )
        cache_key_parts = {
            "adapter": "imagenet1k_val_safe",
            "dataset_name": self.config.name,
            "revision": _REVISION,
            "split": self.config.split,
            "execution_limit": self.execution_limit,
        }

        if use_cache:
            cache_dir = Path(
                self.config.options.get(
                    "sample_cache_dir",
                    str(sample_cache_base_dir()),
                )
            )
            key = compute_cache_key(cache_key_parts)
            cached = load_cached_samples(
                cache_dir, "imagenet1k_val_safe", key
            )
            if cached is not None:
                _log.info(
                    "Loaded %d cached samples for "
                    "imagenet1k_val_safe",
                    len(cached),
                )
                return cached

        samples = self._load_uncached()

        if use_cache:
            write_sample_cache(
                cache_dir,
                "imagenet1k_val_safe",
                key,
                samples,
            )

        return samples

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_rows()
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("image")
            if image is None:
                continue
            class_id = row.get("label")
            image_ref = self.resolve_image(image)
            message = self.build_multimodal_message(
                image_ref=image_ref,
            )
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "label": class_id,
                    "image_sha256": image_ref.sha256,
                },
                row_index,
            )
            normalized.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=[message],
                    label=UnsafeLabel(unsafe=False),
                    category_labels=(),
                    metadata={"imagenet_class_id": class_id},
                )
            )
        return normalized

    def _load_rows(self) -> list[dict[str, object]]:
        try:
            datasets = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "imagenet1k_val_safe requires the 'datasets' package"
            ) from exc

        split = self.config.split
        if self.execution_limit is not None:
            split = f"{split}[:{self.execution_limit}]"
        dataset = datasets.load_dataset(
            _REPO,
            split=split,
            revision=_REVISION,
            trust_remote_code=True,
        )
        if hasattr(dataset, "__iter__"):
            return [dict(row) for row in dataset]
        return []
