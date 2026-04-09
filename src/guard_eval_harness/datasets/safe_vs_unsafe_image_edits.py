"""Public paired safe/unsafe image edits dataset adapter."""

from __future__ import annotations

import importlib
from typing import Any

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REVISION_BATCH1 = "6ac3e4c0b25845ef181f51eee169d2ddda6d360b"
_REVISION_BATCH2 = "e67051c147bc3613dee9bba7e63aa0ce6526bd3a"
_REPO_BY_VARIANT = {
    "batch1": ("Advait-s06/safe-vs-unsafe-image-edits", _REVISION_BATCH1),
    "batch2": (
        "Advait-s06/safe-vs-unsafe-image-edits-batch2",
        _REVISION_BATCH2,
    ),
}


@dataset_registry.register("safe_vs_unsafe_image_edits")
class SafeVsUnsafeImageEditsDataset(MultimodalDatasetAdapter):
    """Load public paired safe/unsafe image edits as image-only samples."""

    display_name = "Safe vs Unsafe Image Edits"
    source_uri = "https://huggingface.co/datasets/Advait-s06/safe-vs-unsafe-image-edits"
    supported_splits = ("train",)

    def _variant(self) -> str:
        """Return which public pair dataset variant to use."""
        variant = str(
            self.config.options.get("variant", "batch1")
        ).strip().lower()
        if variant not in _REPO_BY_VARIANT:
            supported = ", ".join(sorted(_REPO_BY_VARIANT))
            raise ValueError(
                "safe_vs_unsafe_image_edits supports options.variant values: "
                f"{supported}"
            )
        return variant

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured paired image dataset."""
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                "safe_vs_unsafe_image_edits supports splits: "
                f"{supported}"
            )

        variant = self._variant()
        repo_id, revision = _REPO_BY_VARIANT[variant]
        self._source_metadata = {
            "display_name": self.display_name,
            "source_uri": f"https://huggingface.co/datasets/{repo_id}",
            "languages": ("en",),
            "categories": ("unsafe_edit",),
            "version": revision,
        }

        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "safe_vs_unsafe_image_edits",
                "variant": variant,
                "revision": revision,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=lambda: self._load_uncached(
                repo_id=repo_id,
                revision=revision,
                variant=variant,
            ),
        )

    def _load_uncached(
        self,
        *,
        repo_id: str,
        revision: str,
        variant: str,
    ) -> list[NormalizedSample]:
        rows = self._load_rows(repo_id=repo_id, revision=revision)
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            normalized.extend(
                self._paired_rows(
                    row,
                    row_index=row_index,
                    variant=variant,
                )
            )
        return normalized

    def _load_rows(
        self,
        *,
        repo_id: str,
        revision: str,
    ) -> list[dict[str, Any]]:
        """Load rows directly so embedded images stay as PIL objects."""
        try:
            datasets = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "safe_vs_unsafe_image_edits requires the 'datasets' package"
            ) from exc

        split = self.config.split
        if self.execution_limit is not None:
            split = f"{split}[:{self.execution_limit}]"
        dataset = datasets.load_dataset(
            repo_id,
            split=split,
            revision=revision,
            trust_remote_code=True,
        )
        if hasattr(dataset, "__iter__"):
            return [dict(row) for row in dataset]
        return []

    def _paired_rows(
        self,
        row: dict[str, Any],
        *,
        row_index: int,
        variant: str,
    ) -> list[NormalizedSample]:
        """Emit one safe sample and one unsafe sample from a paired row."""
        samples: list[NormalizedSample] = []
        for order, image_key, caption_key, unsafe in (
            (1, "safe_image", "safe_caption", False),
            (2, "unsafe_image", "unsafe_caption", True),
        ):
            image = row.get(image_key)
            if image is None:
                continue
            caption = str(row.get(caption_key, "")).strip()
            image_ref = self.resolve_image(image)
            message = self.build_multimodal_message(image_ref=image_ref)
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "order": order,
                    "variant": variant,
                    "caption": caption,
                    "image_sha256": image_ref.sha256,
                },
                row_index * 2 - (2 - order),
            )
            samples.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=[message],
                    label=UnsafeLabel(unsafe=unsafe),
                    metadata={
                        "caption": caption,
                        "pair_variant": variant,
                        "source_role": "unsafe" if unsafe else "safe",
                    },
                )
            )
        return samples
