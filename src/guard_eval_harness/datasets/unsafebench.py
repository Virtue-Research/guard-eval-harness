"""Multimodal UnsafeBench dataset adapter."""

from __future__ import annotations

import importlib

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REVISION = "9f4560ae90059237eb5eafc6bd8108c78639d180"


@dataset_registry.register("unsafebench")
class UnsafeBenchDataset(MultimodalDatasetAdapter):
    """Load UnsafeBench as image-only binary safety samples."""

    display_name = "UnsafeBench"
    source_uri = "https://huggingface.co/datasets/yiting/UnsafeBench"
    license_name = "Research DUA / gated"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = (
        "safety_label",
        "category",
        "source",
        "text",
    )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured UnsafeBench split."""
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"unsafebench supports splits: {supported}"
            )
        self._source_metadata = {
            "display_name": self.display_name,
            "source_uri": self.source_uri,
            "license": self.license_name,
            "languages": self.languages,
            "categories": (
                "Hate",
                "Harassment",
                "Violence",
                "Self-Harm",
                "Sexual",
                "Shocking",
                "Illegal Activity",
                "Deception",
                "Political",
                "Public and Personal Health",
                "Spam",
            ),
        }

        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "unsafebench",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        rows = self._load_rows()
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("image")
            if image is None:
                continue
            safety_label = str(row.get("safety_label", "")).strip()
            category = str(row.get("category", "")).strip()
            image_ref = self.resolve_image(image)
            message = self.build_multimodal_message(image_ref=image_ref)
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "safety_label": safety_label,
                    "category": category,
                    "source": row.get("source"),
                    "text": row.get("text"),
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
                    label=UnsafeLabel(unsafe=safety_label != "Safe"),
                    category_labels=(category,) if category else (),
                    metadata={
                        "safety_label": safety_label,
                        "category": category,
                        "source": row.get("source"),
                        "text": row.get("text"),
                    },
                )
            )
        return normalized

    def _load_rows(self) -> list[dict[str, object]]:
        """Load rows directly so embedded images stay as PIL objects."""
        try:
            datasets = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "unsafebench requires the 'datasets' package"
            ) from exc

        split = self.config.split
        if self.execution_limit is not None:
            split = f"{split}[:{self.execution_limit}]"
        dataset = datasets.load_dataset(
            "yiting/UnsafeBench",
            split=split,
            revision=_REVISION,
        )
        if hasattr(dataset, "__iter__"):
            return [dict(row) for row in dataset]
        return []
