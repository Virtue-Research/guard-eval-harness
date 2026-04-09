"""Shared helpers for source-backed multimodal datasets."""

from __future__ import annotations

import importlib
from typing import Any

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)


class SourceBackedMultimodalDatasetAdapter(MultimodalDatasetAdapter):
    """Base class for multimodal datasets backed by external sources."""

    display_name: str = ""
    source_uri: str | None = None
    license_name: str | None = None
    languages: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    metadata_fields_to_preserve: tuple[str, ...] = ()
    version: str | None = None
    access_mode: str | None = None
    upstream_images_uri: str | None = None
    supported_splits: tuple[str, ...] = ("test",)

    @classmethod
    def from_config(cls, config):
        """Merge dataset-specific metadata fields into the config."""
        merged_fields = tuple(
            dict.fromkeys(
                (*config.metadata_fields, *cls.metadata_fields_to_preserve)
            )
        )
        return cls(config.model_copy(update={"metadata_fields": merged_fields}))

    def _ensure_supported_split(self) -> None:
        """Reject unsupported split names early."""
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"{self.config.adapter} supports splits: {supported}"
            )

    def _build_source_metadata(self) -> dict[str, Any]:
        """Build stable source metadata for manifests and reports."""
        metadata: dict[str, Any] = {
            "display_name": self.display_name or self.config.name,
            "languages": self.languages,
            "categories": self.categories,
        }
        if self.source_uri is not None:
            metadata["source_uri"] = self.source_uri
        if self.license_name is not None:
            metadata["license"] = self.license_name
        if self.version is not None:
            metadata["version"] = self.version
        if self.access_mode is not None:
            metadata["access_mode"] = self.access_mode
        if self.upstream_images_uri is not None:
            metadata["upstream_images_uri"] = self.upstream_images_uri
        return metadata

    def _load_hf_rows(
        self,
        repo_id: str,
        *,
        split: str,
        subset: str | None = None,
        revision: str | None = None,
        data_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load one HF dataset split as plain dictionaries."""
        try:
            datasets = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"{self.config.adapter} requires the 'datasets' package"
            ) from exc

        split_name = split
        if self.execution_limit is not None:
            split_name = f"{split}[:{self.execution_limit}]"
        dataset = datasets.load_dataset(
            repo_id,
            subset,
            split=split_name,
            revision=revision,
            data_dir=data_dir,
            trust_remote_code=True,
        )
        if hasattr(dataset, "to_list"):
            return list(dataset.to_list())
        return [dict(row) for row in dataset]
