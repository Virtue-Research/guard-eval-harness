"""Helpers for source-backed built-in datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
from urllib import request as urllib_request

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.schemas import DatasetMetadata, NormalizedSample


def _cache_root() -> Path:
    """Return the shared dataset cache root."""
    return Path.home() / ".cache" / "guard-eval-harness" / "datasets"


_DOWNLOAD_TIMEOUT = 60.0


def cached_download(*, alias: str, url: str, filename: str) -> Path:
    """Download a remote file into the dataset cache once."""
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    destination = _cache_root() / alias / url_hash / filename
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib_request.urlopen(
        url, timeout=_DOWNLOAD_TIMEOUT
    ) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def load_csv_rows(
    *,
    alias: str,
    url: str,
    filename: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Download and parse a CSV dataset source."""
    path = cached_download(alias=alias, url=url, filename=filename)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if limit is None:
            return [dict(row) for row in reader]
        return [dict(row) for row in islice(reader, limit)]


def load_json_payload(
    *,
    alias: str,
    url: str,
    filename: str,
) -> Any:
    """Download and parse a JSON dataset source."""
    path = cached_download(alias=alias, url=url, filename=filename)
    return json.loads(path.read_text(encoding="utf-8"))


def load_text_lines(
    *,
    alias: str,
    url: str,
    filename: str,
    limit: int | None = None,
) -> list[str]:
    """Download and parse a newline-delimited text dataset source."""
    path = cached_download(alias=alias, url=url, filename=filename)
    with path.open("r", encoding="utf-8") as handle:
        lines = (line.strip() for line in handle)
        filtered_lines = (line for line in lines if line)
        if limit is None:
            return list(filtered_lines)
        return list(islice(filtered_lines, limit))


def load_hf_rows(
    repo_id: str,
    *,
    split: str,
    subset: str | None = None,
    revision: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load a split from Hugging Face datasets as plain dictionaries."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "source-backed datasets require the 'datasets' package"
        ) from exc

    split_name = split
    if limit is not None:
        split_name = f"{split}[{offset}:{offset + limit}]"
    elif offset:
        split_name = f"{split}[{offset}:]"

    kwargs: dict[str, Any] = {
        "split": split_name,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    dataset = load_dataset(
        repo_id,
        subset,
        **kwargs,
    )
    if hasattr(dataset, "to_list"):
        return list(dataset.to_list())
    return [dict(row) for row in dataset]


class SourceBackedDatasetAdapter(DatasetAdapter):
    """Base adapter for built-ins backed by real public sources."""

    display_name: str = ""
    source_uri: str | None = None
    license_name: str | None = None
    languages: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    metric_eligibility: dict[str, bool] = {
        "binary_classification": True,
        "accuracy": True,
        "precision": True,
        "recall": True,
        "f1": True,
        "fpr": True,
        "fnr": True,
    }
    metadata_fields_to_preserve: tuple[str, ...] = ()
    label_mapping_note: str | None = None
    supported_splits: tuple[str, ...] = ("test",)
    version: str | None = None

    @classmethod
    def from_config(cls, config):
        """Merge dataset-specific metadata fields into the config."""
        merged_fields = tuple(
            dict.fromkeys(
                (*config.metadata_fields, *cls.metadata_fields_to_preserve)
            )
        )
        return cls(config.model_copy(update={"metadata_fields": merged_fields}))

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured split from its source."""
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"{self.config.adapter} supports splits: {supported}"
            )
        self._source_metadata = self._build_source_metadata()
        return self._finalize_samples(self.load_source_rows())

    def load_source_rows(self) -> Iterable[dict[str, Any]]:
        """Return normalized row mappings from the upstream source."""
        raise NotImplementedError

    def _build_source_metadata(self) -> dict[str, Any]:
        """Build source metadata consumed by ``describe``."""
        metadata: dict[str, Any] = {
            "display_name": self.display_name or self.config.name,
            "languages": self.languages,
            "categories": self.categories,
            "metric_eligibility": self.metric_eligibility,
        }
        if self.source_uri is not None:
            metadata["source_uri"] = self.source_uri
        if self.license_name is not None:
            metadata["license"] = self.license_name
        if self.version is not None:
            metadata["version"] = self.version
        return metadata

    def describe(self, samples: list[NormalizedSample]) -> DatasetMetadata:
        """Attach stable label-mapping notes to source-backed datasets."""
        metadata = super().describe(samples)
        extra_metadata = dict(metadata.metadata)
        if self.label_mapping_note:
            extra_metadata["label_mapping"] = self.label_mapping_note
        return metadata.model_copy(update={"metadata": extra_metadata})
