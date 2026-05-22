"""Base dataset adapter — universal interface + source-backed defaults.

A ``DatasetAdapter`` normalizes a dataset into ``NormalizedSample`` rows.
By default it follows the *source-backed* flow: subclasses override
``load_source_rows()`` to return raw row mappings, and the default
``load()`` validates the split, builds source metadata, and runs the
normalization pipeline.

Local adapters that read files the user supplied on disk should
override ``load()`` directly and ignore ``load_source_rows()``.
"""

import csv
import hashlib
import json
from abc import ABC
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib import request as urllib_request

from guard_eval_harness.config import ResolvedDatasetConfig
from guard_eval_harness.schemas import (
    DatasetMetadata,
    Message,
    NormalizedSample,
    UnsafeLabel,
)


# ---------------------------------------------------------------------------
# Download helpers (shared cache, content-addressed per URL)
# ---------------------------------------------------------------------------


_DOWNLOAD_TIMEOUT = 60.0


def _cache_root() -> Path:
    """Return the shared dataset cache root."""
    return Path.home() / ".cache" / "guard-eval-harness" / "datasets"


def cached_download(*, alias: str, url: str, filename: str) -> Path:
    """Download a remote file into the dataset cache once."""
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    destination = _cache_root() / alias / url_hash / filename
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib_request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
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

    kwargs: dict[str, Any] = {"split": split_name}
    if revision is not None:
        kwargs["revision"] = revision
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    dataset = load_dataset(repo_id, subset, **kwargs)
    if hasattr(dataset, "to_list"):
        return list(dataset.to_list())
    return [dict(row) for row in dataset]


# ---------------------------------------------------------------------------
# DatasetAdapter — universal ABC + default source-backed flow
# ---------------------------------------------------------------------------


class DatasetAdapter(ABC):
    """Universal dataset adapter.

    Source-backed subclasses override ``load_source_rows()`` and let
    the default ``load()`` handle split validation + normalization.
    Local-file subclasses override ``load()`` directly.
    """

    # --- file-path adapters: which suffixes are allowed for `config.path` ---
    source_suffixes: tuple[str, ...] = ()

    # --- source-backed adapters: dataset-level metadata ---
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

    # --- policy source default ---
    # Tells the runner which kind of dataset-scoped policy to use when
    # the YAML doesn't override ``policy_source``. Built-in values:
    #   "upstream"       — paper's official taxonomy (Tier A datasets)
    #   "generated"      — deployment-style GPT-derived policy (most)
    #   "virtue_general" — global fallback (all-safe / no-taxonomy)
    # Users can register their own and reference it via YAML.
    default_policy_source: str = "generated"

    def __init__(self, config: ResolvedDatasetConfig) -> None:
        self.config = config
        self._source_metadata: dict[str, Any] = {}
        self._source_path: Path | None = None
        raw_limit = self.config.options.get("limit")
        self.execution_limit: int | None = None
        if raw_limit is not None:
            self.execution_limit = int(raw_limit)

    @classmethod
    def from_config(
        cls,
        config: ResolvedDatasetConfig,
    ) -> "DatasetAdapter":
        """Build an adapter from resolved config.

        For source-backed subclasses with declared
        ``metadata_fields_to_preserve``, merge those into the config so
        downstream metadata is preserved without the user having to
        re-declare it in YAML.
        """
        if not cls.metadata_fields_to_preserve:
            return cls(config)

        merged_fields = tuple(
            dict.fromkeys(
                (
                    *config.metadata_fields,
                    *cls.metadata_fields_to_preserve,
                )
            )
        )
        builtin_predict_fields = tuple(
            field
            for field in cls.metadata_fields_to_preserve
            if field in {"policy", "target_role"}
        )
        merged_predict_fields = tuple(
            dict.fromkeys(
                (
                    *config.predict_metadata_fields,
                    *builtin_predict_fields,
                )
            )
        )
        return cls(
            config.model_copy(
                update={
                    "metadata_fields": merged_fields,
                    "predict_metadata_fields": merged_predict_fields,
                }
            )
        )

    # --- main entry point ---

    def load(self) -> list[NormalizedSample]:
        """Default source-backed load flow.

        Subclasses with custom data sources (e.g. local files) should
        override this method directly. Source-backed subclasses should
        override ``load_source_rows()`` instead.
        """
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"{self.config.adapter} supports splits: {supported}"
            )
        self._source_metadata = self._build_source_metadata()
        return self._finalize_samples(self.load_source_rows())

    def load_source_rows(self) -> Iterable[dict[str, Any]]:
        """Return raw row mappings from the upstream source.

        Override this in source-backed subclasses. Local-file
        subclasses that override ``load()`` directly can ignore this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override load() or load_source_rows()"
        )

    def describe(
        self,
        samples: Sequence[NormalizedSample],
    ) -> DatasetMetadata:
        """Build dataset metadata from normalized samples."""
        unsafe_count = sum(1 for sample in samples if sample.label.unsafe)
        source_metadata = self._current_source_metadata()
        raw_display_name = source_metadata.get("display_name")
        if raw_display_name is None:
            display_name = self.config.name.replace("_", " ").title()
        else:
            display_name = str(raw_display_name)
        version = self.config.version or source_metadata.get("version")
        source_uri = (
            self.config.source_uri
            or source_metadata.get("source_uri")
            or self.config.path
        )
        license_name = (
            self.config.license or source_metadata.get("license")
        )
        languages = self._coerce_string_tuple(
            source_metadata.get("languages"),
            field_name="languages",
        )
        categories = self._coerce_string_tuple(
            source_metadata.get("categories"),
            field_name="categories",
        )
        metric_eligibility = {"binary_classification": True}
        metric_eligibility.update(
            self._coerce_bool_mapping(
                source_metadata.get("metric_eligibility"),
                field_name="metric_eligibility",
            )
        )
        metadata: dict[str, Any] = {"adapter": self.config.adapter}
        if source_metadata:
            metadata["source_metadata"] = source_metadata
        if self.label_mapping_note:
            metadata["label_mapping"] = self.label_mapping_note
        return DatasetMetadata(
            name=self.config.name,
            display_name=display_name,
            version=version,
            source_uri=source_uri,
            license=license_name,
            splits=(self.config.split,),
            sample_count=len(samples),
            unsafe_count=unsafe_count,
            languages=languages,
            categories=categories,
            metric_eligibility=metric_eligibility,
            metadata=metadata,
        )

    # --- source metadata ---

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

    def _current_source_metadata(self) -> dict[str, Any]:
        """Return cached source metadata, if available."""
        if self._source_metadata:
            return self._source_metadata
        if self.config.path is None:
            return {}

        source = Path(self.config.path)
        if source.is_dir():
            return self._load_directory_metadata(source)
        return {}

    # --- shared row-normalization helpers ---

    def _make_sample_id(
        self,
        payload: Mapping[str, Any],
        row_number: int,
    ) -> str:
        """Create a deterministic sample ID."""
        serialized = json.dumps(
            payload, sort_keys=True, ensure_ascii=True
        )
        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()[:16]
        return (
            f"{self.config.name}-{self.config.split}-"
            f"{row_number:05d}-{digest}"
        )

    def _load_json_object(self, path: Path) -> dict[str, Any]:
        """Load a JSON object from disk."""
        try:
            raw_value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}") from exc
        if not isinstance(raw_value, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return raw_value

    def _load_directory_metadata(
        self,
        directory: Path,
    ) -> dict[str, Any]:
        """Load optional dataset metadata from a directory."""
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return {}
        return self._load_json_object(metadata_path)

    def _coerce_string_tuple(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        """Normalize a metadata field into a tuple of strings."""
        if value is None:
            return ()
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError(
                    f"{field_name} entries must not be empty"
                )
            return (cleaned,)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a list of strings")

        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"{field_name} entries must be strings"
                )
            cleaned = item.strip()
            if not cleaned:
                raise ValueError(
                    f"{field_name} entries must not be empty"
                )
            items.append(cleaned)
        return tuple(items)

    def _coerce_bool_mapping(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> dict[str, bool]:
        """Normalize a metadata mapping with boolean values."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        result: dict[str, bool] = {}
        for key, flag in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"{field_name} keys must be non-empty strings"
                )
            if not isinstance(flag, bool):
                raise ValueError(
                    f"{field_name} values must be booleans"
                )
            result[key.strip()] = flag
        return result

    def _resolve_source_path(self) -> Path:
        """Resolve the configured source path for file or dir inputs."""
        if self.config.path is None:
            raise ValueError(f"{self.config.adapter} requires a path")

        source = Path(self.config.path)
        if source.is_dir():
            self._source_metadata = self._load_directory_metadata(source)
            for suffix in self.source_suffixes:
                candidate = source / f"{self.config.split}{suffix}"
                if candidate.exists():
                    self._source_path = candidate
                    return candidate
            expected = ", ".join(
                f"{self.config.split}{suffix}"
                for suffix in self.source_suffixes
            )
            raise FileNotFoundError(
                f"dataset directory {source} does not contain any of: "
                f"{expected}"
            )

        if not source.exists():
            raise FileNotFoundError(f"dataset source not found: {source}")

        if self.source_suffixes and source.suffix.lower() not in {
            suffix.lower() for suffix in self.source_suffixes
        }:
            expected = ", ".join(self.source_suffixes)
            raise ValueError(
                f"dataset source {source} must use one of: {expected}"
            )

        self._source_path = source
        return source

    def _messages_from_mapping(
        self,
        row: Mapping[str, Any],
    ) -> list[Message]:
        """Build messages from a row using the configured field names.

        Used by local-file adapters that accept either a ``messages``
        array or a flat ``prompt`` / ``response`` pair.
        """
        if (
            self.config.messages_field
            and row.get(self.config.messages_field)
        ):
            raw_messages = row[self.config.messages_field]
            if isinstance(raw_messages, str):
                try:
                    parsed = json.loads(raw_messages)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid JSON messages field for dataset "
                        f"{self.config.name}"
                    ) from exc
            else:
                parsed = raw_messages
            if not isinstance(parsed, list):
                raise ValueError(
                    "messages field must be a list for dataset "
                    f"{self.config.name}"
                )
            return [
                Message.model_validate(message) for message in parsed
            ]

        prompt_key = self.config.prompt_field
        if prompt_key is None or not row.get(prompt_key):
            raise ValueError(
                "row is missing required prompt/messages field for "
                f"dataset {self.config.name}"
            )

        messages = [Message(role="user", content=str(row[prompt_key]))]
        response_key = self.config.response_field
        if response_key and row.get(response_key):
            messages.append(
                Message(
                    role="assistant",
                    content=str(row[response_key]),
                )
            )
        return messages

    def _extract_metadata(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Select raw metadata fields to preserve."""
        metadata: dict[str, Any] = {}
        for field_name in self.config.metadata_fields:
            if field_name in row:
                metadata[field_name] = row[field_name]
        return metadata

    def _coerce_label(self, raw_value: Any) -> bool:
        """Collapse label values into binary unsafe semantics."""
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            if raw_value in {0, 1}:
                return bool(raw_value)
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"1", "1.0", "true", "unsafe", "yes"}:
                return True
            if normalized in {"0", "0.0", "false", "safe", "no"}:
                return False
        raise ValueError(f"unsupported label value: {raw_value!r}")

    def _normalize_row(
        self,
        row: Mapping[str, Any],
        *,
        row_number: int,
    ) -> NormalizedSample:
        """Normalize a single row."""
        if {
            "id",
            "dataset",
            "split",
            "messages",
            "label",
        }.issubset(row.keys()):
            sample = NormalizedSample.model_validate(row)
            if (
                sample.dataset != self.config.name
                or sample.split != self.config.split
            ):
                sample = sample.model_copy(
                    update={
                        "dataset": self.config.name,
                        "split": self.config.split,
                    }
                )
            return sample

        label_value = row.get(self.config.label_field)
        if label_value is None or (
            isinstance(label_value, str) and not label_value.strip()
        ):
            raise ValueError(
                "row is missing required label field "
                f"{self.config.label_field!r}"
            )

        messages = self._messages_from_mapping(row)
        metadata = self._extract_metadata(row)
        metadata["raw_label"] = label_value

        sample_id = (
            row.get(self.config.id_field)
            if self.config.id_field
            else None
        )
        if not sample_id:
            sample_id = self._make_sample_id(
                {
                    "messages": [
                        message.model_dump(mode="json")
                        for message in messages
                    ],
                    "label": label_value,
                    "metadata": metadata,
                },
                row_number=row_number,
            )

        raw_categories = row.get("category_labels", ())
        if isinstance(raw_categories, str):
            category_labels = (
                tuple(
                    c.strip()
                    for c in raw_categories.split(",")
                    if c.strip()
                )
                if raw_categories.strip()
                else ()
            )
        elif isinstance(raw_categories, (list, tuple)):
            category_labels = tuple(str(c) for c in raw_categories)
        else:
            category_labels = ()

        return NormalizedSample(
            id=str(sample_id),
            dataset=self.config.name,
            split=self.config.split,
            messages=messages,
            label=UnsafeLabel(unsafe=self._coerce_label(label_value)),
            metadata=metadata,
            category_labels=category_labels,
        )

    def _finalize_samples(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[NormalizedSample]:
        """Normalize rows and reject duplicate IDs."""
        samples: list[NormalizedSample] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=1):
            try:
                sample = self._normalize_row(row, row_number=row_number)
            except Exception as exc:
                raise ValueError(
                    f"invalid row {row_number} in dataset "
                    f"{self.config.name}: {exc}"
                ) from exc
            if sample.id in seen_ids:
                raise ValueError(
                    f"duplicate sample id '{sample.id}' in dataset "
                    f"{self.config.name}"
                )
            seen_ids.add(sample.id)
            samples.append(sample)
        return samples
