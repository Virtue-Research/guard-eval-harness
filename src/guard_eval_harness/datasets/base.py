"""Base dataset adapter interfaces."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.schemas import (
    DatasetMetadata,
    Message,
    NormalizedSample,
)


class DatasetAdapter(ABC):
    """Base interface for dataset normalization adapters."""

    source_suffixes: tuple[str, ...] = ()

    def __init__(self, config: ResolvedDatasetConfig) -> None:
        self.config = config
        self._source_metadata: dict[str, Any] = {}
        self._source_path: Path | None = None
        raw_limit = self.config.options.get("limit")
        self.execution_limit: int | None = None
        if raw_limit is not None:
            self.execution_limit = int(raw_limit)

    @classmethod
    def from_config(cls, config: ResolvedDatasetConfig) -> "DatasetAdapter":
        """Build an adapter from resolved config."""
        return cls(config)

    @abstractmethod
    def load(self) -> list[NormalizedSample]:
        """Load and normalize samples."""

    def describe(self, samples: Sequence[NormalizedSample]) -> DatasetMetadata:
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
        license_name = self.config.license or source_metadata.get("license")
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

    def _make_sample_id(self, payload: Mapping[str, Any], row_number: int) -> str:
        """Create a deterministic sample ID."""
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{self.config.name}-{self.config.split}-{row_number:05d}-{digest}"

    def _load_json_object(self, path: Path) -> dict[str, Any]:
        """Load a JSON object from disk."""
        try:
            raw_value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}") from exc

        if not isinstance(raw_value, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return raw_value

    def _load_directory_metadata(self, directory: Path) -> dict[str, Any]:
        """Load optional dataset metadata from a directory."""
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return {}
        return self._load_json_object(metadata_path)

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
                raise ValueError(f"{field_name} entries must not be empty")
            return (cleaned,)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a list of strings")

        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} entries must be strings")
            cleaned = item.strip()
            if not cleaned:
                raise ValueError(f"{field_name} entries must not be empty")
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
                raise ValueError(f"{field_name} keys must be non-empty strings")
            if not isinstance(flag, bool):
                raise ValueError(f"{field_name} values must be booleans")
            result[key.strip()] = flag
        return result

    def _resolve_source_path(self) -> Path:
        """Resolve the configured source path for file or directory inputs."""
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
                f"{self.config.split}{suffix}" for suffix in self.source_suffixes
            )
            raise FileNotFoundError(
                f"dataset directory {source} does not contain any of: {expected}"
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
        """Construct normalized messages from a raw row mapping."""
        if self.config.messages_field and row.get(self.config.messages_field):
            raw_messages = row[self.config.messages_field]
            if isinstance(raw_messages, str):
                try:
                    parsed = json.loads(raw_messages)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON messages field for dataset {self.config.name}"
                    ) from exc
            else:
                parsed = raw_messages
            if not isinstance(parsed, list):
                raise ValueError(
                    f"messages field must be a list for dataset {self.config.name}"
                )
            return [Message.model_validate(message) for message in parsed]

        prompt_key = self.config.prompt_field
        if prompt_key is None or not row.get(prompt_key):
            raise ValueError(
                f"row is missing required prompt/messages field for dataset "
                f"{self.config.name}"
            )

        messages = [Message(role="user", content=str(row[prompt_key]))]
        response_key = self.config.response_field
        if response_key and row.get(response_key):
            messages.append(Message(role="assistant", content=str(row[response_key])))
        return messages

    def _extract_metadata(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Select raw metadata fields to preserve."""
        metadata: dict[str, Any] = {}
        for field_name in self.config.metadata_fields:
            if field_name in row and self._preserve_metadata_field(field_name):
                metadata[field_name] = row[field_name]
        return metadata

    def _preserve_metadata_field(self, field_name: str) -> bool:
        """Return whether a raw field is safe to expose as metadata."""
        return (
            field_name != self.config.label_field
            and field_name != "category_labels"
        )

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
            if sample.dataset != self.config.name or sample.split != self.config.split:
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
                f"row is missing required label field '{self.config.label_field}'"
            )

        messages = self._messages_from_mapping(row)
        metadata = self._extract_metadata(row)

        raw_categories = row.get("category_labels", ())
        if isinstance(raw_categories, str):
            category_labels = (
                tuple(c.strip() for c in raw_categories.split(",") if c.strip())
                if raw_categories.strip()
                else ()
            )
        elif isinstance(raw_categories, (list, tuple)):
            category_labels = tuple(str(c) for c in raw_categories)
        else:
            category_labels = ()

        sample_id = row.get(self.config.id_field) if self.config.id_field else None
        if not sample_id:
            sample_id = self._make_sample_id(
                {
                    "messages": [message.model_dump(mode="json") for message in messages],
                    "label": label_value,
                    "metadata": metadata,
                    "category_labels": category_labels,
                },
                row_number=row_number,
            )

        return NormalizedSample(
            id=str(sample_id),
            dataset=self.config.name,
            split=self.config.split,
            messages=messages,
            label={"unsafe": self._coerce_label(label_value)},
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
                    f"invalid row {row_number} in dataset {self.config.name}: {exc}"
                ) from exc
            if sample.id in seen_ids:
                raise ValueError(
                    f"duplicate sample id '{sample.id}' in dataset {self.config.name}"
                )
            seen_ids.add(sample.id)
            samples.append(sample)
        return samples
