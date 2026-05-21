"""Pydantic schema for run configs and dataset-adapter configs.

This is the single source of truth for what a YAML can contain.
``ResolvedRunConfig`` is the top-level run config (model + datasets +
output). ``ResolvedDatasetConfig`` is the lower-level config that
``DatasetAdapter`` subclasses consume (label_field, prompt_field, etc.).
"""

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ConfigModel(BaseModel):
    """Common base for resolved-config models."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Denylist: fields that must not be exposed to the model as PredictSample
# metadata, since they would leak the ground-truth label.
# ---------------------------------------------------------------------------


PREDICT_METADATA_FIELD_DENYLIST: frozenset[str] = frozenset(
    {
        "raw_label",
        "label",
        "labels",
        "label_name",
        "category_labels",
        "binary_label",
        "majority_label",
        "safety_label",
        "image_safety_label",
        "image_safe",
        "safe",
        "unsafe",
        "is_safe",
        "is_unsafe",
    }
)


# ---------------------------------------------------------------------------
# Dataset-adapter-level config (consumed by DatasetAdapter subclasses)
# ---------------------------------------------------------------------------


class ResolvedDatasetConfig(ConfigModel):
    """Resolved dataset configuration consumed by ``DatasetAdapter``."""

    name: str
    adapter: str
    path: str | None = None
    split: str = "test"
    id_field: str | None = "id"
    messages_field: str | None = "messages"
    prompt_field: str | None = "prompt"
    response_field: str | None = None
    label_field: str = "unsafe"
    metadata_fields: tuple[str, ...] = ()
    predict_metadata_fields: tuple[str, ...] = ()
    version: str | None = None
    source_uri: str | None = None
    license: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "adapter", "split", "label_field")
    @classmethod
    def _required_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("path")
    @classmethod
    def _normalize_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return Path(value).expanduser().as_posix()

    @field_validator("metadata_fields", "predict_metadata_fields")
    @classmethod
    def _validate_metadata_fields(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in value:
            field_name = str(item).strip()
            if not field_name:
                raise ValueError(
                    "metadata field names must not be empty"
                )
            cleaned.append(field_name)
        return tuple(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def _validate_predict_metadata_fields(
        self,
    ) -> "ResolvedDatasetConfig":
        """Reject fields that would expose score-side labels to models."""
        invalid = [
            field
            for field in self.predict_metadata_fields
            if (
                field == self.label_field
                or field in PREDICT_METADATA_FIELD_DENYLIST
            )
        ]
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(
                "predict_metadata_fields cannot include label-like "
                f"fields: {names}"
            )
        return self


# ---------------------------------------------------------------------------
# Run-level config (top-level YAML: model + datasets + output)
# ---------------------------------------------------------------------------


class InlinePolicy(ConfigModel):
    """Inline policy declared directly in a YAML config."""

    name: str
    text: str
    categories: tuple[str, ...] = ()

    @field_validator("name", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("policy name and text must not be empty")
        return cleaned


class ResolvedBackendConfig(ConfigModel):
    """Backend (inference engine) configuration."""

    kind: str
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _kind_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("backend.kind must not be empty")
        return cleaned


class ResolvedModelConfig(ConfigModel):
    """Top-level model group: guard + optional policy/output_format + backend."""

    guard: str
    policy: str | InlinePolicy | None = None
    output_format: str | None = None
    guard_args: dict[str, Any] = Field(default_factory=dict)
    backend: ResolvedBackendConfig

    @field_validator("guard", "output_format")
    @classmethod
    def _non_empty_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned


class ResolvedDatasetSelection(ConfigModel):
    """One dataset entry in a run config, with optional subset selection."""

    name: str
    adapter: str
    path: str | None = None
    split: str = "test"
    limit: int | None = Field(default=None, ge=1)
    sample_ids: tuple[str, ...] = ()
    sample_indices: tuple[int, ...] = ()
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "adapter", "split")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("path")
    @classmethod
    def _normalize_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return Path(value).expanduser().as_posix()

    @field_validator("sample_indices")
    @classmethod
    def _non_negative_indices(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        for index in value:
            if index < 0:
                raise ValueError("sample_indices must be non-negative")
        return value

    @field_validator("sample_ids")
    @classmethod
    def _non_empty_sample_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        for sid in value:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError(
                    "sample_ids entries must be non-empty strings"
                )
        return value

    @model_validator(mode="after")
    def _exclusive_subset_selector(
        self,
    ) -> "ResolvedDatasetSelection":
        selectors = [
            ("limit", self.limit is not None),
            ("sample_ids", bool(self.sample_ids)),
            ("sample_indices", bool(self.sample_indices)),
        ]
        active = [name for name, present in selectors if present]
        if len(active) > 1:
            joined = ", ".join(active)
            raise ValueError(
                f"dataset {self.name!r}: at most one of "
                f"limit/sample_ids/sample_indices may be set "
                f"(got {joined})"
            )
        return self


class ResolvedOutputConfig(ConfigModel):
    """Output / artifacts configuration."""

    run_dir: str
    resume: bool = True
    overwrite: bool = False

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_dir must not be empty")
        return Path(cleaned).expanduser().as_posix()

    @model_validator(mode="after")
    def _resume_xor_overwrite(self) -> "ResolvedOutputConfig":
        if self.resume and self.overwrite:
            raise ValueError(
                "output.resume and output.overwrite are mutually exclusive"
            )
        return self


class ResolvedRunConfig(ConfigModel):
    """Top-level resolved config for one test run."""

    version: int = 2
    run_name: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model: ResolvedModelConfig
    datasets: list[ResolvedDatasetSelection] = Field(min_length=1)
    output: ResolvedOutputConfig
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _version_must_be_two(cls, value: int) -> int:
        if value != 2:
            raise ValueError("ResolvedRunConfig requires version=2")
        return value

    @field_validator("run_name")
    @classmethod
    def _run_name_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_name must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _unique_dataset_names(self) -> "ResolvedRunConfig":
        seen_names: set[str] = set()
        seen_keys: set[str] = set()
        for dataset in self.datasets:
            if dataset.name in seen_names:
                raise ValueError(
                    f"duplicate dataset name: {dataset.name!r}"
                )
            storage_key = dataset.name.replace("/", "__")
            if storage_key in seen_keys:
                raise ValueError(
                    f"dataset names {dataset.name!r} and a previous entry "
                    f"map to the same artifact directory {storage_key!r}"
                )
            seen_names.add(dataset.name)
            seen_keys.add(storage_key)
        return self
