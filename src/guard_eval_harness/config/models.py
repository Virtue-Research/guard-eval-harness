"""Resolved config models used by the coordinator-owned foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigModel(BaseModel):
    """Common pydantic base for resolved config models."""

    model_config = ConfigDict(extra="forbid")


class ResolvedDatasetConfig(ConfigModel):
    """Resolved dataset configuration."""

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
    version: str | None = None
    source_uri: str | None = None
    license: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "adapter", "split", "label_field")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        """Reject empty required config fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        """Normalize paths to POSIX strings."""
        if value is None:
            return None
        return Path(value).expanduser().as_posix()


class ResolvedModelConfig(ConfigModel):
    """Resolved model adapter configuration."""

    adapter: str
    model_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        """Reject empty adapter names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("adapter must not be empty")
        return cleaned


class ResolvedExecutionConfig(ConfigModel):
    """Execution tuning knobs for a run."""

    batch_size: int | Literal["auto"] = 1
    concurrency: int = Field(default=1, ge=1)
    retries: int = Field(default=8, ge=0)
    retry_backoff: float = Field(default=2.0, ge=0.0)
    limit: int | None = Field(default=None, ge=1)
    resume: bool = False

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(
        cls,
        value: int | str,
    ) -> int | Literal["auto"]:
        """Accept positive integers or the literal ``auto``."""
        if isinstance(value, bool):
            raise ValueError(
                "batch_size must be an integer >= 1 or 'auto'"
            )
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "auto":
                return "auto"
            try:
                parsed = int(normalized)
            except ValueError:
                parsed = None
            if (
                parsed is not None
                and parsed >= 1
                and "." not in normalized
                and "e" not in normalized
            ):
                return parsed
            raise ValueError(
                "batch_size must be an integer >= 1 or 'auto'"
            )
        if isinstance(value, int) and value >= 1:
            return value
        raise ValueError("batch_size must be an integer >= 1 or 'auto'")


class ResolvedOutputConfig(ConfigModel):
    """Resolved output directory configuration."""

    run_dir: str
    overwrite: bool = False

    @field_validator("run_dir")
    @classmethod
    def validate_run_dir(cls, value: str) -> str:
        """Reject empty output directories."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_dir must not be empty")
        return Path(cleaned).expanduser().as_posix()


class ResolvedRunConfig(ConfigModel):
    """Top-level resolved run configuration."""

    version: int = 1
    run_name: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model: ResolvedModelConfig
    datasets: list[ResolvedDatasetConfig] = Field(min_length=1)
    execution: ResolvedExecutionConfig = Field(
        default_factory=ResolvedExecutionConfig
    )
    output: ResolvedOutputConfig
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, value: str) -> str:
        """Reject empty run names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_name must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_unique_dataset_names(self) -> "ResolvedRunConfig":
        """Reject configs whose dataset names collide on disk."""
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
