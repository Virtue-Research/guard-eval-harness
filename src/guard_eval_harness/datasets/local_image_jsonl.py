"""Local JSONL adapter for text+image moderation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import (
    MediaPart,
    Message,
    NormalizedSample,
    TextPart,
    UnsafeLabel,
)


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


@dataset_registry.register("local_image_jsonl")
class LocalImageJsonlDataset(MultimodalDatasetAdapter):
    """Load rows from a local JSONL file with image references."""

    source_suffixes = (".jsonl", ".ndjson")

    def load(self) -> list[NormalizedSample]:
        """Load and normalize local JSONL multimodal rows."""
        path = self._resolve_source_path()
        path_stat = path.stat()
        cache_options = dict(sorted(self.config.options.items()))
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "local_image_jsonl",
                "path": path.as_posix(),
                "path_mtime_ns": path_stat.st_mtime_ns,
                "path_size": path_stat.st_size,
                "split": self.config.split,
                "prompt_field": self.config.prompt_field,
                "messages_field": self.config.messages_field,
                "label_field": self.config.label_field,
                "id_field": self.config.id_field,
                "metadata_fields": tuple(self.config.metadata_fields),
                "options": cache_options,
                "execution_limit": self.execution_limit,
            },
            loader=lambda: self._load_rows_from_path(path),
        )

    def _load_rows_from_path(self, path: Path) -> list[NormalizedSample]:
        """Load and normalize local JSONL rows from one resolved path."""
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row {line_number} at {path} must be an object"
                    )
                rows.append(row)
        return self._finalize_image_rows(rows, base_dir=path.parent)

    def _finalize_image_rows(
        self,
        rows: list[Mapping[str, Any]],
        *,
        base_dir: Path,
    ) -> list[NormalizedSample]:
        """Normalize rows and reject duplicate IDs."""
        samples: list[NormalizedSample] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=1):
            try:
                sample = self._normalize_image_row(
                    row,
                    row_number=row_number,
                    base_dir=base_dir,
                )
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

    def _normalize_image_row(
        self,
        row: Mapping[str, Any],
        *,
        row_number: int,
        base_dir: Path,
    ) -> NormalizedSample:
        """Normalize one local multimodal JSONL row."""
        label_value = row.get(self.config.label_field)
        if label_value is None:
            raise ValueError(
                f"row is missing required label field '{self.config.label_field}'"
            )
        image_field = self._image_field_name(row)
        image_value = row.get(image_field)
        if image_value is None or (
            isinstance(image_value, str) and not image_value.strip()
        ):
            raise ValueError(
                f"row is missing required image field '{image_field}'"
            )
        image_ref = self.resolve_image_with_base_dir(
            image_value,
            base_dir=base_dir,
        )
        messages = self._messages_for_row(row, image_ref=image_ref)
        metadata = self._extract_metadata(row)
        metadata["image_field"] = image_field
        metadata["image_value"] = row.get(image_field)
        category_labels = self._category_labels(row)

        sample_id = (
            row.get(self.config.id_field) if self.config.id_field else None
        )
        if not sample_id:
            sample_id = self._make_sample_id(
                {
                    "messages": [
                        message.model_dump(mode="json") for message in messages
                    ],
                    "label": label_value,
                    "image_sha256": image_ref.sha256,
                    "metadata": metadata,
                    "category_labels": category_labels,
                },
                row_number,
            )

        return NormalizedSample(
            id=str(sample_id),
            dataset=self.config.name,
            split=self.config.split,
            messages=messages,
            label=UnsafeLabel(unsafe=self._coerce_label(label_value)),
            category_labels=category_labels,
            metadata=metadata,
        )

    def _image_field_name(self, row: Mapping[str, Any]) -> str:
        """Resolve which row field contains the image path."""
        explicit = self.config.options.get("image_field")
        if explicit is not None:
            return str(explicit)
        for field_name in ("image", "image_path", "path"):
            if field_name in row:
                return field_name
        raise ValueError(
            "local_image_jsonl could not infer an image field; set options.image_field"
        )

    def _messages_for_row(
        self,
        row: Mapping[str, Any],
        *,
        image_ref,
    ) -> list[Message]:
        """Build multimodal messages for one row."""
        if self.config.messages_field and row.get(self.config.messages_field):
            messages = self._messages_from_mapping(row)
            attached = False
            updated: list[Message] = []
            for message in messages:
                if not attached and message.role == "user":
                    updated.append(
                        self._append_image_to_message(message, image_ref)
                    )
                    attached = True
                else:
                    updated.append(message)
            if not attached:
                raise ValueError(
                    "messages field did not contain a user message"
                )
            return updated

        text = self._row_text(row)
        return [
            self.build_multimodal_message(
                text=text or None,
                image_ref=image_ref,
            )
        ]

    def _append_image_to_message(
        self,
        message: Message,
        image_ref,
    ) -> Message:
        """Attach one image to a message while preserving existing text."""
        parts: list[Any] = []
        if isinstance(message.content, str):
            if message.content:
                parts.append(TextPart(text=message.content))
        else:
            parts.extend(message.content)
        parts.append(MediaPart(media=image_ref))
        return message.model_copy(update={"content": parts})

    def _row_text(self, row: Mapping[str, Any]) -> str:
        """Resolve text content for one row when no messages field is used."""
        prompt_field = self.config.prompt_field
        if prompt_field and row.get(prompt_field):
            return _clean_text(row[prompt_field])
        for field_name in ("text", "prompt", "instruction", "caption"):
            if row.get(field_name):
                return _clean_text(row[field_name])
        return ""

    def _category_labels(
        self,
        row: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Resolve one optional category field into labels."""
        category_field = self.config.options.get("category_field")
        if category_field is None:
            return ()
        category_value = _clean_text(row.get(str(category_field)))
        if not category_value:
            return ()
        return (category_value,)
