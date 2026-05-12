"""Local JSONL adapter for text+audio moderation datasets."""

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


@dataset_registry.register("local_audio_jsonl")
class LocalAudioJsonlDataset(MultimodalDatasetAdapter):
    """Load rows from a local JSONL file with audio references."""

    source_suffixes = (".jsonl", ".ndjson")

    def load(self) -> list[NormalizedSample]:
        """Load and normalize local JSONL multimodal rows."""
        path = self._resolve_source_path()
        path_stat = path.stat()
        cache_options = dict(sorted(self.config.options.items()))
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "local_audio_jsonl",
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
        return self._finalize_audio_rows(rows, base_dir=path.parent)

    def _finalize_audio_rows(
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
                sample = self._normalize_audio_row(
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

    def _normalize_audio_row(
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
        audio_field = self._audio_field_name(row)
        audio_value = row.get(audio_field)
        if audio_value is None or (
            isinstance(audio_value, str) and not audio_value.strip()
        ):
            raise ValueError(
                f"row is missing required audio field '{audio_field}'"
            )
        audio_ref = self.resolve_audio_with_base_dir(
            audio_value,
            base_dir=base_dir,
        )
        messages = self._messages_for_row(row, audio_ref=audio_ref)
        metadata = self._extract_metadata(row)
        metadata["audio_field"] = audio_field
        metadata["audio_value"] = row.get(audio_field)
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
                    "audio_sha256": audio_ref.sha256,
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

    def _audio_field_name(self, row: Mapping[str, Any]) -> str:
        """Resolve which row field contains the audio path."""
        explicit = self.config.options.get("audio_field")
        if explicit is not None:
            return str(explicit)
        for field_name in ("audio", "audio_path", "path"):
            if field_name in row:
                return field_name
        raise ValueError(
            "local_audio_jsonl could not infer an audio field; set options.audio_field"
        )

    def _messages_for_row(
        self,
        row: Mapping[str, Any],
        *,
        audio_ref,
    ) -> list[Message]:
        """Build multimodal messages for one row."""
        if self.config.messages_field and row.get(self.config.messages_field):
            messages = self._messages_from_mapping(row)
            target_index = self._audio_attachment_index(messages)
            updated: list[Message] = []
            for index, message in enumerate(messages):
                if index == target_index:
                    if self._message_has_audio(message):
                        updated.append(
                            self._replace_audio_in_message(
                                message,
                                audio_ref,
                            )
                        )
                    else:
                        updated.append(
                            self._append_audio_to_message(message, audio_ref)
                        )
                else:
                    updated.append(
                        self._remove_audio_from_message(message)
                    )
            return updated

        text = self._row_text(row)
        return [
            self.build_multimodal_message(
                text=text or None,
                audio_ref=audio_ref,
            )
        ]

    def _append_audio_to_message(
        self,
        message: Message,
        audio_ref,
    ) -> Message:
        """Attach one audio file to a message while preserving text."""
        parts: list[Any] = []
        if isinstance(message.content, str):
            if message.content:
                parts.append(TextPart(text=message.content))
        else:
            parts.extend(message.content)
        parts.append(MediaPart(media=audio_ref))
        return message.model_copy(update={"content": parts})

    def _replace_audio_in_message(
        self,
        message: Message,
        audio_ref,
    ) -> Message:
        """Replace any existing audio part with one resolved audio ref."""
        if isinstance(message.content, str):
            return self._append_audio_to_message(message, audio_ref)

        parts: list[Any] = []
        inserted_audio = False
        for part in message.content:
            if (
                isinstance(part, MediaPart)
                and part.media.modality == "audio"
            ):
                if not inserted_audio:
                    parts.append(MediaPart(media=audio_ref))
                    inserted_audio = True
                continue
            parts.append(part)
        if not inserted_audio:
            parts.append(MediaPart(media=audio_ref))
        return message.model_copy(update={"content": parts})

    def _message_has_audio(self, message: Message) -> bool:
        """Return whether the message already contains audio media."""
        return bool(message.audio_refs)

    def _remove_audio_from_message(self, message: Message) -> Message:
        """Drop any audio media from a message while preserving other parts."""
        if isinstance(message.content, str):
            return message

        parts = [
            part
            for part in message.content
            if not (
                isinstance(part, MediaPart)
                and part.media.modality == "audio"
            )
        ]
        return message.model_copy(update={"content": parts})

    def _audio_attachment_index(self, messages: list[Message]) -> int:
        """Choose which user turn should carry the resolved audio ref."""
        first_user_index: int | None = None
        last_user_audio_index: int | None = None
        for index, message in enumerate(messages):
            if message.role != "user":
                continue
            if first_user_index is None:
                first_user_index = index
            if self._message_has_audio(message):
                last_user_audio_index = index
        if last_user_audio_index is not None:
            return last_user_audio_index
        if first_user_index is None:
            raise ValueError(
                "messages field did not contain a user message"
            )
        return first_user_index

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
