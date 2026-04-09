"""Nemotron Content Safety Audio dataset adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    cached_download,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel

_log = logging.getLogger(__name__)


@dataset_registry.register("nemotron_content_safety_audio")
class NemotronContentSafetyAudioDataset(
    MultimodalDatasetAdapter,
    SourceBackedDatasetAdapter,
):
    """Normalize the NVIDIA content-safety audio benchmark."""

    display_name = "Nemotron Content Safety Audio Dataset"
    source_uri = (
        "https://huggingface.co/datasets/"
        "nvidia/Nemotron-Content-Safety-Audio-Dataset"
    )
    license_name = "CC-BY-4.0"
    languages = ("en",)
    categories = ("audio_safety", "spoken_content_safety")
    metadata_fields_to_preserve = (
        "audio_filename",
        "audio_duration_seconds",
        "prompt_label_source",
        "response_label_source",
        "speaker_name",
        "speaker_native_language",
    )
    label_mapping_note = (
        "Uses the dataset prompt_label field as the binary unsafe label."
    )
    supported_splits = ("test",)

    def load(self) -> list[NormalizedSample]:
        """Load, download, and normalize Nemotron audio samples."""
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"{self.config.adapter} supports splits: {supported}"
            )
        self._source_metadata = self._build_source_metadata()
        skip_invalid_rows = self._skip_invalid_rows()

        samples: list[NormalizedSample] = []
        seen_ids: set[str] = set()
        skipped_rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(self.load_source_rows(), start=1):
            try:
                sample = self._normalize_source_row(
                    row,
                    row_number=row_number,
                )
            except Exception as exc:
                if not skip_invalid_rows:
                    raise ValueError(
                        f"invalid row {row_number} in dataset {self.config.name}: {exc}"
                    ) from exc
                skipped_rows.append(
                    {
                        "row_number": row_number,
                        "sample_id": str(row.get("id") or ""),
                        "audio_filename": str(row.get("audio_filename") or ""),
                        "error": str(exc),
                    }
                )
                _log.warning(
                    "Skipping invalid Nemotron audio row %s (%s): %s",
                    row_number,
                    row.get("audio_filename") or row.get("id") or "unknown",
                    exc,
                )
                continue
            if sample.id in seen_ids:
                raise ValueError(
                    f"duplicate sample id '{sample.id}' in dataset {self.config.name}"
                )
            seen_ids.add(sample.id)
            samples.append(sample)
        if skipped_rows:
            self._source_metadata["skipped_invalid_row_count"] = len(
                skipped_rows
            )
            self._source_metadata["skipped_invalid_rows"] = skipped_rows[:25]
            if len(skipped_rows) > 25:
                self._source_metadata["skipped_invalid_rows_truncated"] = True
        return samples

    def load_source_rows(self) -> Iterable[dict[str, Any]]:
        """Load raw rows from the upstream HF dataset."""
        return load_hf_rows(
            "nvidia/Nemotron-Content-Safety-Audio-Dataset",
            split=self.config.split,
            limit=self.execution_limit,
        )

    def _normalize_source_row(
        self,
        row: Mapping[str, Any],
        *,
        row_number: int,
    ) -> NormalizedSample:
        """Normalize one raw upstream row."""
        audio_filename = str(row.get("audio_filename") or "").strip()
        if not audio_filename:
            raise ValueError("row is missing audio_filename")
        safe_audio_filename = self._safe_audio_filename(audio_filename)

        audio_path = cached_download(
            alias="nemotron_content_safety_audio",
            url=(
                "https://huggingface.co/datasets/"
                "nvidia/Nemotron-Content-Safety-Audio-Dataset/"
                f"resolve/main/audio/{audio_filename}"
            ),
            filename=safe_audio_filename,
        )
        audio_ref = self.resolve_audio(audio_path)
        unsafe = self._coerce_label(row.get("prompt_label"))
        metadata = self._extract_metadata(row)
        metadata["raw_label"] = row.get("prompt_label")
        if row.get("prompt"):
            metadata["reference_transcript"] = row["prompt"]
        if row.get("response"):
            metadata["reference_response"] = row["response"]

        sample_id = row.get("id")
        if not sample_id:
            sample_id = self._make_sample_id(
                {
                    "audio_filename": audio_filename,
                    "audio_sha256": audio_ref.sha256,
                    "label": row.get("prompt_label"),
                    "violated_categories": row.get("violated_categories"),
                },
                row_number,
            )

        return NormalizedSample(
            id=str(sample_id),
            dataset=self.config.name,
            split=self.config.split,
            messages=[
                self.build_multimodal_message(
                    audio_ref=audio_ref,
                )
            ],
            label=UnsafeLabel(unsafe=unsafe),
            category_labels=self._category_labels(
                row.get("violated_categories")
            ),
            metadata=metadata,
        )

    def _category_labels(self, value: Any) -> tuple[str, ...]:
        """Normalize violated-categories metadata into labels."""
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(
                item.strip()
                for item in value.split(",")
                if item.strip()
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        cleaned = str(value).strip()
        return (cleaned,) if cleaned else ()

    def _safe_audio_filename(self, audio_filename: str) -> str:
        """Reject unsafe upstream filenames before writing to cache."""
        if (
            "/" in audio_filename
            or "\\" in audio_filename
            or audio_filename in {".", ".."}
            or Path(audio_filename).name != audio_filename
        ):
            raise ValueError(
                f"unsafe audio_filename path: {audio_filename!r}"
            )
        return audio_filename

    def _skip_invalid_rows(self) -> bool:
        """Return whether invalid upstream rows should be dropped."""
        value = self.config.options.get("skip_invalid_rows", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)
