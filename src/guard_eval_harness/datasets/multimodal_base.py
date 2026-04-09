"""Base class for datasets containing images or other media."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from io import BytesIO
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from collections.abc import Callable

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.schemas import DatasetMetadata
from guard_eval_harness.datasets.media_cache import (
    audio_metadata,
    compute_sha256,
    default_cache_dir,
    image_dimensions,
    resolve_local_audio,
    resolve_pil_image,
)
from guard_eval_harness.datasets.sample_cache import (
    compute_cache_key,
    load_cached_samples,
    sample_cache_base_dir,
    write_sample_cache,
)
from guard_eval_harness.schemas import (
    MediaPart,
    MediaRef,
    Message,
    NormalizedSample,
    TextPart,
    UnsafeLabel,
)

_log = logging.getLogger(__name__)


class MultimodalDatasetAdapter(DatasetAdapter):
    """Base class for datasets that include media alongside text.

    Subclasses should override ``load()`` and use the helper methods
    below to construct ``NormalizedSample`` objects with typed
    multimodal content parts.
    """

    def describe(self, samples: Sequence[NormalizedSample]) -> DatasetMetadata:
        """Build metadata with modalities inferred from content."""
        meta = super().describe(samples)
        modalities: set[str] = set()
        for sample in samples:
            for message in sample.messages:
                if message.text_content.strip():
                    modalities.add("text")
                for media_ref in message.media_refs:
                    modalities.add(media_ref.modality)
            if (
                "text" in modalities
                and "image" in modalities
                and "audio" in modalities
            ):
                break
        if not modalities:
            modalities.add("text")
        return meta.model_copy(
            update={"input_modalities": tuple(sorted(modalities))}
        )

    def resolve_image(self, source: Any) -> MediaRef:
        """Resolve an image source to a ``MediaRef``.

        Accepts:
        - ``PIL.Image.Image`` — saved to content-addressed cache.
        - ``str`` or ``Path`` — treated as a local file path.
        - Hugging Face image dicts with ``bytes`` or ``path`` keys.

        Returns a ``MediaRef`` with ``uri``, ``sha256``, ``width``,
        and ``height`` populated.
        """
        return self.resolve_image_with_base_dir(source)

    def resolve_image_with_base_dir(
        self,
        source: Any,
        *,
        base_dir: str | Path | None = None,
    ) -> MediaRef:
        """Resolve an image source relative to one optional base directory."""
        cache_dir = Path(
            self.config.options.get(
                "media_cache_dir",
                str(default_cache_dir()),
            )
        )
        resolved_base_dir = (
            Path(base_dir).expanduser() if base_dir is not None else None
        )

        try:
            from PIL import Image  # type: ignore[import-untyped]

            is_pil = isinstance(source, Image.Image)
        except ImportError:
            is_pil = False

        if is_pil:
            local_path, digest = resolve_pil_image(source, cache_dir=cache_dir)
        elif isinstance(source, MappingABC):
            if source.get("bytes") is not None:
                if "Image" not in locals():
                    raise ImportError("Pillow is required for image support")
                payload = source["bytes"]
                if not isinstance(payload, (bytes, bytearray)):
                    raise TypeError("image bytes must be bytes or bytearray")
                with Image.open(BytesIO(payload)) as image:
                    local_path, digest = resolve_pil_image(
                        image.copy(),
                        cache_dir=cache_dir,
                    )
            elif source.get("path") is not None:
                local_path = Path(str(source["path"])).expanduser()
                if (
                    resolved_base_dir is not None
                    and not local_path.is_absolute()
                ):
                    local_path = resolved_base_dir / local_path
                if not local_path.exists():
                    raise FileNotFoundError(f"Image not found: {local_path}")
                digest = compute_sha256(local_path)
            else:
                raise TypeError(
                    "Unsupported image mapping; expected 'bytes' or 'path'"
                )
        elif isinstance(source, (str, Path)):
            local_path = Path(source).expanduser()
            if resolved_base_dir is not None and not local_path.is_absolute():
                local_path = resolved_base_dir / local_path
            if not local_path.exists():
                raise FileNotFoundError(f"Image not found: {local_path}")
            digest = compute_sha256(local_path)
        else:
            raise TypeError(
                f"Unsupported image source type: {type(source).__name__}"
            )

        width, height = image_dimensions(local_path)
        return MediaRef(
            modality="image",
            uri=local_path.as_posix(),
            sha256=digest,
            width=width,
            height=height,
        )

    def build_multimodal_message(
        self,
        *,
        text: str | None = None,
        image_ref: MediaRef | None = None,
        audio_ref: MediaRef | None = None,
        role: str = "user",
    ) -> Message:
        """Construct a ``Message`` with typed content parts.

        At least one content part must be provided.
        """
        parts: list[Any] = []
        if text:
            parts.append(TextPart(text=text))
        if image_ref is not None:
            parts.append(MediaPart(media=image_ref))
        if audio_ref is not None:
            parts.append(MediaPart(media=audio_ref))
        if not parts:
            raise ValueError(
                "At least one of text, image_ref, or audio_ref is required"
            )
        return Message(role=role, content=parts)

    def normalize_multimodal_row(
        self,
        row: Mapping[str, Any],
        *,
        row_number: int,
        text: str | None = None,
        image_ref: MediaRef | None = None,
        audio_ref: MediaRef | None = None,
        unsafe: bool,
        category_labels: tuple[str, ...] = (),
        extra_metadata: dict[str, Any] | None = None,
    ) -> NormalizedSample:
        """Create a ``NormalizedSample`` with multimodal content."""
        message = self.build_multimodal_message(
            text=text,
            image_ref=image_ref,
            audio_ref=audio_ref,
        )
        sample_id = self._make_sample_id(row, row_number)
        metadata = dict(extra_metadata or {})
        for field in self.config.metadata_fields:
            if field in row:
                metadata[field] = row[field]
        return NormalizedSample(
            id=sample_id,
            dataset=self.config.name,
            split=self.config.split,
            messages=[message],
            label=UnsafeLabel(unsafe=unsafe),
            category_labels=category_labels,
            metadata=metadata,
        )

    def resolve_audio(self, source: Any) -> MediaRef:
        """Resolve an audio source to a ``MediaRef``."""
        return self.resolve_audio_with_base_dir(source)

    def resolve_audio_with_base_dir(
        self,
        source: Any,
        *,
        base_dir: str | Path | None = None,
    ) -> MediaRef:
        """Resolve a local audio source relative to one optional base directory."""
        resolved_base_dir = (
            Path(base_dir).expanduser() if base_dir is not None else None
        )

        if not isinstance(source, (str, Path)):
            raise TypeError(
                f"Unsupported audio source type: {type(source).__name__}"
            )

        local_path = Path(source).expanduser()
        if resolved_base_dir is not None and not local_path.is_absolute():
            local_path = resolved_base_dir / local_path
        local_path, digest = resolve_local_audio(local_path)
        duration_seconds, sample_rate_hz, channels, mime_type = audio_metadata(
            local_path
        )
        return MediaRef(
            modality="audio",
            uri=local_path.as_posix(),
            sha256=digest,
            mime_type=mime_type,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )

    # ------------------------------------------------------------------
    # Sample-level caching
    # ------------------------------------------------------------------

    def _load_with_sample_cache(
        self,
        cache_key_parts: dict[str, Any],
        loader: Callable[[], list[NormalizedSample]],
    ) -> list[NormalizedSample]:
        """Return cached samples on hit, otherwise call *loader*.

        Set ``options.no_sample_cache`` to bypass the cache entirely.
        Set ``options.sample_cache_dir`` to override the default
        cache location.
        """
        if self.config.options.get("no_sample_cache"):
            return loader()

        cache_dir = Path(
            self.config.options.get(
                "sample_cache_dir",
                str(sample_cache_base_dir()),
            )
        )
        adapter_name = self.config.adapter or self.config.name
        cache_key_parts.setdefault("dataset_name", self.config.name)
        key = compute_cache_key(cache_key_parts)

        cached = load_cached_samples(cache_dir, adapter_name, key)
        if cached is not None:
            _log.info(
                "Sample cache hit for %s (%d samples)",
                adapter_name,
                len(cached),
            )
            return cached

        samples = loader()
        write_sample_cache(cache_dir, adapter_name, key, samples)
        _log.info(
            "Wrote sample cache for %s (%d samples)",
            adapter_name,
            len(samples),
        )
        return samples
