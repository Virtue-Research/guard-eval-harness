"""Base class for datasets that include images.

Extends ``DatasetAdapter`` with image-resolution helpers and an HF
loader that handles image columns. Subclasses either:

- Override ``load()`` for local-file image datasets, or
- Override ``load_source_rows()`` for HF-hosted image datasets and
  let the inherited default ``load()`` drive the pipeline.
"""

import importlib
import logging
from collections.abc import Callable, Mapping as MappingABC, Sequence
from pathlib import Path
from typing import Any, Mapping

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.utils.media_cache import (
    compute_sha256,
    default_cache_dir,
    image_dimensions,
    resolve_image_bytes,
    resolve_pil_image,
)
from guard_eval_harness.utils.sample_cache import (
    compute_cache_key,
    load_cached_samples,
    sample_cache_base_dir,
    write_sample_cache,
)
from guard_eval_harness.schemas import (
    DatasetMetadata,
    MediaPart,
    MediaRef,
    Message,
    NormalizedSample,
    TextPart,
    UnsafeLabel,
)

_log = logging.getLogger(__name__)


class ImageDatasetAdapter(DatasetAdapter):
    """Base class for datasets that include images alongside text."""

    # Optional metadata for source-backed image datasets.
    access_mode: str | None = None
    upstream_images_uri: str | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def describe(
        self,
        samples: Sequence[NormalizedSample],
    ) -> DatasetMetadata:
        """Build metadata with modalities inferred from content."""
        meta = super().describe(samples)
        modalities: set[str] = set()
        for sample in samples:
            for message in sample.messages:
                if message.text_content.strip():
                    modalities.add("text")
                for media_ref in message.media_refs:
                    modalities.add(media_ref.modality)
            if "text" in modalities and "image" in modalities:
                break
        if not modalities:
            modalities.add("text")
        return meta.model_copy(
            update={"input_modalities": tuple(sorted(modalities))}
        )

    def _build_source_metadata(self) -> dict[str, Any]:
        """Extend base source metadata with image-specific fields."""
        metadata = super()._build_source_metadata()
        if self.access_mode is not None:
            metadata["access_mode"] = self.access_mode
        if self.upstream_images_uri is not None:
            metadata["upstream_images_uri"] = self.upstream_images_uri
        return metadata

    def _ensure_supported_split(self) -> None:
        """Reject unsupported split names early.

        Image datasets that override ``load()`` typically call this to
        avoid silently loading an empty / wrong split.
        """
        if self.config.split not in self.supported_splits:
            supported = ", ".join(self.supported_splits)
            raise ValueError(
                f"{self.config.adapter} supports splits: {supported}"
            )

    # ------------------------------------------------------------------
    # Image resolution
    # ------------------------------------------------------------------

    def resolve_image(self, source: Any) -> MediaRef:
        """Resolve an image source to a ``MediaRef``.

        Accepts:
        - ``PIL.Image.Image`` — saved to content-addressed cache.
        - ``str`` or ``Path`` — treated as a local file path.
        - Hugging Face image dicts with ``bytes`` or ``path`` keys.
        """
        return self.resolve_image_with_base_dir(source)

    def resolve_image_with_base_dir(
        self,
        source: Any,
        *,
        base_dir: str | Path | None = None,
    ) -> MediaRef:
        """Resolve an image relative to one optional base directory."""
        cache_dir = Path(
            self.config.options.get(
                "media_cache_dir",
                str(default_cache_dir()),
            )
        )
        resolved_base_dir = (
            Path(base_dir).expanduser()
            if base_dir is not None
            else None
        )

        try:
            from PIL import Image  # type: ignore[import-untyped]

            is_pil = isinstance(source, Image.Image)
        except ImportError:
            is_pil = False

        if is_pil:
            local_path, digest = resolve_pil_image(
                source, cache_dir=cache_dir
            )
        elif isinstance(source, MappingABC):
            if source.get("bytes") is not None:
                payload = source["bytes"]
                if not isinstance(payload, (bytes, bytearray)):
                    raise TypeError(
                        "image bytes must be bytes or bytearray"
                    )
                local_path, digest = resolve_image_bytes(
                    payload, cache_dir=cache_dir
                )
            elif source.get("path") is not None:
                local_path = Path(str(source["path"])).expanduser()
                if (
                    resolved_base_dir is not None
                    and not local_path.is_absolute()
                ):
                    local_path = resolved_base_dir / local_path
                if not local_path.exists():
                    raise FileNotFoundError(
                        f"Image not found: {local_path}"
                    )
                digest = compute_sha256(local_path)
            else:
                raise TypeError(
                    "Unsupported image mapping; expected "
                    "'bytes' or 'path'"
                )
        elif isinstance(source, (str, Path)):
            local_path = Path(source).expanduser()
            if (
                resolved_base_dir is not None
                and not local_path.is_absolute()
            ):
                local_path = resolved_base_dir / local_path
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {local_path}"
                )
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

    # ------------------------------------------------------------------
    # Message + sample construction
    # ------------------------------------------------------------------

    def build_multimodal_message(
        self,
        *,
        text: str | None = None,
        image_ref: MediaRef | None = None,
        role: str = "user",
    ) -> Message:
        """Construct a ``Message`` with text + image content parts."""
        parts: list[Any] = []
        if text:
            parts.append(TextPart(text=text))
        if image_ref is not None:
            parts.append(MediaPart(media=image_ref))
        if not parts:
            raise ValueError(
                "At least one of text or image_ref is required"
            )
        return Message(role=role, content=parts)

    def normalize_multimodal_row(
        self,
        row: Mapping[str, Any],
        *,
        row_number: int,
        text: str | None = None,
        image_ref: MediaRef | None = None,
        unsafe: bool,
        category_labels: tuple[str, ...] = (),
        extra_metadata: dict[str, Any] | None = None,
    ) -> NormalizedSample:
        """Create a ``NormalizedSample`` with text + image content."""
        message = self.build_multimodal_message(
            text=text,
            image_ref=image_ref,
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

    # ------------------------------------------------------------------
    # HF source loader for image datasets
    # ------------------------------------------------------------------

    def _load_hf_rows(
        self,
        repo_id: str,
        *,
        split: str,
        subset: str | None = None,
        revision: str | None = None,
        data_dir: str | None = None,
        verification_mode: str | None = None,
        image_decode: bool | None = None,
        image_columns: tuple[str, ...] = ("image",),
    ) -> list[dict[str, Any]]:
        """Load one HF dataset split as plain dictionaries.

        Supports controlling image decoding via ``image_decode``.
        Pass ``False`` to keep image bytes lazy (useful for big
        datasets where you only want metadata first).
        """
        try:
            datasets = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"{self.config.adapter} requires the 'datasets' package"
            ) from exc

        split_name = split
        if self.execution_limit is not None:
            split_name = f"{split}[:{self.execution_limit}]"
        kwargs: dict[str, Any] = {
            "split": split_name,
            "revision": revision,
            "data_dir": data_dir,
        }
        if verification_mode is not None:
            kwargs["verification_mode"] = verification_mode
        dataset = datasets.load_dataset(repo_id, subset, **kwargs)
        if image_decode is not None:
            image_feature = datasets.Image(decode=image_decode)
            column_names = set(getattr(dataset, "column_names", ()))
            for column in image_columns:
                if column in column_names:
                    dataset = dataset.cast_column(column, image_feature)
        if hasattr(dataset, "to_list"):
            return list(dataset.to_list())
        return [dict(row) for row in dataset]

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
