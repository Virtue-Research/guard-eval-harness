"""Hateful Memes image dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REPO = "biecho/hateful_memes"
_REVISION = "a3b0091dc58efd09d3661335fe0d764ceb839690"


@dataset_registry.register("hateful_memes")
class HatefulMemesDataset(SourceBackedMultimodalDatasetAdapter):
    """Load Hateful Memes as binary image-safety samples."""

    display_name = "Hateful Memes"
    source_uri = f"https://huggingface.co/datasets/{_REPO}"
    license_name = "Meta Hateful Memes license / non-commercial research"
    categories = ("hate/threatening/terrorism",)
    supported_splits = (
        "train",
        "dev_seen",
        "dev_unseen",
        "test_seen",
        "test_unseen",
    )
    version = _REVISION
    metadata_fields_to_preserve = ("id", "text", "label")

    def load(self) -> list[NormalizedSample]:
        """Load and normalize the configured split."""
        self._ensure_supported_split()
        include_text = self._include_text_context()
        self._source_metadata = {
            **self._build_source_metadata(),
            "label_mapping": "label=1 -> unsafe hateful meme; label=0 -> non-hateful.",
            "hf_repo_id": _REPO,
            "hf_revision": _REVISION,
            "include_text_context": include_text,
        }
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "hateful_memes",
                "revision": _REVISION,
                "split": self.config.split,
                "include_text_context": include_text,
                "execution_limit": self.execution_limit,
            },
            loader=lambda: self._load_uncached(include_text=include_text),
        )

    def _include_text_context(self) -> bool:
        raw = self.config.options.get("include_text_context", False)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    def _load_uncached(
        self,
        *,
        include_text: bool,
    ) -> list[NormalizedSample]:
        rows = self._load_hf_rows(
            _REPO,
            split=self.config.split,
            revision=_REVISION,
        )
        normalized: list[NormalizedSample] = []
        for row_index, row in enumerate(rows, start=1):
            image = row.get("image")
            if image is None:
                continue
            label = int(row.get("label", 0))
            text = str(row.get("text", "") or "").strip()
            image_ref = self.resolve_image(image)
            unsafe = label == 1
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "id": row.get("id"),
                    "label": label,
                    "text": text,
                    "image_sha256": image_ref.sha256,
                },
                row_index,
            )
            normalized.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=[
                        self.build_multimodal_message(
                            text=text if include_text else None,
                            image_ref=image_ref,
                        )
                    ],
                    label=UnsafeLabel(unsafe=unsafe),
                    category_labels=(
                        ("hate/threatening/terrorism",) if unsafe else ()
                    ),
                    metadata={
                        "id": row.get("id"),
                        "text": text,
                        "label": label,
                        "image_sha256": image_ref.sha256,
                    },
                )
            )
        return normalized
