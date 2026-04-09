"""JailBreakV-28K multimodal jailbreak dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REVISION = "f949ca582fff13d396ac8fce59596afafb2b78d3"
_REPO_ID = "JailbreakV-28K/JailBreakV-28k"


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


def _bool_option(
    value: object,
    *,
    option_name: str,
    default: bool,
) -> bool:
    """Parse one optional bool-like dataset option."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{option_name} must be a boolean value")


@dataset_registry.register("jailbreakv_28k")
class JailBreakV28KDataset(SourceBackedMultimodalDatasetAdapter):
    """Load JailBreakV-28K multimodal attack prompts."""

    display_name = "JailBreakV-28K"
    source_uri = "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k"
    license_name = "MIT"
    version = _REVISION
    access_mode = "public_partial"
    languages = ("en",)
    supported_splits = ("mini", "train", "full")
    metadata_fields_to_preserve = (
        "id",
        "jailbreak_query",
        "redteam_query",
        "format",
        "policy",
        "image_path",
        "from",
        "selected_mini",
        "transfer_from_llm",
    )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize one JailBreakV-28K split."""
        self._ensure_supported_split()
        self._source_metadata = {
            **self._build_source_metadata(),
            "source_notes": (
                "The public Hugging Face snapshot is partial; some CSV rows "
                "reference image files that are not present in the repo."
            ),
        }
        skip_missing = _bool_option(
            self.config.options.get("skip_missing_images"),
            option_name="skip_missing_images",
            default=True,
        )
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "jailbreakv_28k",
                "revision": _REVISION,
                "split": self.config.split,
                "execution_limit": self.execution_limit,
                "skip_missing_images": skip_missing,
            },
            loader=self._load_uncached,
        )

    def _load_uncached(self) -> list[NormalizedSample]:
        config_name, split_name = self._hf_config_and_split()
        rows = self._load_hf_rows(
            _REPO_ID,
            split=split_name,
            subset=config_name,
            revision=_REVISION,
        )
        normalized: list[NormalizedSample] = []
        skip_missing_images = _bool_option(
            self.config.options.get("skip_missing_images"),
            option_name="skip_missing_images",
            default=True,
        )
        for row_index, row in enumerate(rows, start=1):
            image_path = _clean_text(row.get("image_path"))
            if not image_path:
                continue
            try:
                local_image_path = self._download_image(image_path)
            except FileNotFoundError:
                if skip_missing_images:
                    continue
                raise
            image_ref = self.resolve_image(local_image_path)
            policy = _clean_text(row.get("policy"))
            sample_id = self._make_sample_id(
                {
                    "id": row.get("id"),
                    "split": split_name,
                    "policy": policy,
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
                            text=_clean_text(row.get("jailbreak_query")),
                            image_ref=image_ref,
                        )
                    ],
                    label=UnsafeLabel(unsafe=True),
                    category_labels=(policy,) if policy else (),
                    metadata={
                        "source_id": row.get("id"),
                        "jailbreak_query": row.get("jailbreak_query"),
                        "redteam_query": row.get("redteam_query"),
                        "format": row.get("format"),
                        "policy": row.get("policy"),
                        "image_path": image_path,
                        "from": row.get("from"),
                        "selected_mini": row.get("selected_mini"),
                        "transfer_from_llm": row.get("transfer_from_llm"),
                    },
                )
            )
        return normalized

    def _hf_config_and_split(self) -> tuple[str, str]:
        """Map harness split names to HF config and split names."""
        split = self.config.split.strip().lower()
        if split == "mini":
            return "JailBreakV_28K", "mini_JailBreakV_28K"
        if split in {"train", "full"}:
            return "JailBreakV_28K", "JailBreakV_28K"
        raise ValueError(f"unsupported JailBreakV-28K split: {split}")

    def _download_image(self, image_path: str) -> Path:
        """Download one JailBreakV-28K image file into the HF cache."""
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "jailbreakv_28k requires the 'huggingface_hub' package"
            ) from exc

        try:
            local_path = hf_hub_download(
                repo_id=_REPO_ID,
                repo_type="dataset",
                filename=f"JailBreakV_28K/{image_path}",
                revision=_REVISION,
            )
        except EntryNotFoundError as exc:
            raise FileNotFoundError(
                f"JailBreakV-28K image not found in public repo: {image_path}"
            ) from exc
        return Path(local_path)
