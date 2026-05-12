"""Local directory adapter for image moderation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


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


@dataset_registry.register("local_image_dir")
class LocalImageDirDataset(MultimodalDatasetAdapter):
    """Load image-only or captioned images from a local directory tree."""

    def load(self) -> list[NormalizedSample]:
        """Load and normalize images from safe/unsafe directories."""
        root = self._resolve_root()
        self._source_metadata = self._current_source_metadata()
        split_root = (
            root / self.config.split
            if (root / self.config.split).exists()
            else root
        )
        label_directories = self._label_directories(split_root)

        normalized: list[NormalizedSample] = []
        row_number = 1
        for unsafe, directories in label_directories:
            for directory in directories:
                for image_path in self._iter_image_paths(directory):
                    image_ref = self.resolve_image(image_path)
                    caption = self._caption_for_image(image_path)
                    relative_path = image_path.relative_to(root).as_posix()
                    normalized.append(
                        NormalizedSample(
                            id=self._make_sample_id(
                                {
                                    "relative_path": relative_path,
                                    "unsafe": unsafe,
                                    "image_sha256": image_ref.sha256,
                                },
                                row_number,
                            ),
                            dataset=self.config.name,
                            split=self.config.split,
                            messages=[
                                self.build_multimodal_message(
                                    text=caption or None,
                                    image_ref=image_ref,
                                )
                            ],
                            label=UnsafeLabel(unsafe=unsafe),
                            metadata={
                                "relative_image_path": relative_path,
                            },
                        )
                    )
                    row_number += 1

        if not normalized:
            raise ValueError(
                f"local_image_dir found no images under {split_root}"
            )
        return normalized

    def _resolve_root(self) -> Path:
        """Resolve the local dataset root directory."""
        if self.config.path is None:
            raise ValueError("local_image_dir requires a path")
        root = Path(self.config.path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"dataset source not found: {root}")
        if not root.is_dir():
            raise ValueError(f"local_image_dir requires a directory: {root}")
        return root

    def _label_directories(
        self,
        split_root: Path,
    ) -> list[tuple[bool, list[Path]]]:
        """Resolve the configured safe and unsafe label directories."""
        safe_dirs = self._named_directories(
            split_root,
            option_name="safe_dirs",
            default_names=("safe",),
        )
        unsafe_dirs = self._named_directories(
            split_root,
            option_name="unsafe_dirs",
            default_names=("unsafe",),
        )
        if not safe_dirs and not unsafe_dirs:
            raise FileNotFoundError(
                "local_image_dir expected configured safe/unsafe directories under "
                f"{split_root}"
            )
        return [
            (False, safe_dirs),
            (True, unsafe_dirs),
        ]

    def _named_directories(
        self,
        split_root: Path,
        *,
        option_name: str,
        default_names: tuple[str, ...],
    ) -> list[Path]:
        """Resolve one configured set of label directories."""
        configured = self.config.options.get(option_name, default_names)
        if isinstance(configured, str):
            names = [name.strip() for name in configured.split(",")]
        elif isinstance(configured, Sequence):
            names = [str(name).strip() for name in configured]
        else:
            raise ValueError(f"{option_name} must be a string or list")
        directories = []
        for name in names:
            if not name:
                continue
            candidate = split_root / name
            if candidate.exists() and candidate.is_dir():
                directories.append(candidate)
        return directories

    def _iter_image_paths(self, directory: Path) -> list[Path]:
        """Return sorted image paths beneath one label directory."""
        recursive = _bool_option(
            self.config.options.get("recursive"),
            option_name="recursive",
            default=True,
        )
        if recursive:
            iterator = directory.rglob("*")
        else:
            iterator = directory.glob("*")
        return sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )

    def _caption_for_image(self, image_path: Path) -> str:
        """Load an optional `.txt` caption sidecar."""
        if not _bool_option(
            self.config.options.get("caption_sidecars"),
            option_name="caption_sidecars",
            default=True,
        ):
            return ""
        sidecar_path = image_path.with_suffix(".txt")
        if not sidecar_path.exists():
            return ""
        return sidecar_path.read_text(encoding="utf-8").strip()
