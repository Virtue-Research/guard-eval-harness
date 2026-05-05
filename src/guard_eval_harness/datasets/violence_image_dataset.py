"""ChinaZhangPeng Violence-Image-Dataset adapter."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from guard_eval_harness.datasets.multimodal_source_backed import (
    SourceBackedMultimodalDatasetAdapter,
)
from guard_eval_harness.datasets.source_backed import (
    cached_download,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_REPO = "ChinaZhangPeng/Violence-Image-Dataset"
_REVISION = "79fef289d093658ba2caeaead4a5243d7755353a"
_TREE_URL = (
    f"https://api.github.com/repos/{_REPO}/git/trees/{_REVISION}?recursive=1"
)
_RAW_ROOT = f"https://raw.githubusercontent.com/{_REPO}/{_REVISION}"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


@dataset_registry.register("violence_image_dataset")
class ViolenceImageDataset(SourceBackedMultimodalDatasetAdapter):
    """Load violence images from the public GitHub repository."""

    display_name = "Violence Image Dataset"
    source_uri = f"https://github.com/{_REPO}"
    license_name = "Unknown"
    categories = ("blood/gore",)
    supported_splits = ("train",)
    version = _REVISION

    def load(self) -> list[NormalizedSample]:
        """Load and normalize configured image files."""
        self._ensure_supported_split()
        subset = self._subset()
        self._source_metadata = {
            **self._build_source_metadata(),
            "label_mapping": "All repository images are treated as unsafe blood/gore-category samples.",
            "github_revision": _REVISION,
            "subset": subset,
        }
        return self._load_with_sample_cache(
            cache_key_parts={
                "adapter": "violence_image_dataset",
                "revision": _REVISION,
                "split": self.config.split,
                "subset": subset,
                "execution_limit": self.execution_limit,
            },
            loader=lambda: self._load_uncached(subset=subset),
        )

    def _subset(self) -> str:
        subset = str(self.config.options.get("subset", "rgb")).strip().lower()
        if subset not in {"rgb", "skeleton", "all"}:
            raise ValueError(
                "violence_image_dataset supports options.subset values: "
                "rgb, skeleton, all"
            )
        return subset

    def _load_uncached(self, *, subset: str) -> list[NormalizedSample]:
        paths = self._image_paths(subset=subset)
        if self.execution_limit is not None:
            paths = paths[: self.execution_limit]

        normalized: list[NormalizedSample] = []
        for row_index, image_path in enumerate(paths, start=1):
            local_path = self._download_image(image_path)
            image_ref = self.resolve_image(local_path)
            sample_id = self._make_sample_id(
                {
                    "row_index": row_index,
                    "image_path": image_path,
                    "image_sha256": image_ref.sha256,
                    "revision": _REVISION,
                },
                row_index,
            )
            normalized.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=[
                        self.build_multimodal_message(image_ref=image_ref)
                    ],
                    label=UnsafeLabel(unsafe=True),
                    category_labels=("blood/gore",),
                    metadata={
                        "image_path": image_path,
                        "source_subset": image_path.split("/", 1)[0],
                        "github_revision": _REVISION,
                        "image_sha256": image_ref.sha256,
                    },
                )
            )
        return normalized

    def _image_paths(self, *, subset: str) -> list[str]:
        payload = load_json_payload(
            alias="violence_image_dataset",
            url=_TREE_URL,
            filename="tree.json",
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("tree"), list
        ):
            raise ValueError("GitHub tree response did not include tree[]")

        paths: list[str] = []
        for item in payload["tree"]:
            if not isinstance(item, dict) or item.get("type") != "blob":
                continue
            path = str(item.get("path", ""))
            if not path.lower().endswith(_IMAGE_SUFFIXES):
                continue
            if subset != "all" and not path.startswith(f"{subset}/"):
                continue
            paths.append(path)
        return sorted(paths)

    def _download_image(self, image_path: str) -> Path:
        encoded_path = quote(image_path, safe="/")
        return cached_download(
            alias="violence_image_dataset",
            url=f"{_RAW_ROOT}/{encoded_path}",
            filename=Path(image_path).name,
        )
