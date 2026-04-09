"""Tests for the ImageNet 1K validation (all-safe) dataset adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.imagenet1k_val_safe import (
    ImageNet1KValSafeDataset,
)


class ImageNet1KValSafeDatasetTest(unittest.TestCase):
    """Validate ImageNet 1K loads as all-safe image samples."""

    def test_load_marks_all_samples_safe(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="imagenet1k_val_safe",
            adapter="imagenet1k_val_safe",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = ImageNet1KValSafeDataset.from_config(config)

        images = [
            Image.new("RGB", (24, 24), color=c)
            for c in ("green", "blue", "red")
        ]
        rows = [
            {"image": images[0], "label": 0},
            {"image": images[1], "label": 42},
            {"image": images[2], "label": 999},
        ]

        with patch.object(dataset, "_load_rows", return_value=rows):
            samples = dataset.load()

        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertFalse(
                sample.label.unsafe,
                "ImageNet samples must be labelled safe",
            )
            self.assertEqual(len(sample.messages), 1)
            self.assertEqual(
                sample.messages[0].image_refs[0].modality, "image"
            )

    def test_preserves_class_id_in_metadata(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="imagenet1k_val_safe",
            adapter="imagenet1k_val_safe",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = ImageNet1KValSafeDataset.from_config(config)

        rows = [
            {"image": Image.new("RGB", (16, 16), "white"), "label": 123},
        ]

        with patch.object(dataset, "_load_rows", return_value=rows):
            samples = dataset.load()

        self.assertEqual(samples[0].metadata["imagenet_class_id"], 123)

    def test_skips_rows_with_missing_image(self) -> None:
        config = ResolvedDatasetConfig(
            name="imagenet1k_val_safe",
            adapter="imagenet1k_val_safe",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = ImageNet1KValSafeDataset.from_config(config)

        rows = [
            {"image": None, "label": 0},
            {"label": 42},
        ]

        with patch.object(dataset, "_load_rows", return_value=rows):
            samples = dataset.load()

        self.assertEqual(len(samples), 0)

    def test_rejects_unsupported_split(self) -> None:
        config = ResolvedDatasetConfig(
            name="imagenet1k_val_safe",
            adapter="imagenet1k_val_safe",
            split="test",
        )
        dataset = ImageNet1KValSafeDataset.from_config(config)

        with self.assertRaises(ValueError):
            dataset.load()

    def test_deterministic_sample_ids(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="imagenet1k_val_safe",
            adapter="imagenet1k_val_safe",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = ImageNet1KValSafeDataset.from_config(config)
        img = Image.new("RGB", (16, 16), "gray")

        with patch.object(
            dataset, "_load_rows", return_value=[{"image": img, "label": 7}]
        ):
            first = dataset.load()

        with patch.object(
            dataset, "_load_rows", return_value=[{"image": img, "label": 7}]
        ):
            second = dataset.load()

        self.assertEqual(first[0].id, second[0].id)


if __name__ == "__main__":
    unittest.main()
