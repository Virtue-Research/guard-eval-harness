"""Tests for phase-2 image dataset adapters."""

from __future__ import annotations

import unittest
import base64
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.ai_vs_real import AIVsRealDataset
from guard_eval_harness.datasets.hateful_memes import HatefulMemesDataset
from guard_eval_harness.datasets.safe_vs_unsafe_image_edits import (
    SafeVsUnsafeImageEditsDataset,
)
from guard_eval_harness.datasets.self_harm_image_dataset import (
    SelfHarmImageDataset,
)
from guard_eval_harness.datasets.unsafebench import UnsafeBenchDataset
from guard_eval_harness.datasets.violence_image_dataset import (
    ViolenceImageDataset,
)


class SafeVsUnsafeImageEditsDatasetTest(unittest.TestCase):
    """Validate the public paired image edits adapter."""

    def test_load_emits_safe_and_unsafe_samples(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        safe = Image.new("RGB", (24, 24), color="green")
        unsafe = Image.new("RGB", (24, 24), color="red")
        config = ResolvedDatasetConfig(
            name="safe_vs_unsafe",
            adapter="safe_vs_unsafe_image_edits",
            split="train",
            options={"variant": "batch1", "no_sample_cache": True},
        )
        dataset = SafeVsUnsafeImageEditsDataset.from_config(config)

        with patch.object(
            dataset,
            "_load_rows",
            return_value=[
                {
                    "safe_image": safe,
                    "unsafe_image": unsafe,
                    "safe_caption": "safe caption",
                    "unsafe_caption": "unsafe caption",
                }
            ],
        ):
            samples = dataset.load()

        self.assertEqual(len(samples), 2)
        self.assertFalse(samples[0].label.unsafe)
        self.assertTrue(samples[1].label.unsafe)
        self.assertEqual(samples[0].messages[0].image_refs[0].modality, "image")
        self.assertEqual(samples[1].metadata["caption"], "unsafe caption")


class UnsafeBenchDatasetTest(unittest.TestCase):
    """Validate the UnsafeBench multimodal adapter."""

    def test_load_coerces_safe_label_and_category(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="unsafebench",
            adapter="unsafebench",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = UnsafeBenchDataset.from_config(config)
        safe = Image.new("RGB", (16, 16), color="white")
        unsafe = Image.new("RGB", (16, 16), color="black")

        with patch.object(
            dataset,
            "_load_rows",
            return_value=[
                {
                    "image": safe,
                    "safety_label": "Safe",
                    "category": "",
                    "source": "Laion5B",
                    "text": "safe prompt",
                },
                {
                    "image": unsafe,
                    "safety_label": "Violence",
                    "category": "Violence",
                    "source": "Lexica",
                    "text": "unsafe prompt",
                },
            ],
        ):
            samples = dataset.load()

        self.assertEqual(len(samples), 2)
        self.assertFalse(samples[0].label.unsafe)
        self.assertTrue(samples[1].label.unsafe)
        self.assertEqual(samples[1].category_labels, ("Violence",))
        self.assertEqual(samples[1].metadata["source"], "Lexica")


class ViolenceImageDatasetTest(unittest.TestCase):
    """Validate the GitHub violence image adapter."""

    def test_load_marks_repository_images_unsafe(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="violence_image_dataset",
            adapter="violence_image_dataset",
            split="train",
            options={"subset": "rgb", "no_sample_cache": True},
        )
        dataset = ViolenceImageDataset.from_config(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "fight.jpg"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            with patch.object(
                dataset,
                "_image_paths",
                return_value=["rgb/images/fight.jpg"],
            ), patch.object(
                dataset,
                "_download_image",
                return_value=image_path,
            ):
                samples = dataset.load()

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].label.unsafe)
        self.assertEqual(samples[0].category_labels, ("blood/gore",))
        self.assertEqual(samples[0].metadata["source_subset"], "rgb")


class SelfHarmImageDatasetTest(unittest.TestCase):
    """Validate the base64-backed self-harm dataset adapter."""

    def test_load_decodes_base64_and_maps_labels(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        buffer = io.BytesIO()
        Image.new("RGB", (12, 12), color="black").save(
            buffer,
            format="JPEG",
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        config = ResolvedDatasetConfig(
            name="self_harm_image_dataset",
            adapter="self_harm_image_dataset",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = SelfHarmImageDataset.from_config(config)

        with patch.object(
            dataset,
            "_load_hf_rows",
            return_value=[
                {
                    "index": 1,
                    "image_fname": "unsafe.jpg",
                    "label": 1,
                    "image": encoded,
                },
                {
                    "index": 2,
                    "image_fname": "safe.jpg",
                    "label": 0,
                    "image": encoded,
                },
            ],
        ):
            samples = dataset.load()

        self.assertEqual(len(samples), 2)
        self.assertTrue(samples[0].label.unsafe)
        self.assertFalse(samples[1].label.unsafe)
        self.assertEqual(samples[0].category_labels, ("blood/gore",))
        self.assertEqual(samples[1].category_labels, ())


class HatefulMemesDatasetTest(unittest.TestCase):
    """Validate Hateful Memes label and optional text mapping."""

    def test_load_maps_hateful_label_and_text_context(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="hateful_memes",
            adapter="hateful_memes",
            split="dev_seen",
            options={
                "include_text_context": True,
                "no_sample_cache": True,
            },
        )
        dataset = HatefulMemesDataset.from_config(config)
        image = Image.new("RGB", (20, 20), color="white")

        with patch.object(
            dataset,
            "_load_hf_rows",
            return_value=[
                {
                    "id": "123",
                    "text": "meme text",
                    "label": 1,
                    "image": image,
                }
            ],
        ):
            samples = dataset.load()

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].label.unsafe)
        self.assertEqual(
            samples[0].category_labels,
            ("hate/threatening/terrorism",),
        )
        self.assertIn("meme text", samples[0].messages[0].text_content)


class AIVsRealDatasetTest(unittest.TestCase):
    """Validate AI-vs-Real label mapping."""

    def test_load_treats_ai_generated_as_unsafe(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        config = ResolvedDatasetConfig(
            name="ai_vs_real",
            adapter="ai_vs_real",
            split="train",
            options={"no_sample_cache": True},
        )
        dataset = AIVsRealDataset.from_config(config)
        buffer = io.BytesIO()
        Image.new("RGB", (18, 18), color="blue").save(
            buffer,
            format="PNG",
        )
        image = {"bytes": buffer.getvalue(), "path": None}

        with patch.object(
            dataset,
            "_load_hf_rows",
            return_value=[
                {"binary_label": 0, "image": image},
                {"binary_label": 1, "image": image},
            ],
        ) as mock_load_hf_rows:
            samples = dataset.load()

        mock_load_hf_rows.assert_called_once_with(
            "Parveshiiii/AI-vs-Real",
            split="train",
            revision="bce7ac5b95c36c5013389341b94c75aa44882165",
            verification_mode="no_checks",
            image_decode=False,
        )
        self.assertEqual(len(samples), 2)
        self.assertTrue(samples[0].label.unsafe)
        self.assertFalse(samples[1].label.unsafe)
        self.assertEqual(samples[0].category_labels, ("genai/deepfakes",))
        self.assertNotIn("label_name", samples[1].metadata)
        self.assertNotIn("binary_label", samples[1].metadata)


if __name__ == "__main__":
    unittest.main()
