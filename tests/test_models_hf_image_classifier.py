"""Tests for the Hugging Face image-classification adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.hf_image_classifier import (
    HuggingFaceImageClassifierAdapter,
)
from guard_eval_harness.schemas import MediaPart, MediaRef, NormalizedSample, TextPart


def _image_sample(path: Path, *, sample_id: str = "image-1") -> NormalizedSample:
    """Build a one-image normalized sample."""
    return NormalizedSample(
        id=sample_id,
        dataset="image-demo",
        split="train",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "media",
                        "media": {
                            "modality": "image",
                            "uri": path.as_posix(),
                        },
                    }
                ],
            }
        ],
        label={"unsafe": True},
    )


class HuggingFaceImageClassifierAdapterTest(unittest.TestCase):
    """Validate image batch scoring and label mapping."""

    def test_predict_batch_scores_falconsai_style_labels(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 48), color="red").save(image_path)
            sample = _image_sample(image_path)
            config = ResolvedModelConfig(
                adapter="hf_image_classifier",
                model_name="Falconsai/nsfw_image_detection",
                args={"label_score_mapping": {"normal": 0.0, "nsfw": 1.0}},
            )
            adapter = HuggingFaceImageClassifierAdapter.from_config(config)

            fake_processor = lambda images, return_tensors: {  # noqa: E731
                "pixel_values": torch.ones((len(images), 3, 32, 32))
            }
            class _FakeModel:
                config = SimpleNamespace(id2label={0: "normal", 1: "nsfw"})

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):
                    return SimpleNamespace(
                        logits=torch.tensor(
                            [[0.1, 2.0]], dtype=torch.float32
                        )
                    )

            fake_model = _FakeModel()
            with (
                patch.object(adapter, "_get_image_processor", return_value=fake_processor),
                patch.object(
                    adapter,
                    "_get_image_model",
                    return_value=(fake_model, None),
                ),
            ):
                predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        self.assertTrue(predictions[0].unsafe_label)
        expected_score = float(torch.softmax(torch.tensor([0.1, 2.0]), dim=0)[1])
        self.assertAlmostEqual(predictions[0].unsafe_score, expected_score, places=5)
        self.assertLess(predictions[0].unsafe_score, 1.0)
        self.assertEqual(predictions[0].metadata["top_label"], "nsfw")

    def test_multiple_images_raise(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_image_classifier",
            model_name="demo",
        )
        adapter = HuggingFaceImageClassifierAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="image-demo",
            split="train",
            messages=[
                {
                    "role": "user",
                    "content": [
                        MediaPart(
                            media=MediaRef(modality="image", uri="/tmp/1.png")
                        ),
                        TextPart(text="ignored"),
                        MediaPart(
                            media=MediaRef(modality="image", uri="/tmp/2.png")
                        ),
                    ],
                }
            ],
            label={"unsafe": False},
        )

        with self.assertRaisesRegex(ValueError, "multiple images"):
            adapter._image_for_sample(sample)

    def test_predict_batch_drops_unreadable_images_when_partial_allowed(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 48), color="red").save(image_path)
            good = _image_sample(image_path, sample_id="good")
            bad = _image_sample(image_path, sample_id="bad")
            config = ResolvedModelConfig(
                adapter="hf_image_classifier",
                model_name="Falconsai/nsfw_image_detection",
            )
            adapter = HuggingFaceImageClassifierAdapter.from_config(config)

            class _FakeModel:
                config = SimpleNamespace(id2label={0: "normal", 1: "nsfw"})

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):
                    return SimpleNamespace(
                        logits=torch.tensor(
                            [[0.1, 2.0]], dtype=torch.float32
                        )
                    )

            fake_model = _FakeModel()
            fake_processor = lambda images, return_tensors: {  # noqa: E731
                "pixel_values": torch.ones((len(images), 3, 32, 32))
            }
            with (
                patch.object(adapter, "_get_image_processor", return_value=fake_processor),
                patch.object(
                    adapter,
                    "_get_image_model",
                    return_value=(fake_model, None),
                ),
                patch.object(
                    adapter,
                    "_image_for_sample",
                    side_effect=[
                        ValueError("corrupt image"),
                        Image.open(image_path).copy(),
                    ],
                ),
            ):
                predictions = adapter.predict_batch(
                    [bad, good],
                    threshold=0.5,
                )

        self.assertEqual([prediction.sample_id for prediction in predictions], ["good"])

    def test_predict_batch_uses_sigmoid_for_multi_label_models(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 48), color="red").save(image_path)
            sample = _image_sample(image_path)
            config = ResolvedModelConfig(
                adapter="hf_image_classifier",
                model_name="multi-label-demo",
                args={"label_score_mapping": {"unsafe": 1.0, "safe": 0.0}},
            )
            adapter = HuggingFaceImageClassifierAdapter.from_config(config)

            fake_processor = lambda images, return_tensors: {  # noqa: E731
                "pixel_values": torch.ones((len(images), 3, 32, 32))
            }

            class _FakeModel:
                config = SimpleNamespace(
                    id2label={0: "unsafe", 1: "safe"},
                    problem_type="multi_label_classification",
                )

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):
                    return SimpleNamespace(
                        logits=torch.tensor(
                            [[1.0, 1.0]], dtype=torch.float32
                        )
                    )

            fake_model = _FakeModel()
            with (
                patch.object(adapter, "_get_image_processor", return_value=fake_processor),
                patch.object(
                    adapter,
                    "_get_image_model",
                    return_value=(fake_model, None),
                ),
            ):
                predictions = adapter.predict_batch([sample], threshold=0.5)

        expected_score = float(torch.sigmoid(torch.tensor(1.0)))
        self.assertAlmostEqual(predictions[0].unsafe_score, expected_score, places=5)


    def test_empty_score_rows_raises_descriptive_error(self) -> None:
        """Empty model output raises ValueError, not IndexError."""
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 48), color="red").save(image_path)
            sample = _image_sample(image_path)
            config = ResolvedModelConfig(
                adapter="hf_image_classifier",
                model_name="Falconsai/nsfw_image_detection",
                args={
                    "label_score_mapping": {
                        "normal": 0.0,
                        "nsfw": 1.0,
                    },
                },
            )
            adapter = (
                HuggingFaceImageClassifierAdapter.from_config(config)
            )

            def fake_processor(images, return_tensors):  # noqa: E731
                return {
                    "pixel_values": torch.ones(
                        (len(images), 3, 32, 32)
                    )
                }

            class _FakeModel:
                config = SimpleNamespace(
                    id2label={0: "normal", 1: "nsfw"},
                )

                def eval(self):
                    return None

                def __call__(self, **kwargs):
                    return SimpleNamespace(
                        logits=torch.empty(
                            (0, 2), dtype=torch.float32
                        ),
                    )

            fake_model = _FakeModel()
            with (
                patch.object(
                    adapter,
                    "_get_image_processor",
                    return_value=fake_processor,
                ),
                patch.object(
                    adapter,
                    "_get_image_model",
                    return_value=(fake_model, None),
                ),
            ):
                with self.assertRaises(ValueError) as ctx:
                    adapter.predict_batch(
                        [sample], threshold=0.5
                    )
                self.assertIn(
                    "empty logits", str(ctx.exception)
                )


if __name__ == "__main__":
    unittest.main()
