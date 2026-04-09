"""Tests for the Hugging Face ShieldGemma 2 adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.hf_shieldgemma2 import (
    HuggingFaceShieldGemma2Adapter,
)
from guard_eval_harness.schemas import NormalizedSample


class _FakeBatch(dict):
    """Simple processor batch wrapper with a no-op `.to()`."""

    def to(self, device):  # noqa: ANN001
        return self


def _image_sample(
    path: Path,
    *,
    sample_id: str = "sample-1",
    text: str = "",
) -> NormalizedSample:
    """Build a one-image sample with optional text content."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append(
        {
            "type": "media",
            "media": {
                "modality": "image",
                "uri": path.as_posix(),
            },
        }
    )
    return NormalizedSample(
        id=sample_id,
        dataset="image-demo",
        split="test",
        messages=[{"role": "user", "content": content}],
        label={"unsafe": True},
    )


class HuggingFaceShieldGemma2AdapterTest(unittest.TestCase):
    """Validate policy aggregation and text-handling behavior."""

    def test_predict_batch_aggregates_policy_scores(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (28, 28), color="orange").save(image_path)
            sample_a = _image_sample(image_path, sample_id="sample-a")
            sample_b = _image_sample(image_path, sample_id="sample-b")
            config = ResolvedModelConfig(
                adapter="hf_shieldgemma2",
                model_name="google/shieldgemma-2-4b-it",
                args={"policies": ["dangerous", "violence"]},
            )
            adapter = HuggingFaceShieldGemma2Adapter.from_config(config)

            class _FakeProcessor:
                def __init__(self) -> None:
                    self.captured_kwargs = None

                def __call__(self, **kwargs):  # noqa: ANN001
                    self.captured_kwargs = kwargs
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones((4, 3), dtype=torch.long),
                            "attention_mask": torch.ones(
                                (4, 3), dtype=torch.long
                            ),
                        }
                    )

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):
                    return SimpleNamespace(
                        probabilities=torch.tensor(
                            [
                                [0.1, 0.9],
                                [0.7, 0.3],
                                [0.4, 0.6],
                                [0.2, 0.8],
                            ],
                            dtype=torch.float32,
                        )
                    )

            fake_processor = _FakeProcessor()
            fake_model = _FakeModel()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_processor,
                ),
                patch.object(
                    adapter,
                    "_get_model",
                    return_value=(fake_model, None),
                ),
            ):
                predictions = adapter.predict_batch(
                    [sample_a, sample_b],
                    threshold=0.5,
                )

        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.7)
        self.assertTrue(predictions[0].unsafe_label)
        self.assertEqual(
            predictions[0].predicted_categories,
            ("violence",),
        )
        self.assertAlmostEqual(
            predictions[0].category_scores["dangerous"],
            0.1,
        )
        self.assertAlmostEqual(predictions[1].unsafe_score, 0.4)
        self.assertFalse(predictions[1].unsafe_label)
        self.assertEqual(
            fake_processor.captured_kwargs["policies"],
            ["dangerous", "violence"],
        )

    def test_rejects_text_input_without_explicit_opt_in(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (20, 20), color="green").save(image_path)
            sample = _image_sample(image_path, text="Check this image")
            config = ResolvedModelConfig(
                adapter="hf_shieldgemma2",
                model_name="google/shieldgemma-2-4b-it",
                args={"policies": ["dangerous"]},
            )
            adapter = HuggingFaceShieldGemma2Adapter.from_config(config)

            with self.assertRaisesRegex(ValueError, "does not support text"):
                adapter._image_for_sample(sample)

    def test_policy_aliases_match_current_processor_keys(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_shieldgemma2",
            model_name="google/shieldgemma-2-4b-it",
            args={"policies": ["sexually_explicit", "dangerous"]},
        )
        adapter = HuggingFaceShieldGemma2Adapter.from_config(config)

        policies = adapter._resolved_policies(SimpleNamespace())

        self.assertEqual(policies, ["sexual", "dangerous"])


if __name__ == "__main__":
    unittest.main()
