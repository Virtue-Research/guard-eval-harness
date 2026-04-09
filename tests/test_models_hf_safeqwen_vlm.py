"""Tests for the SafeQwen VLM safety classifier adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import (
    ResolvedModelConfig,
)
from guard_eval_harness.models.hf_safeqwen_vlm import (
    SafeQwenVLMAdapter,
)
from guard_eval_harness.schemas import NormalizedSample


class _FakeBatch(dict):
    """Processor batch wrapper with a no-op ``.to()``."""

    def to(self, device):  # noqa: ANN001
        return self


def _image_sample(
    path: Path,
    *,
    sample_id: str = "sample-1",
    text: str = "Is this image safe?",
) -> NormalizedSample:
    """Build a single-image multimodal sample."""
    return NormalizedSample(
        id=sample_id,
        dataset="image-demo",
        split="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "media",
                        "media": {
                            "modality": "image",
                            "uri": path.as_posix(),
                        },
                    },
                ],
            }
        ],
        label={"unsafe": True},
    )


class SafeQwenVLMAdapterTest(unittest.TestCase):
    """Validate SafeQwen VLM adapter batch prediction."""

    def test_single_sample_safe_prediction(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "safe.png"
            Image.new("RGB", (32, 32), "green").save(
                img_path
            )
            sample = _image_sample(img_path)
            config = ResolvedModelConfig(
                adapter="hf_safeqwen_vlm",
                model_name="etri-vilab/SafeQwen2.5-VL-7B",
                args={"trust_remote_code": True},
            )
            adapter = SafeQwenVLMAdapter.from_config(
                config
            )

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def apply_chat_template(
                    self, messages, **kwargs
                ):  # noqa: ANN001
                    return "prompt-text"

                def __call__(self, **kwargs):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                            "attention_mask": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                        }
                    )

            # Safety probs: index 0 = safe (0.9),
            # rest are unsafe categories
            safety_probs = torch.zeros(1, 20)
            safety_probs[0, 0] = 0.9
            safety_probs[0, 1] = 0.05
            safety_probs[0, 17] = 0.05

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):  # noqa: ANN001
                    return SimpleNamespace(
                        img_safety_probs=safety_probs,
                    )

            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=_FakeProcessor(),
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                preds = adapter.predict_batch(
                    [sample], threshold=0.5
                )

        self.assertEqual(len(preds), 1)
        self.assertAlmostEqual(
            preds[0].unsafe_score, 0.1, places=5
        )
        self.assertFalse(preds[0].unsafe_label)

    def test_batch_with_varied_lengths(self) -> None:
        """Batch of two samples with different text."""
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_a = Path(tmpdir) / "a.png"
            img_b = Path(tmpdir) / "b.png"
            Image.new("RGB", (24, 24), "red").save(img_a)
            Image.new("RGB", (48, 48), "blue").save(img_b)
            short = _image_sample(
                img_a,
                sample_id="short",
                text="safe?",
            )
            long = _image_sample(
                img_b,
                sample_id="long",
                text="Please analyze this "
                "photograph for harmful content "
                "including violence.",
            )
            config = ResolvedModelConfig(
                adapter="hf_safeqwen_vlm",
                model_name=(
                    "etri-vilab/SafeQwen2.5-VL-7B"
                ),
                args={"trust_remote_code": True},
            )
            adapter = SafeQwenVLMAdapter.from_config(
                config
            )

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def apply_chat_template(
                    self, messages, **kwargs
                ):  # noqa: ANN001
                    return "prompt-text"

                def __call__(self, **kwargs):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (2, 5),
                                dtype=torch.long,
                            ),
                            "attention_mask": torch.ones(
                                (2, 5),
                                dtype=torch.long,
                            ),
                        }
                    )

            safety_probs = torch.zeros(2, 20)
            # Sample 0: safe (high safe prob)
            safety_probs[0, 0] = 0.85
            safety_probs[0, 14] = 0.15
            # Sample 1: unsafe (low safe prob,
            # high violence)
            safety_probs[1, 0] = 0.1
            safety_probs[1, 16] = 0.7
            safety_probs[1, 15] = 0.2

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):  # noqa: ANN001
                    return SimpleNamespace(
                        img_safety_probs=safety_probs,
                    )

            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=_FakeProcessor(),
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                preds = adapter.predict_batch(
                    [short, long], threshold=0.5
                )

        self.assertEqual(len(preds), 2)
        # First sample: safe
        self.assertFalse(preds[0].unsafe_label)
        self.assertAlmostEqual(
            preds[0].unsafe_score, 0.15, places=5
        )
        # Second sample: unsafe
        self.assertTrue(preds[1].unsafe_label)
        self.assertAlmostEqual(
            preds[1].unsafe_score, 0.9, places=5
        )
        self.assertEqual(
            preds[1].predicted_categories,
            ("weapon_related_violence",),
        )

    def test_empty_batch_returns_empty(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_safeqwen_vlm",
            model_name="etri-vilab/SafeQwen2.5-VL-7B",
        )
        adapter = SafeQwenVLMAdapter.from_config(config)
        preds = adapter.predict_batch([], threshold=0.5)
        self.assertEqual(preds, [])

    def test_include_label_scores(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (16, 16), "red").save(
                img_path
            )
            sample = _image_sample(img_path)
            config = ResolvedModelConfig(
                adapter="hf_safeqwen_vlm",
                model_name=(
                    "etri-vilab/SafeQwen2.5-VL-7B"
                ),
                args={
                    "trust_remote_code": True,
                    "include_label_scores": True,
                },
            )
            adapter = SafeQwenVLMAdapter.from_config(
                config
            )

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def apply_chat_template(
                    self, messages, **kwargs
                ):  # noqa: ANN001
                    return "prompt-text"

                def __call__(self, **kwargs):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                            "attention_mask": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                        }
                    )

            safety_probs = torch.zeros(1, 20)
            safety_probs[0, 0] = 0.2
            safety_probs[0, 17] = 0.8

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def __call__(self, **kwargs):  # noqa: ANN001
                    return SimpleNamespace(
                        img_safety_probs=safety_probs,
                    )

            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=_FakeProcessor(),
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                preds = adapter.predict_batch(
                    [sample], threshold=0.5
                )

        self.assertTrue(preds[0].unsafe_label)
        self.assertIn("safe", preds[0].category_scores)
        self.assertIn(
            "sexual_content",
            preds[0].category_scores,
        )


if __name__ == "__main__":
    unittest.main()
