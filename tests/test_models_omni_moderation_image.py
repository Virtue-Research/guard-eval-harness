"""Smoke tests for omni-moderation with image-only inputs.

The existing test_models_openai_moderation.py covers text+image and
text-only requests. These tests add coverage for:
- Image-only samples (no text content, e.g. ImageNet)
- Batched predictions (batch_size > 1)
- Safe vs unsafe scoring on image-only inputs
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.openai_moderation import (
    OpenAIModerationAdapter,
)
from guard_eval_harness.schemas import NormalizedSample

_MOCK_PATCH = (
    "guard_eval_harness.models.openai_moderation.json_post_with_retry"
)


def _image_only_sample(
    image_path: Path,
    *,
    sample_id: str = "img-1",
    unsafe: bool = False,
) -> NormalizedSample:
    """Build a sample with only an image (no text)."""
    return NormalizedSample(
        id=sample_id,
        dataset="demo",
        split="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "media",
                        "media": {
                            "modality": "image",
                            "uri": image_path.as_posix(),
                        },
                    },
                ],
            }
        ],
        label={"unsafe": unsafe},
    )


def _moderation_response(
    *,
    flagged: bool = False,
    violence: float = 0.01,
    sexual: float = 0.01,
) -> dict:
    return {
        "id": "modr-test",
        "model": "omni-moderation-latest",
        "results": [
            {
                "flagged": flagged,
                "categories": {
                    "violence": violence > 0.5,
                    "sexual": sexual > 0.5,
                },
                "category_scores": {
                    "violence": violence,
                    "sexual": sexual,
                },
            }
        ],
    }


class OmniModerationImageOnlyTest(unittest.TestCase):
    """Validate omni-moderation handles image-only samples."""

    def setUp(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        self._tmpdir = tempfile.TemporaryDirectory()
        tmpdir = Path(self._tmpdir.name)
        self._safe_path = tmpdir / "safe.png"
        self._unsafe_path = tmpdir / "unsafe.png"
        Image.new("RGB", (16, 16), color="green").save(self._safe_path)
        Image.new("RGB", (16, 16), color="red").save(self._unsafe_path)

        self._config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={"url": "https://example.test/v1/moderations"},
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_image_only_sample_sends_image_url(self) -> None:
        adapter = OpenAIModerationAdapter.from_config(self._config)
        sample = _image_only_sample(self._safe_path)

        with patch(
            _MOCK_PATCH,
            return_value=_moderation_response(flagged=False),
        ) as mock_post:
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        payload = mock_post.call_args.args[1]
        image_parts = [
            p for p in payload["input"] if p["type"] == "image_url"
        ]
        self.assertGreaterEqual(len(image_parts), 1)
        self.assertTrue(
            image_parts[0]["image_url"]["url"].startswith(
                "data:image/"
            )
        )

    def test_safe_image_scores_low(self) -> None:
        adapter = OpenAIModerationAdapter.from_config(self._config)
        sample = _image_only_sample(self._safe_path, unsafe=False)

        with patch(
            _MOCK_PATCH,
            return_value=_moderation_response(
                flagged=False, violence=0.02, sexual=0.01
            ),
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertLess(predictions[0].unsafe_score, 0.5)
        self.assertFalse(predictions[0].unsafe_label)

    def test_unsafe_image_scores_high(self) -> None:
        adapter = OpenAIModerationAdapter.from_config(self._config)
        sample = _image_only_sample(
            self._unsafe_path, sample_id="img-unsafe", unsafe=True
        )

        with patch(
            _MOCK_PATCH,
            return_value=_moderation_response(
                flagged=True, violence=0.9, sexual=0.1
            ),
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertGreater(predictions[0].unsafe_score, 0.5)
        self.assertTrue(predictions[0].unsafe_label)

    def test_batch_of_two_images(self) -> None:
        """Batch size > 1 must produce one prediction per sample."""
        adapter = OpenAIModerationAdapter.from_config(self._config)
        samples = [
            _image_only_sample(
                self._safe_path, sample_id="s1", unsafe=False
            ),
            _image_only_sample(
                self._unsafe_path, sample_id="s2", unsafe=True
            ),
        ]

        with patch(
            _MOCK_PATCH,
            side_effect=[
                _moderation_response(
                    flagged=False, violence=0.01
                ),
                _moderation_response(
                    flagged=True, violence=0.95
                ),
            ],
        ) as mock_post:
            predictions = adapter.predict_batch(
                samples, threshold=0.5
            )

        self.assertEqual(len(predictions), 2)
        self.assertEqual(mock_post.call_count, 2)
        ids = {p.sample_id for p in predictions}
        self.assertEqual(ids, {"s1", "s2"})


if __name__ == "__main__":
    unittest.main()
