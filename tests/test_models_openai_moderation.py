"""Tests for the OpenAI moderation adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.openai_moderation import (
    _MAX_CONCURRENCY,
    OpenAIModerationAdapter,
)
from guard_eval_harness.schemas import NormalizedSample

_MOCK_PATCH = "guard_eval_harness.models.openai_moderation.json_post_with_retry"


class OpenAIModerationAdapterTest(unittest.TestCase):
    """Validate moderation request and response handling."""

    def test_api_concurrency_cap_allows_benchmark_throughput(self) -> None:
        self.assertGreaterEqual(_MAX_CONCURRENCY, 500)

    def test_predict_batch_builds_text_image_request(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            sample = NormalizedSample(
                id="sample-1",
                dataset="demo",
                split="test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check this image"},
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
                label={"unsafe": True},
            )
            config = ResolvedModelConfig(
                adapter="openai_moderation",
                model_name="omni-moderation-latest",
                args={"url": "https://example.test/v1/moderations"},
            )
            adapter = OpenAIModerationAdapter.from_config(config)

            with patch(
                _MOCK_PATCH,
                return_value={
                    "id": "modr-123",
                    "model": "omni-moderation-latest",
                    "results": [
                        {
                            "flagged": True,
                            "categories": {"violence": True, "sexual": False},
                            "category_scores": {
                                "violence": 0.8,
                                "sexual": 0.1,
                            },
                        }
                    ],
                },
            ) as mock_post:
                predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        payload = mock_post.call_args.args[1]
        self.assertEqual(payload["model"], "omni-moderation-latest")
        self.assertEqual(payload["input"][0]["type"], "text")
        self.assertEqual(payload["input"][1]["type"], "image_url")
        self.assertTrue(
            payload["input"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.8)
        self.assertEqual(predictions[0].predicted_categories, ("violence",))
        self.assertEqual(predictions[0].metadata["request_id"], "modr-123")

    def test_categories_subset_limits_scores(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://example.test/v1/moderations",
                "categories": ["violence"],
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={
                "results": [
                    {
                        "flagged": True,
                        "categories": {"violence": False, "sexual": True},
                        "category_scores": {
                            "violence": 0.4,
                            "sexual": 0.9,
                        },
                    }
                ]
            },
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.4)
        self.assertEqual(predictions[0].predicted_categories, ())
        self.assertEqual(
            predictions[0].category_scores,
            {"violence": 0.4, "sexual": 0.9},
        )

    def test_any_category_aggregation_uses_booleans(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://example.test/v1/moderations",
                "binary_aggregation": "any_category",
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={
                "results": [
                    {
                        "flagged": False,
                        "categories": {"violence": True},
                        "category_scores": {"violence": 0.2},
                    }
                ]
            },
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].unsafe_score, 1.0)
        self.assertTrue(predictions[0].unsafe_label)

    def test_partial_predictions_drop_failed_samples(self) -> None:
        samples = []
        for index in range(2):
            samples.append(
                NormalizedSample(
                    id=f"sample-{index}",
                    dataset="demo",
                    split="test",
                    messages=[{"role": "user", "content": f"Check {index}"}],
                    label={"unsafe": True},
                )
            )
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={"url": "https://example.test/v1/moderations"},
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            side_effect=[
                ValueError("bad response"),
                {
                    "results": [
                        {
                            "flagged": True,
                            "categories": {"violence": True},
                            "category_scores": {"violence": 0.8},
                        }
                    ]
                },
            ],
        ):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["sample-1"],
        )


    def test_flagged_category_absent_from_scores_contributes_zero(
        self,
    ) -> None:
        """Category flagged True but missing from category_scores
        must still appear in selected_scores (as 0.0) so it affects
        unsafe_score aggregation symmetrically with
        predicted_categories.
        """
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://example.test/v1/moderations",
                "binary_aggregation": "mean_score",
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={
                "results": [
                    {
                        "flagged": True,
                        "categories": {
                            "violence": True,
                            "harassment": True,
                        },
                        "category_scores": {
                            "violence": 0.9,
                            # harassment deliberately absent
                        },
                    }
                ]
            },
        ):
            predictions = adapter.predict_batch(
                [sample], threshold=0.5
            )

        # harassment should contribute 0.0 -> mean = (0.9+0.0)/2
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.45)
        self.assertIn(
            "harassment", predictions[0].predicted_categories
        )


    def test_missing_api_key_fails_fast(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        # No url override -> default api.openai.com endpoint requires a key.
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={},
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch(_MOCK_PATCH) as mock_post:
                with self.assertRaises(ValueError) as ctx:
                    adapter.predict_batch([sample], threshold=0.5)

        self.assertIn("API key is missing", str(ctx.exception))
        mock_post.assert_not_called()

    def test_auth_error_raises_instead_of_dropping_serial(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://api.openai.com/v1/moderations",
                "api_key": "sk-test",
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        http_error = urllib_error.HTTPError(
            "https://api.openai.com/v1/moderations",
            401,
            "Unauthorized",
            {},
            None,
        )
        # drop_failed_predictions defaults True: without the fast-fail this
        # would silently return an empty list. It must now raise instead.
        with patch(_MOCK_PATCH, side_effect=http_error):
            with self.assertRaises(ValueError) as ctx:
                adapter.predict_batch([sample], threshold=0.5)

        self.assertIn("authentication failed", str(ctx.exception))

    def test_auth_error_raises_in_concurrent_path(self) -> None:
        samples = [
            NormalizedSample(
                id=f"sample-{index}",
                dataset="demo",
                split="test",
                messages=[{"role": "user", "content": f"Check {index}"}],
                label={"unsafe": True},
            )
            for index in range(2)
        ]
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://api.openai.com/v1/moderations",
                "api_key": "sk-test",
                "concurrency": 4,
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        http_error = urllib_error.HTTPError(
            "https://api.openai.com/v1/moderations",
            403,
            "Forbidden",
            {},
            None,
        )
        with patch(_MOCK_PATCH, side_effect=http_error) as mock_post:
            with self.assertRaises(ValueError) as ctx:
                adapter.predict_batch(samples, threshold=0.5)

        self.assertIn("authentication failed", str(ctx.exception))
        # Fail fast: the single-request probe must abort before the rest of
        # the (concurrency=4) batch is dispatched.
        self.assertEqual(mock_post.call_count, 1)

    def test_concurrent_path_probes_then_merges_results(self) -> None:
        samples = [
            NormalizedSample(
                id=f"sample-{index}",
                dataset="demo",
                split="test",
                messages=[{"role": "user", "content": f"Check {index}"}],
                label={"unsafe": True},
            )
            for index in range(3)
        ]
        config = ResolvedModelConfig(
            adapter="openai_moderation",
            model_name="omni-moderation-latest",
            args={
                "url": "https://api.openai.com/v1/moderations",
                "api_key": "sk-test",
                "concurrency": 4,
            },
        )
        adapter = OpenAIModerationAdapter.from_config(config)

        response = {
            "results": [
                {
                    "flagged": True,
                    "categories": {"violence": True},
                    "category_scores": {"violence": 0.8},
                }
            ]
        }
        # Probe handles sample-0, the pool handles the rest; results must come
        # back in the original sample order.
        with patch(_MOCK_PATCH, return_value=response) as mock_post:
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["sample-0", "sample-1", "sample-2"],
        )
        self.assertEqual(mock_post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
