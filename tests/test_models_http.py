"""Tests for the generic HTTP adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.http import HttpAdapter
from guard_eval_harness.schemas import NormalizedSample


_MOCK_PATCH = "guard_eval_harness.models.http.json_post_with_retry"


def _sample(
    sample_id: str = "sample-1",
    content: str = "Check this",
    unsafe: bool = False,
) -> NormalizedSample:
    """Build a minimal normalized sample for tests."""
    return NormalizedSample(
        id=sample_id,
        dataset="demo",
        split="test",
        messages=[{"role": "user", "content": content}],
        label={"unsafe": unsafe},
    )


class HttpAdapterTest(unittest.TestCase):
    """Validate templated request rendering and scoring."""

    def test_predict_batch_renders_payload_template(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "payload_template": {
                    "sample_id": "{sample_id}",
                    "prompt": "{messages_text}",
                },
                "response_path": "result",
                "score_path": "unsafe_score",
                "headers": {"X-Mode": "test"},
                "api_key": "secret-token",
            },
        )
        adapter = HttpAdapter.from_config(config)
        sample = _sample()

        with patch(
            _MOCK_PATCH,
            return_value={"result": {"unsafe_score": 0.2}},
        ) as mock_post:
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.2)
        self.assertFalse(predictions[0].unsafe_label)

        payload = mock_post.call_args.args[1]
        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(payload["prompt"], "user: Check this")

        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["X-Mode"], "test")
        self.assertEqual(headers["Authorization"], "Bearer secret-token")

    def test_capabilities_are_declared(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={"concurrency": 4},
        )
        adapter = HttpAdapter.from_config(config)

        self.assertTrue(adapter.capabilities.batching)
        self.assertTrue(adapter.capabilities.probability_scores)
        self.assertTrue(adapter.capabilities.concurrency)
        self.assertTrue(adapter.capabilities.supports_category_outputs)

    def test_predict_batch_extracts_judge_categories(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={"url": "https://example.test/score"},
        )
        adapter = HttpAdapter.from_config(config)
        sample = _sample()

        with patch(
            _MOCK_PATCH,
            return_value="#judge: yes\n#type: CWE-79",
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].predicted_categories, ("CWE-79",))

    def test_predict_batch_extracts_categories_from_structured_response(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "score_path": "choices.0.message.content",
            },
        )
        adapter = HttpAdapter.from_config(config)
        sample = _sample()

        with patch(
            _MOCK_PATCH,
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "#judge: yes\n#type: CWE-79"
                        }
                    }
                ]
            },
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].unsafe_score, 1.0)
        self.assertEqual(predictions[0].predicted_categories, ("CWE-79",))

    def test_request_payload_omits_model_when_unset_for_messages(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={"url": "https://example.test/score"},
        )
        adapter = HttpAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {
                    "role": "user",
                    "content": "Check this",
                    "metadata": {"trace_id": "abc123"},
                }
            ],
            label={"unsafe": False},
        )

        payload = adapter._request_payload(sample)

        self.assertNotIn("model", payload)
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "Check this"}],
        )

    def test_request_payload_omits_model_when_unset_for_prompt(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "input_mode": "prompt",
            },
        )
        adapter = HttpAdapter.from_config(config)
        sample = _sample()

        payload = adapter._request_payload(sample)

        self.assertEqual(payload, {"prompt": "user: Check this"})

    def test_retry_arguments_are_forwarded(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "retries": 3,
                "retry_backoff": 2.5,
            },
        )
        adapter = HttpAdapter.from_config(config)

        with patch(_MOCK_PATCH, return_value={"score": 0.3}) as mock_post:
            adapter.predict_batch([_sample()], threshold=0.5)

        self.assertEqual(mock_post.call_args.kwargs["retries"], 3)
        self.assertEqual(mock_post.call_args.kwargs["backoff"], 2.5)


class HttpConcurrencyTest(unittest.TestCase):
    """Validate concurrent execution within predict_batch."""

    def test_concurrent_execution_preserves_order(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "concurrency": 4,
            },
        )
        adapter = HttpAdapter.from_config(config)
        samples = [_sample(sample_id=f"s-{i}") for i in range(5)]

        with patch(
            _MOCK_PATCH,
            side_effect=lambda *args, **kwargs: {"score": 0.2},
        ) as mock_post:
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(mock_post.call_count, 5)
        self.assertEqual(len(predictions), 5)
        for index, prediction in enumerate(predictions):
            self.assertEqual(prediction.sample_id, f"s-{index}")

    def test_empty_batch_returns_empty(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "concurrency": 4,
            },
        )
        adapter = HttpAdapter.from_config(config)

        predictions = adapter.predict_batch([], threshold=0.5)

        self.assertEqual(predictions, [])


class HttpBatchErrorHandlingTest(unittest.TestCase):
    """Verify failures raise with clear context."""

    def test_batch_raises_on_failure(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "score_path": "unsafe_score",
                "drop_failed_predictions": False,
            },
        )
        adapter = HttpAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        def side_effect(url, payload, **kwargs):
            if payload.get("messages", [{}])[0].get("content") == "s-1":
                raise RuntimeError("API 500")
            return {"unsafe_score": 0.2}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.predict_batch(samples, threshold=0.5)

        self.assertIn("s-1", str(ctx.exception))

    def test_batch_drops_failed_predictions(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "score_path": "unsafe_score",
                "drop_failed_predictions": True,
            },
        )
        adapter = HttpAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        def side_effect(url, payload, **kwargs):
            if payload.get("messages", [{}])[0].get("content") == "s-1":
                raise RuntimeError("API 500")
            return {"unsafe_score": 0.2}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(len(predictions), 2)
        returned_ids = {prediction.sample_id for prediction in predictions}
        self.assertEqual(returned_ids, {"s-0", "s-2"})

    def test_concurrent_batch_drops_failed_predictions(self) -> None:
        config = ResolvedModelConfig(
            adapter="http",
            args={
                "url": "https://example.test/score",
                "score_path": "unsafe_score",
                "drop_failed_predictions": True,
                "concurrency": 4,
            },
        )
        adapter = HttpAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(4)
        ]

        def side_effect(url, payload, **kwargs):
            content = payload.get("messages", [{}])[0].get("content")
            if content in {"s-1", "s-3"}:
                raise RuntimeError("API 500")
            return {"unsafe_score": 0.2}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["s-0", "s-2"],
        )


if __name__ == "__main__":
    unittest.main()
