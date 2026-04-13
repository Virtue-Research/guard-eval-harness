"""Tests for the OpenAI-compatible adapter."""

from __future__ import annotations

import http.client
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.openai_compatible import (
    OpenAICompatibleAdapter,
)
from guard_eval_harness.models.templates import (
    json_post_with_retry,
)
from guard_eval_harness.schemas import NormalizedSample


def _sample(
    sample_id: str = "sample-1",
    content: str = "Check this",
    unsafe: bool = True,
) -> NormalizedSample:
    """Build a minimal normalized sample for tests."""
    return NormalizedSample(
        id=sample_id,
        dataset="demo",
        split="test",
        messages=[{"role": "user", "content": content}],
        label={"unsafe": unsafe},
    )


_MOCK_PATCH = "guard_eval_harness.models.openai_compatible.json_post_with_retry"
_ASYNC_MOCK_PATCH = (
    "guard_eval_harness.models.openai_compatible.async_json_post_with_retry"
)


def _as_async_side_effect(sync_side_effect):
    """Wrap a sync side_effect into an async one matching the async helper."""

    async def _wrapped(client, url, payload, **kwargs):
        return sync_side_effect(url, payload, **kwargs)

    return _wrapped


class OpenAICompatibleAdapterTest(unittest.TestCase):
    """Validate request formatting and response scoring."""

    def test_predict_batch_builds_chat_request(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "prompt_template": "{messages_text}",
                "headers": {"X-Test": "1"},
                "api_key": "token-123",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = _sample()

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.75},
        ) as mock_post:
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.75)
        self.assertTrue(predictions[0].unsafe_label)

        payload = mock_post.call_args.args[1]
        self.assertEqual(payload["model"], "demo-openai")
        self.assertEqual(
            payload["messages"][0]["content"],
            "user: Check this",
        )

        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["X-Test"], "1")
        self.assertEqual(headers["Authorization"], "Bearer token-123")

    def test_predict_batch_omits_message_metadata(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={"url": "https://example.test/v1/chat/completions"},
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
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
            label={"unsafe": True},
        )

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.75},
        ) as mock_post:
            adapter.predict_batch([sample], threshold=0.5)

        payload = mock_post.call_args.args[1]
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "Check this"}],
        )

    def test_capabilities_report_batched_requests(
        self,
    ) -> None:
        config = ResolvedModelConfig(adapter="openai_compatible")
        adapter = OpenAICompatibleAdapter.from_config(config)

        self.assertTrue(adapter.capabilities.batching)
        self.assertTrue(adapter.capabilities.probability_scores)
        self.assertIn("image", adapter.capabilities.supported_input_modalities)
        self.assertIn("code", adapter.capabilities.supported_input_modalities)
        self.assertTrue(adapter.capabilities.supports_category_outputs)

    def test_predict_batch_builds_multimodal_chat_request(self) -> None:
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
                adapter="openai_compatible",
                model_name="demo-openai",
                args={
                    "url": "https://example.test/v1/chat/completions",
                },
            )
            adapter = OpenAICompatibleAdapter.from_config(config)

            with patch(
                _MOCK_PATCH,
                return_value={"score": 0.75},
            ) as mock_post:
                adapter.predict_batch([sample], threshold=0.5)

        payload = mock_post.call_args.args[1]
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "Check this image")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_multimodal_chat_request_sniffs_extensionless_png(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample"
            Image.new("RGB", (16, 16), color="red").save(
                image_path,
                format="PNG",
            )
            sample = NormalizedSample(
                id="sample-1",
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
                            }
                        ],
                    }
                ],
                label={"unsafe": True},
            )
            config = ResolvedModelConfig(
                adapter="openai_compatible",
                model_name="demo-openai",
                args={"url": "https://example.test/v1/chat/completions"},
            )
            adapter = OpenAICompatibleAdapter.from_config(config)

            with patch(
                _MOCK_PATCH,
                return_value={"score": 0.75},
            ) as mock_post:
                adapter.predict_batch([sample], threshold=0.5)

        payload = mock_post.call_args.args[1]
        content = payload["messages"][0]["content"]
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_prompt_template_rejects_multimodal_samples_by_default(
        self,
    ) -> None:
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
                                "uri": "https://example.test/image.png",
                            },
                        },
                    ],
                }
            ],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "prompt_template": "{messages_text}",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with self.assertRaisesRegex(ValueError, "drop image content"):
            adapter._request_payload(sample)

    def test_prompt_template_allows_explicit_text_only_mode(self) -> None:
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
                                "uri": "https://example.test/image.png",
                            },
                        },
                    ],
                }
            ],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "prompt_template": "{messages_text}",
                "prompt_template_multimodal_mode": "text_only",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        payload = adapter._request_payload(sample)

        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "user: Check this image"}],
        )


class SystemPromptTest(unittest.TestCase):
    """Validate system_prompt injection in chat mode."""

    def test_system_prompt_prepended_to_messages(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "system_prompt": "You are a safety classifier.",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = _sample()

        payload = adapter._request_payload(sample)

        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(
            payload["messages"][0]["content"],
            "You are a safety classifier.",
        )
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_system_prompt_ignored_in_completions_mode(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/completions",
                "mode": "completions",
                "system_prompt": "You are a safety classifier.",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = _sample()

        payload = adapter._request_payload(sample)

        self.assertIn("prompt", payload)
        self.assertNotIn("messages", payload)

    def test_system_prompt_preserved_with_prompt_template(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "system_prompt": "You are a safety classifier.",
                "prompt_template": "{messages_text}",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = _sample()

        payload = adapter._request_payload(sample)

        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(
            payload["messages"][0]["content"],
            "You are a safety classifier.",
        )
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(
            payload["messages"][1]["content"],
            "user: Check this",
        )

    def test_prompt_template_drops_sample_system_message_without_model_system_prompt(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="demo-openai",
            args={
                "url": "https://example.test/v1/chat/completions",
                "prompt_template": "{messages_text}",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "system", "content": "dataset system"},
                {"role": "user", "content": "Check this"},
            ],
            label={"unsafe": True},
        )

        payload = adapter._request_payload(sample)

        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": "system: dataset system\nuser: Check this",
                }
            ],
        )


class ConcurrencyTest(unittest.TestCase):
    """Validate concurrent execution within predict_batch."""

    def test_sequential_when_concurrency_is_one(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 1,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [_sample(sample_id=f"s-{i}") for i in range(3)]

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.1},
        ) as mock_post:
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(len(predictions), 3)
        self.assertFalse(adapter.capabilities.concurrency)

    def test_concurrent_execution_preserves_order(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 4,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [_sample(sample_id=f"s-{i}") for i in range(5)]

        def side_effect(url, payload, **kwargs):
            return {"score": 0.2}

        with patch(
            _ASYNC_MOCK_PATCH,
            side_effect=_as_async_side_effect(side_effect),
        ) as mock_post:
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(mock_post.call_count, 5)
        self.assertEqual(len(predictions), 5)
        for i, pred in enumerate(predictions):
            self.assertEqual(pred.sample_id, f"s-{i}")
        self.assertTrue(adapter.capabilities.concurrency)

    def test_empty_batch_returns_empty(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 4,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        predictions = adapter.predict_batch([], threshold=0.5)

        self.assertEqual(predictions, [])

    def test_concurrency_capped_at_max(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 999,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        self.assertTrue(adapter.capabilities.concurrency)


class TokenTrackingTest(unittest.TestCase):
    """Validate token-usage metadata extraction."""

    def test_usage_attached_to_prediction_metadata(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={"url": "https://example.test/v1/chat/completions"},
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        sample = _sample()

        response = {
            "choices": [{"message": {"content": "unsafe"}}],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 1,
                "total_tokens": 43,
            },
        }

        with patch(_MOCK_PATCH, return_value=response):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        usage = predictions[0].metadata.get("usage", {})
        self.assertEqual(usage["prompt_tokens"], 42)
        self.assertEqual(usage["completion_tokens"], 1)
        self.assertEqual(usage["total_tokens"], 43)

    def test_missing_usage_omitted_from_metadata(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={"url": "https://example.test/v1/chat/completions"},
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.3},
        ):
            predictions = adapter.predict_batch([_sample()], threshold=0.5)

        self.assertNotIn("usage", predictions[0].metadata)

    def test_capabilities_report_token_accounting(
        self,
    ) -> None:
        config = ResolvedModelConfig(adapter="openai_compatible")
        adapter = OpenAICompatibleAdapter.from_config(config)
        self.assertTrue(adapter.capabilities.token_accounting)


class RetryWiringTest(unittest.TestCase):
    """Validate that retry args are forwarded correctly."""

    def test_retry_args_forwarded_to_json_post_with_retry(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "retries": 3,
                "retry_backoff": 2.5,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.4},
        ) as mock_post:
            adapter.predict_batch([_sample()], threshold=0.5)

        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["retries"], 3)
        self.assertAlmostEqual(kwargs["backoff"], 2.5)

    def test_default_retry_is_zero(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={"url": "https://example.test/v1/chat/completions"},
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.5},
        ) as mock_post:
            adapter.predict_batch([_sample(unsafe=False)], threshold=0.6)

        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["retries"], 0)


class JsonPostWithRetryTest(unittest.TestCase):
    """Validate the retry helper in templates.py."""

    @patch("guard_eval_harness.models.templates.json_post")
    def test_succeeds_without_retry(self, mock_post) -> None:
        mock_post.return_value = {"ok": True}
        result = json_post_with_retry(
            "https://example.test",
            {"key": "val"},
            retries=0,
        )
        self.assertEqual(result, {"ok": True})
        mock_post.assert_called_once()

    @patch("guard_eval_harness.models.templates.time.sleep")
    @patch("guard_eval_harness.models.templates.json_post")
    def test_retries_on_429(self, mock_post, mock_sleep) -> None:
        headers = http.client.HTTPMessage()
        error_429 = urllib_error.HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(b""),
        )
        mock_post.side_effect = [
            error_429,
            {"ok": True},
        ]
        result = json_post_with_retry(
            "https://example.test",
            {},
            retries=1,
            backoff=0.1,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("guard_eval_harness.models.templates.time.sleep")
    @patch("guard_eval_harness.models.templates.json_post")
    def test_retries_on_500(self, mock_post, mock_sleep) -> None:
        headers = http.client.HTTPMessage()
        error_500 = urllib_error.HTTPError(
            "https://example.test",
            500,
            "Internal Server Error",
            headers,
            io.BytesIO(b""),
        )
        mock_post.side_effect = [
            error_500,
            error_500,
            {"ok": True},
        ]
        result = json_post_with_retry(
            "https://example.test",
            {},
            retries=2,
            backoff=0.1,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_post.call_count, 3)

    @patch("guard_eval_harness.models.templates.json_post")
    def test_raises_on_non_retryable_error(self, mock_post) -> None:
        headers = http.client.HTTPMessage()
        error_400 = urllib_error.HTTPError(
            "https://example.test",
            400,
            "Bad Request",
            headers,
            io.BytesIO(b""),
        )
        mock_post.side_effect = error_400
        with self.assertRaises(urllib_error.HTTPError) as ctx:
            json_post_with_retry(
                "https://example.test",
                {},
                retries=3,
            )
        self.assertEqual(ctx.exception.code, 400)
        mock_post.assert_called_once()

    @patch("guard_eval_harness.models.templates.time.sleep")
    @patch("guard_eval_harness.models.templates.json_post")
    def test_exhausted_retries_raises(self, mock_post, mock_sleep) -> None:
        headers = http.client.HTTPMessage()
        error_502 = urllib_error.HTTPError(
            "https://example.test",
            502,
            "Bad Gateway",
            headers,
            io.BytesIO(b""),
        )
        mock_post.side_effect = error_502
        with self.assertRaises(urllib_error.HTTPError):
            json_post_with_retry(
                "https://example.test",
                {},
                retries=2,
                backoff=0.01,
            )
        self.assertEqual(mock_post.call_count, 3)

    @patch("guard_eval_harness.models.templates.time.sleep")
    @patch("guard_eval_harness.models.templates.json_post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep) -> None:
        mock_post.side_effect = [
            ConnectionError("refused"),
            {"ok": True},
        ]
        result = json_post_with_retry(
            "https://example.test",
            {},
            retries=1,
            backoff=0.1,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_post.call_count, 2)

    @patch("guard_eval_harness.models.templates.time.sleep")
    @patch("guard_eval_harness.models.templates.json_post")
    def test_respects_retry_after_header(self, mock_post, mock_sleep) -> None:
        headers = http.client.HTTPMessage()
        headers["Retry-After"] = "5"
        error_429 = urllib_error.HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(b""),
        )
        mock_post.side_effect = [
            error_429,
            {"ok": True},
        ]
        json_post_with_retry(
            "https://example.test",
            {},
            retries=1,
            backoff=0.1,
        )
        mock_sleep.assert_called_once_with(5.0)


class EndpointResolutionTest(unittest.TestCase):
    """Validate OpenRouter and OpenAI endpoint construction."""

    def test_openrouter_root_url(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="openai/gpt-5-mini",
            args={
                "root_url": "https://openrouter.ai/api",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.0},
        ) as mock_post:
            adapter.predict_batch([_sample(unsafe=False)], threshold=0.5)

        called_url = mock_post.call_args.args[0]
        self.assertEqual(
            called_url,
            "https://openrouter.ai/api/v1/chat/completions",
        )

    def test_completions_mode_endpoint(self) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={"mode": "completions"},
        )
        adapter = OpenAICompatibleAdapter.from_config(config)

        with patch(
            _MOCK_PATCH,
            return_value={"score": 0.0},
        ) as mock_post:
            adapter.predict_batch([_sample(unsafe=False)], threshold=0.5)

        called_url = mock_post.call_args.args[0]
        self.assertIn("/v1/completions", called_url)


class BatchErrorHandlingTest(unittest.TestCase):
    """Verify failures raise with clear context."""

    def test_concurrent_batch_drops_failed_predictions_by_default(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 4,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(5)
        ]

        def side_effect(url, payload, **kwargs):
            if payload["messages"][0]["content"] == "s-2":
                raise RuntimeError("API 500")
            return {"score": 0.3}

        with patch(
            _ASYNC_MOCK_PATCH,
            side_effect=_as_async_side_effect(side_effect),
        ):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["s-0", "s-1", "s-3", "s-4"],
        )

    def test_sequential_batch_raises_on_failure(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 1,
                "drop_failed_predictions": False,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        def side_effect(url, payload, **kwargs):
            if payload["messages"][0]["content"] == "s-1":
                raise RuntimeError("API 500")
            return {"score": 0.3}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.predict_batch(samples, threshold=0.5)

        self.assertIn("s-1", str(ctx.exception))

    def test_sequential_batch_drops_failed_predictions_by_default(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 1,
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        def side_effect(url, payload, **kwargs):
            if payload["messages"][0]["content"] == "s-1":
                raise RuntimeError("API 500")
            return {"score": 0.3}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["s-0", "s-2"],
        )

    def test_string_false_disables_dropping_failed_predictions(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="openai_compatible",
            model_name="m",
            args={
                "url": "https://example.test/v1/chat/completions",
                "concurrency": 1,
                "drop_failed_predictions": "false",
            },
        )
        adapter = OpenAICompatibleAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        def side_effect(url, payload, **kwargs):
            if payload["messages"][0]["content"] == "s-1":
                raise RuntimeError("API 500")
            return {"score": 0.3}

        with patch(_MOCK_PATCH, side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.predict_batch(samples, threshold=0.5)

        self.assertIn("s-1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
