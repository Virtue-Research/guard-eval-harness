"""Tests for the Anthropic adapter."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch
from urllib import error as urllib_error

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.anthropic import AnthropicAdapter
from guard_eval_harness.schemas import NormalizedSample

_MOCK_PATCH = "guard_eval_harness.models.anthropic.json_post_with_retry"
_ASYNC_MOCK_PATCH = (
    "guard_eval_harness.models.anthropic.async_json_post_with_retry"
)


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


def _http_error(status: int, reason: str = "Unauthorized"):
    return urllib_error.HTTPError(
        "https://api.anthropic.com/v1/messages",
        status,
        reason,
        {},
        io.BytesIO(b"{}"),
    )


class AnthropicAdapterAPIKeyTest(unittest.TestCase):
    """Validate early API key validation."""

    def test_missing_api_key_raises(self) -> None:
        """predict_batch raises ValueError when no key."""
        config = ResolvedModelConfig(
            adapter="anthropic",
            model_name="claude-3-haiku-20240307",
            args={},
        )
        adapter = AnthropicAdapter.from_config(config)

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                adapter.predict_batch(
                    [_sample()], threshold=0.5
                )
            self.assertIn(
                "API key is missing", str(ctx.exception)
            )


    def test_custom_header_casing_accepted(self) -> None:
        """API key via headers with non-lowercase casing."""
        config = ResolvedModelConfig(
            adapter="anthropic",
            model_name="claude-3-haiku-20240307",
            args={
                "headers": {"X-API-Key": "sk-test"},
            },
        )
        adapter = AnthropicAdapter.from_config(config)

        with patch.dict("os.environ", {}, clear=True):
            headers = adapter._headers()
        # Key is present (case-insensitive) — validation
        # must not reject it.
        has_key = any(
            k.lower() == "x-api-key" and v
            for k, v in headers.items()
        )
        self.assertTrue(has_key)

    def test_concurrent_auth_error_fails_before_async_dispatch(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="anthropic",
            model_name="claude-3-haiku-20240307",
            args={
                "api_key": "bad-key",
                "concurrency": 4,
            },
        )
        adapter = AnthropicAdapter.from_config(config)
        samples = [
            _sample(sample_id=f"s-{i}", content=f"s-{i}") for i in range(3)
        ]

        with patch(_MOCK_PATCH, side_effect=_http_error(401)) as mock_post:
            with patch(_ASYNC_MOCK_PATCH) as mock_async_post:
                with self.assertRaises(ValueError) as ctx:
                    adapter.predict_batch(samples, threshold=0.5)

        self.assertIn("authentication failed", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)
        mock_async_post.assert_not_called()


class AnthropicAdapterCacheSystemPromptTest(unittest.TestCase):
    """Caching the system prompt must not alter what the model sees.

    Why: prompt caching is documented as a server-side optimization;
    it must not change the prompt text, model, messages, or sampling
    parameters reaching the model.
    """

    def _adapter(self, cache: bool) -> AnthropicAdapter:
        config = ResolvedModelConfig(
            adapter="anthropic",
            model_name="claude-haiku-4-5-20251001",
            args={
                "headers": {"x-api-key": "sk-test"},
                "system_prompt": "You are a strict safety judge.",
                "max_tokens": 256,
                "temperature": 0,
                "cache_system_prompt": cache,
            },
        )
        return AnthropicAdapter.from_config(config)

    def test_cache_only_adds_cache_control(self) -> None:
        sample = _sample(content="Is this safe? hello")
        off = self._adapter(False)._request_payload(sample)
        on = self._adapter(True)._request_payload(sample)

        # Everything except `system` is byte-identical.
        self.assertEqual(
            {k: v for k, v in off.items() if k != "system"},
            {k: v for k, v in on.items() if k != "system"},
        )

        # cache=off: system is a plain string.
        self.assertIsInstance(off["system"], str)

        # cache=on: system is a single text block with the same
        # text plus an ephemeral cache_control hint.
        self.assertEqual(len(on["system"]), 1)
        block = on["system"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], off["system"])
        self.assertEqual(
            block["cache_control"], {"type": "ephemeral"}
        )
        # No other fields snuck in.
        self.assertEqual(
            set(block.keys()), {"type", "text", "cache_control"}
        )

    def test_cache_off_payload_unchanged_from_pre_change(self) -> None:
        """When the flag is unset, payload is the legacy string form.

        Why: existing configs (the vast majority) leave the flag unset
        and must keep their exact prior behaviour.
        """
        config = ResolvedModelConfig(
            adapter="anthropic",
            model_name="claude-haiku-4-5-20251001",
            args={
                "headers": {"x-api-key": "sk-test"},
                "system_prompt": "You are a strict safety judge.",
                "max_tokens": 256,
            },
        )
        adapter = AnthropicAdapter.from_config(config)
        payload = adapter._request_payload(_sample())
        self.assertIsInstance(payload["system"], str)
        self.assertEqual(
            payload["system"], "You are a strict safety judge."
        )


_PNG_PIXEL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


def _media_sample() -> NormalizedSample:
    """A sample carrying an inline image part."""
    return NormalizedSample(
        id="sample-img",
        dataset="demo",
        split="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "classify"},
                    {"type": "image_url", "image_url": {"url": _PNG_PIXEL}},
                ],
            }
        ],
        label={"unsafe": False},
    )


def _template_adapter(**args) -> AnthropicAdapter:
    config = ResolvedModelConfig(
        adapter="anthropic",
        model_name="claude-haiku-4-5-20251001",
        args={"api_key": "sk-test", **args},
    )
    return AnthropicAdapter.from_config(config)


class AnthropicAdapterPromptTemplateTest(unittest.TestCase):
    """Cover prompt_template and user_prompt_template handling.

    These two knobs collapse the conversation to a single user turn in
    different ways; both must keep working and the multimodal guards
    must fire so image content is never silently dropped.
    """

    def test_prompt_template_renders_single_user_turn(self) -> None:
        adapter = _template_adapter(prompt_template="JUDGE: {messages_text}")
        payload = adapter._request_payload(_sample(content="is this safe?"))
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertIn("JUDGE:", str(payload["messages"][0]["content"]))

    def test_prompt_template_rejects_media_by_default(self) -> None:
        adapter = _template_adapter(prompt_template="JUDGE: {messages_text}")
        with self.assertRaisesRegex(ValueError, "drop image content"):
            adapter._request_payload(_media_sample())

    def test_prompt_template_text_only_mode_allows_media(self) -> None:
        adapter = _template_adapter(
            prompt_template="JUDGE: {messages_text}",
            prompt_template_multimodal_mode="text_only",
        )
        payload = adapter._request_payload(_media_sample())
        self.assertEqual(len(payload["messages"]), 1)

    def test_invalid_multimodal_mode_raises(self) -> None:
        adapter = _template_adapter(
            prompt_template="JUDGE: {messages_text}",
            prompt_template_multimodal_mode="bogus",
        )
        with self.assertRaisesRegex(ValueError, "error.*text_only"):
            adapter._request_payload(_media_sample())

    def test_user_prompt_template_renders_single_user_turn(self) -> None:
        adapter = _template_adapter(
            user_prompt_template="UPT: {messages_text}",
            system_prompt="SYS",
        )
        payload = adapter._request_payload(_sample(content="is this safe?"))
        self.assertEqual(len(payload["messages"]), 1)
        self.assertIn("UPT:", str(payload["messages"][0]["content"]))
        self.assertEqual(payload["system"], "SYS")

    def test_user_prompt_template_rejects_media(self) -> None:
        adapter = _template_adapter(user_prompt_template="UPT: {messages_text}")
        with self.assertRaisesRegex(ValueError, "drop image/media"):
            adapter._request_payload(_media_sample())

    def test_prompt_template_takes_precedence_over_user_prompt_template(
        self,
    ) -> None:
        adapter = _template_adapter(
            prompt_template="JUDGE: {messages_text}",
            user_prompt_template="UPT: {messages_text}",
        )
        payload = adapter._request_payload(_sample())
        self.assertIn("JUDGE:", str(payload["messages"][0]["content"]))

    def test_generated_text_recorded_in_metadata(self) -> None:
        adapter = _template_adapter(prompt_template="{messages_text}")
        response = {
            "content": [{"type": "text", "text": "unsafe"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        with patch(_MOCK_PATCH, return_value=response):
            predictions = adapter.predict_batch([_sample()], threshold=0.5)
        self.assertEqual(
            predictions[0].metadata["generated_text"], "unsafe"
        )


if __name__ == "__main__":
    unittest.main()
