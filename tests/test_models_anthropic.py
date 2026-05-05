"""Tests for the Anthropic adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.anthropic import AnthropicAdapter
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


if __name__ == "__main__":
    unittest.main()
