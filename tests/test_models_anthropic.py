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


if __name__ == "__main__":
    unittest.main()
