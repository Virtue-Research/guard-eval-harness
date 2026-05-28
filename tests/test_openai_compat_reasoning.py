"""Reasoning-model knobs on openai_compat backend (token_param + reasoning_effort)."""

import os
import unittest
from unittest import mock

from guard_eval_harness.backends.base import BackendConfig
from guard_eval_harness.backends.openai_compat import (
    OpenAICompatibleBackend,
)


def _config(**args) -> BackendConfig:
    base = {
        "base_url": "https://example.invalid/v1",
        "api_key_env": "_TEST_KEY",
    }
    base.update(args)
    return BackendConfig(kind="openai_compat", model="m", args=base)


class OpenAICompatReasoningTest(unittest.TestCase):
    """The new token_param + reasoning_effort + omit_temperature flags."""

    def setUp(self) -> None:
        os.environ["_TEST_KEY"] = "sk-test"
        self.addCleanup(os.environ.pop, "_TEST_KEY", None)

    def test_default_uses_max_tokens_and_temperature(self) -> None:
        be = OpenAICompatibleBackend(_config())
        # Capture the payload via mocked _post_chat
        captured: dict = {}
        be._post_chat = lambda payload: (
            captured.update(payload)
            or {"choices": [{"message": {"content": "safe"}}]}
        )
        from guard_eval_harness.schemas import Message
        out = be.generate([[Message(role="user", content="hi")]])
        self.assertEqual(out, ["safe"])
        self.assertIn("max_tokens", captured)
        self.assertNotIn("max_completion_tokens", captured)
        self.assertIn("temperature", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_max_completion_tokens_param(self) -> None:
        be = OpenAICompatibleBackend(
            _config(token_param="max_completion_tokens",
                    max_new_tokens=42,
                    reasoning_effort="high",
                    omit_temperature=True)
        )
        captured: dict = {}
        be._post_chat = lambda payload: (
            captured.update(payload)
            or {"choices": [{"message": {"content": "safe"}}]}
        )
        from guard_eval_harness.schemas import Message
        be.generate([[Message(role="user", content="hi")]])
        self.assertEqual(captured.get("max_completion_tokens"), 42)
        self.assertNotIn("max_tokens", captured)
        self.assertEqual(captured.get("reasoning_effort"), "high")
        self.assertNotIn("temperature", captured)

    def test_invalid_token_param_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleBackend(_config(token_param="garbage"))


if __name__ == "__main__":
    unittest.main()
