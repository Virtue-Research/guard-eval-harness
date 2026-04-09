"""Tests for Llama Guard chat-template helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.llama_guard import (
    prepare_llama_guard_chat_messages,
    uses_llama_guard_chat_template,
)
from guard_eval_harness.schemas import NormalizedSample


class LlamaGuardChatTemplateTest(unittest.TestCase):
    """Validate Llama Guard chat shaping logic."""

    def test_profile_detection_accepts_explicit_profile(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"chat_template_profile": "llama_guard"},
        )

        self.assertTrue(uses_llama_guard_chat_template(config))

    def test_profile_detection_accepts_llama_guard_model_names(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="meta-llama/Llama-Guard-3-1B",
        )

        self.assertTrue(uses_llama_guard_chat_template(config))

    def test_prepare_chat_messages_wraps_content_for_block_templates(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(
            chat_template="selectattr('type', 'equalto', 'text')"
        )

        messages = prepare_llama_guard_chat_messages(
            sample,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Check this"}],
                }
            ],
        )

    def test_prepare_chat_messages_keeps_string_content_for_string_templates(
        self,
    ) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(chat_template="{{ message['content'] }}")

        messages = prepare_llama_guard_chat_messages(
            sample,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            messages,
            [{"role": "user", "content": "Check this"}],
        )

    def test_prepare_chat_messages_normalizes_assistant_first_conversations(
        self,
    ) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "assistant", "content": "Previous agent turn"},
                {"role": "user", "content": "Previous user turn"},
                {"role": "assistant", "content": "Current agent turn"},
                {"role": "user", "content": "Current user turn"},
            ],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(
            chat_template=(
                "Conversation roles must alternate "
                "user/assistant/user/assistant/..."
            )
        )

        messages = prepare_llama_guard_chat_messages(
            sample,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Previous agent turn"},
                {"role": "user", "content": "Previous user turn"},
                {"role": "assistant", "content": "Current agent turn"},
                {"role": "user", "content": "Current user turn"},
            ],
        )

    def test_prepare_chat_messages_preserves_adjacent_same_role_turns(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "user", "content": "First user turn"},
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Second user turn"},
            ],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(
            chat_template=(
                "Conversation roles must alternate "
                "user/assistant/user/assistant/..."
            )
        )

        messages = prepare_llama_guard_chat_messages(
            sample,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "First user turn"},
                {"role": "assistant", "content": ""},
                {
                    "role": "user",
                    "content": "Second user turn",
                },
            ],
        )

    def test_prepare_chat_messages_skips_unsupported_roles(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Actual user turn"},
            ],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(
            chat_template=(
                "Conversation roles must alternate "
                "user/assistant/user/assistant/..."
            )
        )

        messages = prepare_llama_guard_chat_messages(
            sample,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            messages,
            [{"role": "user", "content": "Actual user turn"}],
        )


if __name__ == "__main__":
    unittest.main()
