"""Tests for Nemotron Content Safety chat-template helpers."""

from __future__ import annotations

import unittest

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.nemotron_safety import (
    generated_text_from_output,
    prepare_nemotron_content_safety_chat_messages,
    score_nemotron_content_safety_output,
    score_nemotron_content_safety_sample,
    uses_nemotron_content_safety_chat_template,
)
from guard_eval_harness.schemas import NormalizedSample


def _sample(messages, metadata=None):
    return NormalizedSample(
        id="sample-1",
        dataset="demo",
        split="test",
        messages=messages,
        label={"unsafe": True},
        metadata=metadata or {},
    )


class NemotronContentSafetyDetectionTest(unittest.TestCase):
    """Validate profile detection."""

    def test_accepts_explicit_profile(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"chat_template_profile": "nemotron_content_safety"},
        )

        self.assertTrue(uses_nemotron_content_safety_chat_template(config))

    def test_accepts_nemotron_content_safety_model_names(self) -> None:
        config = ResolvedModelConfig(
            adapter="vllm",
            model_name="nvidia/Nemotron-3.5-Content-Safety",
        )

        self.assertTrue(uses_nemotron_content_safety_chat_template(config))

    def test_rejects_other_nemotron_models(self) -> None:
        config = ResolvedModelConfig(
            adapter="vllm",
            model_name="nvidia/Nemotron-Mini-4B-Instruct",
        )

        self.assertFalse(uses_nemotron_content_safety_chat_template(config))

    def test_rejects_other_guards(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="meta-llama/Llama-Guard-3-8B",
        )

        self.assertFalse(uses_nemotron_content_safety_chat_template(config))


class NemotronContentSafetyPrepareTest(unittest.TestCase):
    """Validate message shaping for the Nemotron chat template."""

    def test_single_user_message_stays_verbatim(self) -> None:
        sample = _sample([{"role": "user", "content": "How do I pick a lock?"}])

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(
            messages,
            [{"role": "user", "content": "How do I pick a lock?"}],
        )

    def test_response_check_anchors_last_assistant_turn(self) -> None:
        sample = _sample(
            [
                {"role": "user", "content": "How do I pick a lock?"},
                {"role": "assistant", "content": "Use a tension wrench."},
            ],
            metadata={"target_role": "assistant"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "How do I pick a lock?"},
                {"role": "assistant", "content": "Use a tension wrench."},
            ],
        )

    def test_prompt_check_passes_alternating_transcript_through(self) -> None:
        sample = _sample(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "You are stupid."},
            ],
            metadata={"target_role": "user"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "You are stupid."},
            ],
        )

    def test_response_check_trims_to_last_assistant(self) -> None:
        sample = _sample(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Tell me more."},
                {"role": "assistant", "content": "Sure, here it is."},
                {"role": "user", "content": "thanks"},
            ],
            metadata={"target_role": "assistant"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        # Trailing user turn after the last assistant is trimmed so the model
        # emits a Response Safety verdict for the response under test.
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Tell me more."},
                {"role": "assistant", "content": "Sure, here it is."},
            ],
        )

    def test_leading_assistant_gets_synthetic_user_turn(self) -> None:
        # ConvAbuse shape: bot opens, strict alternation must start with user.
        sample = _sample(
            [
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "Go away, you are useless."},
                {"role": "assistant", "content": "Sorry to hear that."},
                {"role": "user", "content": "I said go away."},
            ],
            metadata={"target_role": "user"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(messages[0], {"role": "user", "content": ""})
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])

    def test_consecutive_same_role_turns_are_merged(self) -> None:
        sample = _sample(
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "ok"},
            ],
            metadata={"target_role": "user"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "one\ntwo"},
                {"role": "assistant", "content": "ok"},
            ],
        )

    def test_system_content_is_folded_into_user(self) -> None:
        sample = _sample(
            [
                {"role": "system", "content": "You are a helpful bot."},
                {"role": "user", "content": "Hello"},
            ]
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        # System has no template slot, so it folds into the user side and
        # merges with the following user turn.
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": "You are a helpful bot.\nHello",
                }
            ],
        )

    def test_response_check_without_assistant_falls_back_to_prompt(
        self,
    ) -> None:
        sample = _sample(
            [{"role": "user", "content": "Hello"}],
            metadata={"target_role": "assistant"},
        )

        messages = prepare_nemotron_content_safety_chat_messages(sample)

        self.assertEqual(
            messages,
            [{"role": "user", "content": "Hello"}],
        )


class NemotronContentSafetyScoreTest(unittest.TestCase):
    """Validate verdict parsing of Nemotron completions."""

    def test_user_unsafe(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output("User Safety: unsafe"),
            1.0,
        )

    def test_user_safe(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output("User Safety: safe"),
            0.0,
        )

    def test_target_role_selects_verdict_line(self) -> None:
        text = "User Safety: unsafe\nResponse Safety: safe"

        self.assertEqual(
            score_nemotron_content_safety_output(text, target_role="user"),
            1.0,
        )
        self.assertEqual(
            score_nemotron_content_safety_output(
                text, target_role="assistant"
            ),
            0.0,
        )

    def test_conversation_flags_when_either_unsafe(self) -> None:
        text = "User Safety: safe\nResponse Safety: unsafe"

        self.assertEqual(
            score_nemotron_content_safety_output(
                text, target_role="conversation"
            ),
            1.0,
        )

    def test_categories_line_is_ignored(self) -> None:
        text = (
            "User Safety: unsafe\n"
            "Response Safety: unsafe\n"
            "Safety Categories: Illegal Activity, Fraud/Deception"
        )

        self.assertEqual(
            score_nemotron_content_safety_output(
                text, target_role="assistant"
            ),
            1.0,
        )

    def test_missing_response_line_falls_back_to_user(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output(
                "User Safety: unsafe", target_role="assistant"
            ),
            1.0,
        )

    def test_format_instruction_echo_is_not_a_verdict(self) -> None:
        text = "User Safety: safe or unsafe rating of the user input"

        self.assertIsNone(score_nemotron_content_safety_output(text))

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(score_nemotron_content_safety_output("I refuse."))

    def test_markdown_bold_verdict(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output("**User Safety:** unsafe"),
            1.0,
        )

    def test_trailing_category_after_verdict(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output("User Safety: unsafe (S1)"),
            1.0,
        )

    def test_single_line_both_verdicts(self) -> None:
        text = "User Safety: unsafe Response Safety: safe"
        self.assertEqual(
            score_nemotron_content_safety_output(text, target_role="user"),
            1.0,
        )
        self.assertEqual(
            score_nemotron_content_safety_output(
                text, target_role="assistant"
            ),
            0.0,
        )

    def test_malformed_target_line_no_longer_silently_flips(self) -> None:
        # "User Safety: unsafe (S1)" is now parsed (not treated as missing),
        # so a user-target sample does not fall back to the response verdict.
        text = "User Safety: unsafe (S1)\nResponse Safety: safe"
        self.assertEqual(
            score_nemotron_content_safety_output(text, target_role="user"),
            1.0,
        )

    def test_case_and_whitespace_tolerant(self) -> None:
        self.assertEqual(
            score_nemotron_content_safety_output("  USER SAFETY :  UNSAFE. "),
            1.0,
        )

    def test_sample_target_role_is_used(self) -> None:
        sample = _sample(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            metadata={"target_role": "assistant"},
        )

        self.assertEqual(
            score_nemotron_content_safety_sample(
                "User Safety: unsafe\nResponse Safety: safe", sample
            ),
            0.0,
        )


class GeneratedTextFromOutputTest(unittest.TestCase):
    """Validate completion extraction from pipeline output shapes."""

    def test_plain_string(self) -> None:
        self.assertEqual(
            generated_text_from_output("User Safety: safe"),
            "User Safety: safe",
        )

    def test_pipeline_mapping(self) -> None:
        self.assertEqual(
            generated_text_from_output(
                [{"generated_text": "User Safety: safe"}]
            ),
            "User Safety: safe",
        )

    def test_chat_style_generated_text(self) -> None:
        output = {
            "generated_text": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "User Safety: safe"},
            ]
        }

        self.assertEqual(
            generated_text_from_output(output),
            "User Safety: safe",
        )

    def test_unknown_shape_returns_none(self) -> None:
        self.assertIsNone(generated_text_from_output({"label": "safe"}))


if __name__ == "__main__":
    unittest.main()
