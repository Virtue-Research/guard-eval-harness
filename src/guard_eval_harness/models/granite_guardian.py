"""Helpers for Granite Guardian chat-template integration."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.schemas import NormalizedSample

GRANITE_GUARDIAN_CHAT_TEMPLATE_PROFILE = "granite_guardian"


def uses_granite_guardian_chat_template(
    config: ResolvedModelConfig,
) -> bool:
    """Return whether a config should use Granite Guardian chat shaping."""
    profile = config.args.get("chat_template_profile")
    if isinstance(profile, str):
        if profile.strip().lower() == GRANITE_GUARDIAN_CHAT_TEMPLATE_PROFILE:
            return True

    model_name = config.model_name or config.args.get("pretrained")
    if not isinstance(model_name, str):
        return False
    normalized = model_name.strip().lower()
    return "granite-guardian" in normalized


def prepare_granite_guardian_chat_messages(
    sample: NormalizedSample,
) -> list[dict[str, Any]]:
    """Shape normalized messages for the Granite Guardian chat template.

    The Granite Guardian Jinja2 template inspects only the last two
    messages and requires one of these role combinations:

    * single ``user`` message  (prompt-level harm check)
    * ``user`` then ``assistant``  (response-level harm check)

    Prompt checks collapse to the last user message. Response checks
    anchor on the last assistant message and only reuse the immediately
    preceding user turn when that exact pair occurred in the transcript.
    """
    # Determine the target role from metadata when available.
    target_role = "user"
    raw_target = sample.metadata.get("target_role")
    if isinstance(raw_target, str) and raw_target.strip().lower() in {
        "assistant",
        "response",
        "conversation",
    }:
        target_role = "assistant"

    # If the dataset targets the assistant response, anchor on the last
    # assistant turn in the original transcript order. Reuse the previous
    # user turn only when it is directly adjacent.
    if target_role == "assistant":
        for index in range(len(sample.messages) - 1, -1, -1):
            assistant_message = sample.messages[index]
            if assistant_message.role != "assistant":
                continue
            if index > 0 and sample.messages[index - 1].role == "user":
                return [
                    {
                        "role": "user",
                        "content": sample.messages[index - 1].text_content,
                    },
                    {"role": "assistant", "content": assistant_message.text_content},
                ]
            return [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": assistant_message.text_content},
            ]

    # Default: evaluate the last user message as a prompt-level check.
    for message in reversed(sample.messages):
        if message.role == "user":
            return [{"role": "user", "content": message.text_content}]

    # Fallback: only assistant messages exist — wrap the last one with a
    # synthetic user turn so the template's response-check path is used.
    for message in reversed(sample.messages):
        if message.role == "assistant":
            return [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": message.text_content},
            ]

    return [{"role": "user", "content": ""}]
