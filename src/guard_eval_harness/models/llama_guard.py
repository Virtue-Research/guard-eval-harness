"""Helpers for Llama Guard chat-template integration."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.schemas import NormalizedSample

LLAMA_GUARD_CHAT_TEMPLATE_PROFILE = "llama_guard"


def uses_llama_guard_chat_template(config: ResolvedModelConfig) -> bool:
    """Return whether a config should use Llama Guard chat shaping."""
    profile = config.args.get("chat_template_profile")
    if isinstance(profile, str):
        normalized_profile = profile.strip().lower()
        if normalized_profile == LLAMA_GUARD_CHAT_TEMPLATE_PROFILE:
            return True

    model_name = config.model_name or config.args.get("pretrained")
    if not isinstance(model_name, str):
        return False
    normalized_name = model_name.strip().lower()
    return "llama-guard" in normalized_name or "llamaguard" in normalized_name


def prepare_llama_guard_chat_messages(
    sample: NormalizedSample,
    *,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Shape normalized messages for Llama Guard tokenizer templates."""
    template = _chat_template_source(tokenizer)
    use_content_blocks = _chat_template_uses_content_blocks(template)
    requires_alternation = _chat_template_requires_alternation(template)

    if not requires_alternation:
        return [
            {
                "role": message.role,
                "content": _chat_message_content(
                    message.text_content,
                    use_content_blocks=use_content_blocks,
                ),
            }
            for message in sample.messages
        ]

    prepared: list[dict[str, Any]] = []
    for message in sample.messages:
        if message.role not in {"user", "assistant"}:
            continue
        role = message.role
        content = _chat_message_content(
            message.text_content,
            use_content_blocks=use_content_blocks,
        )
        if not prepared:
            if role != "user":
                prepared.append(
                    {
                        "role": "user",
                        "content": _chat_message_content(
                            "",
                            use_content_blocks=use_content_blocks,
                        ),
                    }
                )
            prepared.append({"role": role, "content": content})
            continue
        if prepared[-1]["role"] == role:
            prepared.append(
                {
                    "role": _alternate_role(role),
                    "content": _chat_message_content(
                        "",
                        use_content_blocks=use_content_blocks,
                    ),
                }
            )
        prepared.append({"role": role, "content": content})
    return prepared


def _chat_template_source(tokenizer: Any) -> str:
    """Return the tokenizer chat template as text when available."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return ""
    return str(template)


def _chat_template_uses_content_blocks(template: str) -> bool:
    """Detect templates that expect multimodal-style content blocks."""
    return (
        "selectattr('type', 'equalto', 'text')" in template
        or 'selectattr("type", "equalto", "text")' in template
        or "content['text']" in template
        or 'content["text"]' in template
    )


def _chat_template_requires_alternation(template: str) -> bool:
    """Detect templates that only accept user/assistant alternation."""
    return "Conversation roles must alternate" in template


def _chat_message_content(
    text: str,
    *,
    use_content_blocks: bool,
) -> Any:
    """Format one chat message content payload for the tokenizer."""
    if use_content_blocks:
        return [{"type": "text", "text": text}]
    return text


def _alternate_role(role: str) -> str:
    """Return the synthetic role needed to preserve alternation."""
    if role == "assistant":
        return "user"
    return "assistant"
