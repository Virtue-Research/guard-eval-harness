"""Llama Guard 2 / 3 / 4 family guard.

Llama Guard models ship a tokenizer ``chat_template`` that embeds the
MLCommons safety taxonomy automatically. This guard forwards the
conversation as-is and lets the backend's tokenizer apply that
template — there is no policy / output_format knob to tune.

Backend usage:
- Use ``hf_generate`` for local inference; the tokenizer applies the
  chat template automatically.
- Use ``openai_compat`` against a vLLM (or similar) server hosting a
  Llama Guard checkpoint; vLLM's ``/v1/chat/completions`` endpoint
  applies the chat template server-side.
"""

import logging
import re
from typing import Any

from guard_eval_harness.guards.base import Guard, guard_registry
from guard_eval_harness.output_formats import ParsedLabel
from guard_eval_harness.schemas import Message, PredictSample

_log = logging.getLogger(__name__)

_CATEGORY_PATTERN = re.compile(r"\bS\d{1,2}\b", re.IGNORECASE)


def _dedupe_categories(text: str) -> tuple[str, ...]:
    """Extract Sx category tags in order, deduped, uppercased."""
    seen: dict[str, None] = {}
    for match in _CATEGORY_PATTERN.finditer(text):
        seen.setdefault(match.group(0).upper(), None)
    return tuple(seen)


@guard_registry.register("llama_guard")
class LlamaGuard(Guard):
    """Llama Guard family (2 / 3 / 4) — fixed MLCommons-aligned taxonomy."""

    name = "llama_guard"
    backend_kind = "generate"
    accepts_policy = False
    accepts_output_format = False

    def build_messages(
        self,
        sample: PredictSample,
        *,
        policy: Any = None,
        output_format: Any = None,
    ) -> list[Message]:
        """Forward the conversation as-is.

        Llama Guard's chat_template embeds the taxonomy. We must not
        prepend a system message — that would derail the template.
        """
        if policy is not None:
            _log.info(
                "LlamaGuard ignores caller-supplied policy "
                "(model uses its own taxonomy)"
            )
        if output_format is not None:
            _log.info(
                "LlamaGuard ignores caller-supplied output_format "
                "(model uses its own safe/unsafe format)"
            )
        return list(sample.messages)

    def parse(self, output: Any) -> ParsedLabel:
        """Parse `safe` / `unsafe\\nS1,S3` style verdicts."""
        if not isinstance(output, str):
            raise ValueError(
                "LlamaGuard expects str output, got "
                f"{type(output).__name__}"
            )
        text = output.strip()
        if not text:
            raise ValueError("empty Llama Guard output")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lowered = [line.lower() for line in lines]
        categories = _dedupe_categories(text)

        if "unsafe" in lowered:
            return ParsedLabel(
                unsafe_score=1.0,
                predicted_categories=categories,
            )
        if "safe" in lowered:
            return ParsedLabel(unsafe_score=0.0)
        # Fallback: if Sx category tags appear, treat as unsafe.
        if categories:
            return ParsedLabel(
                unsafe_score=1.0,
                predicted_categories=categories,
            )
        raise ValueError(f"unparseable Llama Guard output: {output!r}")
