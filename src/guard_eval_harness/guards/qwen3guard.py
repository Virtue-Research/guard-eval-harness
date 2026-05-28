"""Qwen3Guard text safety guard (Qwen/Qwen3Guard-Gen-{0.6B,4B}).

Qwen3Guard ships its own chat template baked into the tokenizer, so
this guard simply forwards the sample's messages as-is and lets
``hf_generate`` (or a vLLM endpoint hosting the same checkpoint) apply
the template server-side.

Output format is three labels:
  - ``Safe`` — classifier verdict 0.0
  - ``Unsafe`` — classifier verdict 1.0 (followed by category line)
  - ``Controversial`` — classifier verdict 0.5
"""

import logging
import re
from typing import Any

from guard_eval_harness.guards.base import Guard, guard_registry
from guard_eval_harness.output_formats import ParsedLabel
from guard_eval_harness.schemas import Message, PredictSample

_log = logging.getLogger(__name__)


_SAFETY_LINE_RE = re.compile(
    r"safety\s*:\s*(safe|unsafe|controversial)", re.IGNORECASE
)
_CATEGORIES_LINE_RE = re.compile(
    r"categor(?:y|ies)\s*:\s*(.+)", re.IGNORECASE
)


def _parse_categories(line: str) -> tuple[str, ...]:
    """Split a Qwen3Guard category line on commas / semicolons."""
    cleaned: list[str] = []
    for token in re.split(r"[,;]+", line):
        item = token.strip()
        if item and item.lower() != "none":
            cleaned.append(item)
    return tuple(dict.fromkeys(cleaned))


@guard_registry.register("qwen3guard")
class Qwen3GuardGuard(Guard):
    """Qwen3Guard-Gen — Safe / Unsafe / Controversial classifier."""

    name = "qwen3guard"
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

        The Qwen3Guard tokenizer's chat template embeds the moderator
        framing automatically. Prepending a system message would
        derail the trained template.
        """
        if policy is not None:
            _log.info(
                "Qwen3GuardGuard ignores caller-supplied policy "
                "(model uses its own taxonomy)"
            )
        if output_format is not None:
            _log.info(
                "Qwen3GuardGuard ignores caller-supplied output_format "
                "(model emits its own Safety/Categories format)"
            )
        return list(sample.messages)

    def parse(self, output: Any) -> ParsedLabel:
        """Parse ``Safety: <verdict>`` (+ optional ``Categories:`` line)."""
        if not isinstance(output, str):
            raise ValueError(
                "Qwen3GuardGuard expects str output, got "
                f"{type(output).__name__}"
            )
        text = output.strip()
        if not text:
            raise ValueError("empty Qwen3Guard output")

        safety_match = _SAFETY_LINE_RE.search(text)
        if safety_match is None:
            # Tolerant fallback: bare verdict on the first line.
            first_token = text.splitlines()[0].strip().lower()
            if first_token in {"safe", "unsafe", "controversial"}:
                verdict = first_token
            else:
                raise ValueError(
                    f"unparseable Qwen3Guard output (missing Safety: "
                    f"line): {output!r}"
                )
        else:
            verdict = safety_match.group(1).lower()

        categories_match = _CATEGORIES_LINE_RE.search(text)
        categories: tuple[str, ...] = ()
        if categories_match is not None:
            categories = _parse_categories(categories_match.group(1))

        if verdict == "safe":
            return ParsedLabel(unsafe_score=0.0)
        if verdict == "unsafe":
            return ParsedLabel(
                unsafe_score=1.0,
                predicted_categories=categories,
            )
        # controversial
        return ParsedLabel(
            unsafe_score=0.5,
            predicted_categories=categories,
            raw_metadata={"verdict": "controversial"},
        )
