"""Granite Guardian text safety guard (IBM ibm-granite/granite-guardian-*).

Granite Guardian models ship a tokenizer chat template that embeds
IBM's risk taxonomy automatically. The guard forwards the
conversation as-is; the model emits a single ``Yes`` (unsafe) or
``No`` (safe) token.

Backend usage:
  - ``hf_generate`` with ``apply_chat_template=true`` and
    ``max_new_tokens: 2`` is sufficient.
  - The same checkpoint can be hosted on a vLLM server and reached
    via ``openai_compat``.
"""

import logging
import re
from typing import Any

from guard_eval_harness.guards.base import Guard, guard_registry
from guard_eval_harness.output_formats import ParsedLabel
from guard_eval_harness.schemas import Message, PredictSample

_log = logging.getLogger(__name__)


_FIRST_TOKEN_RE = re.compile(r"[A-Za-z]+")


@guard_registry.register("granite_guardian")
class GraniteGuardianGuard(Guard):
    """Granite Guardian — single-token Yes/No safety classifier."""

    name = "granite_guardian"
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

        Granite Guardian's chat template adds the risk-framing system
        prompt internally. Prepending one here would derail it.
        """
        if policy is not None:
            _log.info(
                "GraniteGuardianGuard ignores caller-supplied policy "
                "(model uses its own taxonomy)"
            )
        if output_format is not None:
            _log.info(
                "GraniteGuardianGuard ignores caller-supplied "
                "output_format (model emits its own Yes/No format)"
            )
        return list(sample.messages)

    def parse(self, output: Any) -> ParsedLabel:
        """Parse the first alphabetic token: ``Yes`` (unsafe) or ``No`` (safe)."""
        if not isinstance(output, str):
            raise ValueError(
                "GraniteGuardianGuard expects str output, got "
                f"{type(output).__name__}"
            )
        text = output.strip()
        if not text:
            raise ValueError("empty Granite Guardian output")
        match = _FIRST_TOKEN_RE.search(text)
        if match is None:
            raise ValueError(
                f"unparseable Granite Guardian output (no alphabetic "
                f"token): {output!r}"
            )
        first = match.group(0).lower()
        if first == "yes":
            return ParsedLabel(unsafe_score=1.0)
        if first == "no":
            return ParsedLabel(unsafe_score=0.0)
        raise ValueError(
            f"unparseable Granite Guardian output (first token must "
            f"be 'Yes' or 'No'): {output!r}"
        )
