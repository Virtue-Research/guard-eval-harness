"""Prompt-injection / jailbreak text classifier guard.

Wraps the family of small discriminative text-classifier safety
heads:

  - ``meta-llama/Prompt-Guard-86M`` — 3 labels (BENIGN / INJECTION
    / JAILBREAK).
  - ``meta-llama/Llama-Prompt-Guard-2-22M`` — 2 labels (BENIGN /
    INJECTION).

The backend (``hf_text_classifier``) returns a label distribution;
this guard maps it to a single ``unsafe_score``. Configurability
mirrors ``HFImageClassifierGuard``: pick either ``unsafe_labels``
(sum-of-probs) or ``label_score_mapping`` (weighted sum).
"""

import logging
from typing import Any

from guard_eval_harness.guards.base import Guard, guard_registry
from guard_eval_harness.output_formats import ParsedLabel
from guard_eval_harness.schemas import Message, PredictSample

_log = logging.getLogger(__name__)


@guard_registry.register("prompt_guard")
class PromptGuardGuard(Guard):
    """Generic text-classifier guard (prompt injection / jailbreak)."""

    name = "prompt_guard"
    backend_kind = "classify"
    accepts_policy = False
    accepts_output_format = False

    def __init__(
        self,
        *,
        unsafe_labels: tuple[str, ...] | list[str] = (),
        label_score_mapping: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if bool(unsafe_labels) == bool(label_score_mapping):
            raise ValueError(
                "PromptGuardGuard: provide exactly one of "
                "`unsafe_labels` or `label_score_mapping` in guard_args"
            )
        self.unsafe_labels = tuple(unsafe_labels)
        self.label_score_mapping = (
            dict(label_score_mapping)
            if label_score_mapping is not None
            else None
        )
        if self.label_score_mapping is not None:
            for label, weight in self.label_score_mapping.items():
                if not 0.0 <= float(weight) <= 1.0:
                    raise ValueError(
                        "PromptGuardGuard: label_score_mapping weight "
                        f"for {label!r} must be in [0, 1], got {weight}"
                    )

    def build_messages(
        self,
        sample: PredictSample,
        *,
        policy: Any = None,
        output_format: Any = None,
    ) -> list[Message]:
        """Forward messages as-is; the backend concatenates text."""
        if policy is not None:
            _log.info(
                "PromptGuardGuard ignores caller-supplied policy"
            )
        if output_format is not None:
            _log.info(
                "PromptGuardGuard ignores caller-supplied output_format"
            )
        return list(sample.messages)

    def parse(self, output: Any) -> ParsedLabel:
        """Map a label-probability dict into ``unsafe_score``."""
        if not isinstance(output, dict):
            raise ValueError(
                "PromptGuardGuard expects dict[str, float] from the "
                f"backend, got {type(output).__name__}"
            )
        category_scores = {
            str(label): float(prob) for label, prob in output.items()
        }

        if self.unsafe_labels:
            unsafe_score = sum(
                category_scores.get(label, 0.0)
                for label in self.unsafe_labels
            )
            unsafe_score = max(0.0, min(1.0, unsafe_score))
            predicted = tuple(
                label
                for label in self.unsafe_labels
                if category_scores.get(label, 0.0) > 0.0
            )
            return ParsedLabel(
                unsafe_score=unsafe_score,
                predicted_categories=predicted,
                category_scores=category_scores,
            )

        assert self.label_score_mapping is not None
        unsafe_score = sum(
            category_scores.get(label, 0.0) * weight
            for label, weight in self.label_score_mapping.items()
        )
        unsafe_score = max(0.0, min(1.0, unsafe_score))
        return ParsedLabel(
            unsafe_score=unsafe_score,
            predicted_categories=tuple(
                label
                for label in self.label_score_mapping
                if category_scores.get(label, 0.0) > 0.0
            ),
            category_scores=category_scores,
        )
