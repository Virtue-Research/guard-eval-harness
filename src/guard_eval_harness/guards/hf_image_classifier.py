"""Image-classifier guard: maps a label distribution to ``unsafe_score``.

Pair with the ``hf_image_classifier`` backend (or any classifier
backend that returns ``dict[str, float]`` of label probabilities).

Two scoring strategies, choose one in ``guard_args``:

  unsafe_labels: list[str]
    Label names whose probabilities are *summed* to produce the
    unsafe score. Any label not in this list contributes 0.
    Example: ``["nsfw", "porn"]``

  label_score_mapping: dict[str, float]
    Explicit per-label weight in ``[0, 1]``. The unsafe score is
    ``sum(probs[label] * weight)`` clamped to ``[0, 1]``.
    Example: ``{"nsfw": 1.0, "questionable": 0.5}``

Exactly one of the two strategies must be supplied (in ``guard_args``).
"""

import logging
from typing import Any

from guard_eval_harness.guards.base import Guard, guard_registry
from guard_eval_harness.output_formats import ParsedLabel
from guard_eval_harness.schemas import Message, PredictSample

_log = logging.getLogger(__name__)


@guard_registry.register("hf_image_classifier")
class HFImageClassifierGuard(Guard):
    """Generic discriminative-image-classifier guard."""

    name = "hf_image_classifier"
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
                "HFImageClassifierGuard: provide exactly one of "
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
                        "HFImageClassifierGuard: label_score_mapping "
                        f"weight for {label!r} must be in [0, 1], "
                        f"got {weight}"
                    )

    def build_messages(
        self,
        sample: PredictSample,
        *,
        policy: Any = None,
        output_format: Any = None,
    ) -> list[Message]:
        """Forward messages as-is; the backend extracts the image."""
        if policy is not None:
            _log.info(
                "HFImageClassifierGuard ignores caller-supplied policy"
            )
        if output_format is not None:
            _log.info(
                "HFImageClassifierGuard ignores caller-supplied "
                "output_format"
            )
        return list(sample.messages)

    def parse(self, output: Any) -> ParsedLabel:
        """Map label probabilities to ``unsafe_score``."""
        if not isinstance(output, dict):
            raise ValueError(
                "HFImageClassifierGuard expects dict[str, float] from "
                f"the backend, got {type(output).__name__}"
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
