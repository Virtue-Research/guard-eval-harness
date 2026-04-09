"""Deterministic mock backend used for tests and examples."""

from __future__ import annotations

from typing import Sequence

from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    NormalizedSample,
)


@model_registry.register("mock")
class MockModelAdapter(ModelAdapter):
    """Deterministic mock model backend."""

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="mock",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            notes=("deterministic", "test-friendly"),
        )

    def predict_batch(
        self,
        samples: Sequence[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        strategy = self.config.args.get("strategy", "label_echo")
        safe_score = float(self.config.args.get("safe_score", 0.1))
        unsafe_score = float(self.config.args.get("unsafe_score", 0.9))
        latency_ms = float(self.config.args.get("latency_ms", 1.0))

        predictions: list[NormalizedPrediction] = []
        for sample in samples:
            if strategy == "label_echo":
                score = unsafe_score if sample.label.unsafe else safe_score
            else:
                content = " ".join(message.text_content for message in sample.messages)
                keywords = tuple(self.config.args.get("unsafe_keywords", ("unsafe",)))
                score = (
                    unsafe_score
                    if any(keyword in content.lower() for keyword in keywords)
                    else safe_score
                )

            predictions.append(
                NormalizedPrediction(
                    sample_id=sample.id,
                    unsafe_score=score,
                    unsafe_label=score >= threshold,
                    threshold=threshold,
                    latency_ms=latency_ms,
                    metadata={"adapter": "mock", "strategy": strategy},
                )
            )
        return predictions
