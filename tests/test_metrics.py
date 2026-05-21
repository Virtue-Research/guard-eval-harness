"""Tests for binary metric helpers."""

import unittest

from guard_eval_harness.metrics import compute_binary_metrics
from guard_eval_harness.schemas import NormalizedPrediction, NormalizedSample


def _sample(sample_id: str, *, unsafe: bool) -> NormalizedSample:
    """Build a minimal normalized sample."""
    return NormalizedSample(
        id=sample_id,
        dataset="metrics-demo",
        split="test",
        messages=[{"role": "user", "content": "x"}],
        label={"unsafe": unsafe},
    )


def _prediction(sample_id: str, score: float, *, threshold: float = 0.5) -> NormalizedPrediction:
    """Build a minimal normalized prediction."""
    return NormalizedPrediction(
        sample_id=sample_id,
        unsafe_score=score,
        unsafe_label=score >= threshold,
        threshold=threshold,
        latency_ms=0.0,
    )


class BinaryMetricsTest(unittest.TestCase):
    """Validate AUROC/AUPRC and threshold sweep outputs."""

    def test_score_metrics_present(self) -> None:
        samples = [
            _sample("s1", unsafe=False),
            _sample("s2", unsafe=False),
            _sample("s3", unsafe=True),
            _sample("s4", unsafe=True),
        ]
        predictions = [
            _prediction("s1", 0.1),
            _prediction("s2", 0.3),
            _prediction("s3", 0.7),
            _prediction("s4", 0.9),
        ]

        metrics = compute_binary_metrics(samples, predictions)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["auprc"], 1.0)
        self.assertIsInstance(metrics["threshold_sweep"], list)
        self.assertEqual(metrics["threshold_sweep"][5]["threshold"], 0.5)

    def test_auroc_and_auprc_none_without_both_classes(self) -> None:
        samples = [_sample("s1", unsafe=False), _sample("s2", unsafe=False)]
        predictions = [_prediction("s1", 0.1), _prediction("s2", 0.2)]

        metrics = compute_binary_metrics(samples, predictions)

        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["auprc"])


if __name__ == "__main__":
    unittest.main()
