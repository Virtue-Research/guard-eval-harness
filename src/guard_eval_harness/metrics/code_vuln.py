"""VulnLLM-R–compatible vulnerability detection metrics.

Implements the exact evaluation logic from VulnLLM-R
(arXiv:2512.07533), where a true positive requires both
a correct ``#judge: yes`` AND a matching CWE prediction.

Confusion matrix rules (from ``evaluate_examples`` in
``VulnLLM-R/vulscan/test/test_utils/generation_utils.py``):

- **TP**: pred=yes, true=yes, single CWE predicted, CWE
  matches ground truth (substring match).
- **FP**: pred=yes, true=no.
- **FN**: pred=no AND true=yes; OR pred=yes AND true=yes
  but wrong/missing/multiple CWE; OR invalid format AND
  true=yes.
- **TN**: pred=no, true=no.
- **Invalid format**: counted as FN (if true=yes) or FP
  (if true=no).
"""

from __future__ import annotations

import re
from typing import Sequence

from guard_eval_harness.metrics.binary import _safe_divide
from guard_eval_harness.schemas import (
    NormalizedPrediction,
    NormalizedSample,
)


_CWE_PATTERN = re.compile(r"CWE-\d+", re.IGNORECASE)


def _cwe_matches(
    predicted_cwe: str,
    ground_truth_cwes: tuple[str, ...],
) -> bool:
    """Check CWE match using VulnLLM-R's substring logic.

    VulnLLM-R uses:
    ``pred_vul_type in true_vul_type or
    true_vul_type in pred_vul_type``
    """
    pred = predicted_cwe.upper()
    for gt in ground_truth_cwes:
        gt_upper = gt.upper()
        if pred in gt_upper or gt_upper in pred:
            return True
    return False


def _has_single_cwe(predicted_categories: tuple[str, ...]) -> bool:
    """Check that exactly one CWE is predicted.

    VulnLLM-R's ``check_single_cwe`` counts CWE-\\d+
    matches and requires exactly 1.
    """
    if not predicted_categories:
        return False
    all_cwes: list[str] = []
    for cat in predicted_categories:
        all_cwes.extend(_CWE_PATTERN.findall(cat))
    return len(all_cwes) == 1


def compute_code_vuln_metrics(
    samples: Sequence[NormalizedSample],
    predictions: Sequence[NormalizedPrediction],
) -> dict[str, float | int | None]:
    """Compute VulnLLM-R–compatible CWE-aware metrics.

    This function replicates the exact TP/FP/FN/TN counting
    from ``VulnLLM-R/vulscan/test/test_utils/
    generation_utils.py:evaluate_examples``.

    Results include the same metric keys as VulnLLM-R:
    ``accuracy``, ``pos_Precision``, ``pos_Recall``,
    ``positive F1``, ``negative F1``, ``overall F1``,
    ``false_positive_rate``, ``false_negative_rate``,
    ``wrong_num``.
    """
    prediction_by_id = {
        prediction.sample_id: prediction for prediction in predictions
    }

    tp = tn = fp = fn = 0
    wrong_num = 0
    total = len(samples)

    for sample in samples:
        prediction = prediction_by_id.get(sample.id)
        true_unsafe = sample.label.unsafe
        if prediction is None:
            # Partial-prediction adapters may drop malformed
            # judge outputs. VulnLLM-R treats those as invalid
            # format, which counts as FN for vulnerable samples
            # and FP for benign samples.
            if true_unsafe:
                fn += 1
            else:
                fp += 1
            continue

        pred_unsafe = prediction.unsafe_label

        if pred_unsafe and true_unsafe:
            # Both say vulnerable — check CWE correctness
            pred_cats = prediction.predicted_categories
            if not _has_single_cwe(pred_cats):
                # Multiple or zero CWEs predicted → FN
                fn += 1
                wrong_num += 1
            elif _cwe_matches(
                pred_cats[0],
                sample.category_labels,
            ):
                tp += 1
            else:
                # Wrong CWE → FN
                fn += 1
        elif pred_unsafe and not true_unsafe:
            fp += 1
        elif not pred_unsafe and true_unsafe:
            fn += 1
        elif not pred_unsafe and not true_unsafe:
            tn += 1

    # Positive class = vulnerable
    pos_precision = _safe_divide(tp, tp + fp)
    pos_recall = _safe_divide(tp, tp + fn)
    pos_f1 = None
    if (
        pos_precision is not None
        and pos_recall is not None
        and (pos_precision + pos_recall) > 0
    ):
        pos_f1 = (
            2 * pos_precision * pos_recall
            / (pos_precision + pos_recall)
        )

    # Negative class = benign
    neg_precision = _safe_divide(tn, tn + fn)
    neg_recall = _safe_divide(tn, tn + fp)
    neg_f1 = None
    if (
        neg_precision is not None
        and neg_recall is not None
        and (neg_precision + neg_recall) > 0
    ):
        neg_f1 = (
            2 * neg_precision * neg_recall
            / (neg_precision + neg_recall)
        )

    overall_f1 = None
    if pos_f1 is not None and neg_f1 is not None:
        overall_f1 = (pos_f1 + neg_f1) / 2

    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    accuracy = _safe_divide(tp + tn, total) if total else None

    return {
        "count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "wrong_num": wrong_num,
        "accuracy": accuracy,
        "pos_Precision": pos_precision,
        "pos_Recall": pos_recall,
        "positive F1": pos_f1,
        "negative F1": neg_f1,
        "overall F1": overall_f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }
