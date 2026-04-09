# Metrics

Guard Eval Harness computes **binary classification metrics** for each dataset in a run. All metrics are computed from the confusion matrix of ground-truth labels vs. model predictions (thresholded at the configured threshold).

## Available Metrics

```bash
geh list metrics
```

### Binary Classification

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted unsafe, how many are truly unsafe |
| **Recall** | TP / (TP + FN) | Of truly unsafe, how many were caught |
| **F1** | 2 * (P * R) / (P + R) | Harmonic mean of precision and recall |
| **AUROC** | Area under ROC | Ranking quality across all thresholds |
| **AUPRC** | Area under PR curve | Precision-recall tradeoff quality |
| **FPR** | FP / (FP + TN) | False positive rate (over-blocking) |
| **FNR** | FN / (FN + TP) | False negative rate (missed unsafe content) |

### Confusion Matrix

| Metric | Description |
|--------|-------------|
| **TP** | True positives — correctly identified unsafe |
| **TN** | True negatives — correctly identified safe |
| **FP** | False positives — safe content flagged as unsafe |
| **FN** | False negatives — unsafe content missed |

## How Metrics Are Computed

1. Each sample gets a `unsafe_score` in `[0.0, 1.0]` from the model adapter
2. The score is thresholded: `unsafe_label = unsafe_score >= threshold`
3. Binary metrics are computed from the confusion matrix of `label` vs `unsafe_label`
4. AUROC and AUPRC use the raw `unsafe_score` (threshold-independent)

## Output Format

Metrics are written to `metrics.json` per dataset:

```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.95,
  "f1": 0.92,
  "auroc": 0.97,
  "auprc": 0.96,
  "fpr": 0.11,
  "fnr": 0.05,
  "tp": 190,
  "tn": 230,
  "fp": 28,
  "fn": 10
}
```

Aggregated metrics across all datasets appear in `summary.json`.

## Interpreting Results

### For Safety Guardrails

- **High recall** is critical — you don't want to miss unsafe content (low FNR)
- **Acceptable FPR** depends on your use case — some over-blocking may be tolerable
- **F1** balances precision and recall — good for overall comparison
- **AUROC** measures ranking quality independent of threshold — useful for model comparison

### Threshold Tuning

The `threshold` parameter directly affects precision/recall tradeoff:

| Threshold | Effect |
|-----------|--------|
| Lower (e.g., 0.3) | Higher recall, more false positives (conservative) |
| Default (0.5) | Balanced |
| Higher (e.g., 0.7) | Higher precision, more false negatives (permissive) |

!!! tip
    Use AUROC to compare models independently of threshold, then tune the threshold for your deployment requirements.

## Partial Predictions

If a model adapter fails on some samples (with `drop_failed_predictions: true`), the harness validates the prediction set and flags runs with high drop rates:

| Drop Rate | Status |
|-----------|--------|
| < 5% | Normal — metrics computed on available predictions |
| 5–20% | Warning logged |
| > 20% | Run flagged as `"partial"` in manifest |

## Code Vulnerability Metrics

For code benchmarks, additional specialized metrics are computed beyond binary classification, including per-vulnerability-type breakdowns (SQL injection, XSS, buffer overflow, etc.).
