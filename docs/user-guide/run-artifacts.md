# Run Artifacts

Every evaluation run produces a **self-contained output directory** with all predictions, metrics, and reports. This directory is portable and can be inspected, compared, or exported without re-running the evaluation.

## Directory Layout

```
out/my-run/
├── manifest.json              # Run metadata
├── resolved-config.json       # Exact config snapshot (sanitized)
├── resume-signature.json      # Hash for resume validation
├── summary.json               # Aggregated metrics across all datasets
├── report.html                # Static HTML report
└── datasets/
    ├── xstest/
    │   ├── predictions.jsonl      # Per-sample predictions
    │   ├── metrics.json           # Dataset-level metrics
    │   └── dataset-manifest.json  # Dataset metadata
    └── toxic_chat/
        ├── predictions.jsonl
        ├── metrics.json
        └── dataset-manifest.json
```

## File Formats

### `manifest.json`

Top-level run metadata including:

- Tool version and run name
- Run status: `"completed"`, `"failed"`, or `"partial"`
- Start and end timestamps
- Resolved config hash
- Model and execution configuration
- Per-dataset metadata and adapter capabilities
- Environment info (Python version, platform, hostname)

### `predictions.jsonl`

One JSON object per line, each representing a `NormalizedPrediction`:

```json
{
  "sample_id": "xstest-001",
  "unsafe_score": 0.87,
  "unsafe_label": true,
  "threshold": 0.5,
  "latency_ms": 42.3,
  "predicted_categories": ["violence"],
  "category_scores": {"violence": 0.87, "sexual": 0.02},
  "metadata": {}
}
```

### `metrics.json`

Dataset-level binary classification metrics:

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

### `summary.json`

Aggregated metrics across all datasets in the run.

### `report.html`

A static, single-file HTML report with:

- Per-dataset metrics table
- Sortable columns
- Responsive design — open in any browser, no server needed

## Inspecting Runs

```bash
# View run manifest, summary, and artifacts
geh inspect --run-dir out/my-run

# Rebuild the HTML report
geh report --run-dir out/my-run
```

## Comparing Runs

```bash
geh compare --run-a out/run1 --run-b out/run2
```

Produces a side-by-side diff of metrics for datasets present in both runs, with deltas highlighted.

## Exporting

```bash
# Export to CSV
geh export --run-dir out/my-run --format csv --output results.csv

# Export to Excel
geh export --run-dir out/my-run --format xlsx --output results.xlsx

# Export to JSON
geh export --run-dir out/my-run --format json --output results.json
```
