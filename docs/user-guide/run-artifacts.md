# Run Artifacts

Every run writes a self-contained directory: predictions, metrics, and a
resume signature.

## Layout

```
out/<run-name>/
├── manifest.json              # run metadata + status + config hash
├── resolved-config.json       # exact config snapshot
├── summary.json               # aggregated metrics across datasets
└── datasets/
    ├── xstest/
    │   ├── predictions.jsonl     # one record per sample
    │   ├── metrics.json          # per-dataset metrics
    │   └── dataset-manifest.json # adapter version + source URI
    └── toxic_chat/
        ├── predictions.jsonl
        ├── metrics.json
        └── dataset-manifest.json
```

## File formats

### `manifest.json`

Top-level run metadata:

- harness version, run name, status (`completed` / `partial` / `failed`)
- `config_hash` — SHA-256 used to gate resume
- start / end timestamps
- resolved model + dataset entries
- environment (Python, platform, hostname)

### `predictions.jsonl`

One JSON object per sample:

```json
{
  "sample_id": "xstest-001",
  "unsafe_score": 0.87,
  "unsafe_label": true,
  "threshold": 0.5,
  "latency_ms": 42.3,
  "predicted_categories": ["violence"],
  "category_scores": {"violence": 0.87, "sexual": 0.02},
  "raw_output": "...",
  "metadata": {}
}
```

### `metrics.json`

Per-dataset binary-classification metrics:

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
  "tp": 190, "tn": 230, "fp": 28, "fn": 10,
  "count": 458,
  "evaluated_sample_count": 458,
  "failed_sample_count": 0,
  "threshold_sweep": [ /* per-threshold breakdown */ ]
}
```

### `summary.json`

`{"datasets": [...]}` with one entry per dataset — same keys as `metrics.json`
plus `name`.

## Inspecting a run

```bash
geh inspect --run-dir out/my-run
```

Prints the manifest + summary as JSON, ready for `jq`.

## Resume

`manifest.json` records a SHA-256 hash of the resolved config. Re-running with
the same config picks up at the next unprocessed sample. Running with a
different config in a non-empty run dir fails fast — use `--overwrite` to
wipe it.

## Recomputing metrics

If you change the threshold or add a metric, you don't need to re-run inference:

```bash
geh run --config run.yaml --recompute-metrics
```

This re-scores existing `predictions.jsonl` files and rewrites the metric files.
