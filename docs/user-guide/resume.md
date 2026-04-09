# Resuming Runs

Guard Eval Harness supports **resuming interrupted runs** so you don't lose progress on long evaluations.

## How It Works

When `resume: true` is set in the config (or using `--config` with a resume-enabled YAML), the harness:

1. Checks for an existing run directory with a matching **resume signature**
2. Loads cached predictions from `predictions.jsonl`
3. Skips already-evaluated samples
4. Runs inference only on **pending samples**
5. Merges cached and fresh predictions
6. Recomputes metrics on the full set

The **resume signature** is a hash of the model config, dataset config, and threshold — ensuring you only resume when the configuration is identical.

## Configuration

### YAML Config

```yaml
execution:
  resume: true
  batch_size: auto    # Adaptive batch sizing works well with resume

output:
  run_dir: out/my-long-run
```

### Example

```yaml title="examples/run-mock-jsonl-auto-resume.yaml"
version: 1
run_name: mock-jsonl-auto-resume
threshold: 0.5
model:
  adapter: mock
  args:
    strategy: label_echo
    safe_score: 0.1
    unsafe_score: 0.9
    latency_ms: 1.0
datasets:
  - name: mock_jsonl
    adapter: local_jsonl
    path: datasets/mock_samples.jsonl
    split: test
output:
  run_dir: out/mock-jsonl-auto-resume
execution:
  batch_size: auto
  concurrency: 1
  resume: true
```

## Signature Validation

If you change the model, dataset, or threshold between runs, the resume signature won't match and the harness will start fresh. This prevents mixing results from incompatible configurations.

!!! warning
    Resume relies on the `run_dir` path remaining the same. If you change `run_dir`, there are no cached predictions to resume from.

## When to Use Resume

- **Large dataset evaluations** that may be interrupted (OOM, timeout, API rate limits)
- **Iterative development** where you want to add samples incrementally
- **API-based evaluations** with retry/backoff that may partially complete

## Auto Batch Size

When combined with `batch_size: auto`, the harness will adaptively reduce batch sizes on OOM errors for local models. This is especially useful for GPU-based evaluations where memory limits are hard to predict.
