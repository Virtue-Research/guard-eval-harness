# Quickstart

This guide gets you from a fresh checkout to a finished evaluation run
quickly, then shows the cleanest next step for a real backend.

## 1. Install The Base Package

```bash
pip install -e "."
```

## 2. Run A Smoke Test

```bash
geh run --dataset xstest --model mock --limit 50
```

Why this first:

- no GPU required
- no API keys required
- you get the full artifact layout immediately

## 3. Inspect The Output

The command prints a JSON payload with the run directory and artifact paths.
The output directory looks like this:

```text
out/mock/xstest/
  manifest.json
  resolved-config.json
  summary.json
  report.html
  datasets/
    xstest/
      predictions.jsonl
      metrics.json
      dataset-manifest.json
```

Follow it with:

```bash
geh inspect --run-dir out/mock/xstest
```

## 4. Try A Real Backend

### Local HuggingFace

```bash
pip install -e ".[hf]"
geh run --dataset xstest,toxic_chat \
    --model hf \
    --model-name meta-llama/Llama-Guard-3-8B \
    --batch-size 16
```

### OpenAI Moderation

```bash
pip install -e ".[api]"
export OPENAI_API_KEY=sk-...
geh run --pack core --model openai_moderation --limit 100
```

### Reproducible YAML Run

```bash
geh run --config examples/run-mock-jsonl.yaml
```

## 5. Compare Or Export Results

```bash
geh compare --run-a out/run-a --run-b out/run-b
geh export --run-dir out/mock/xstest --format csv --output results.csv
```

## What To Do Next

- Use [Run Modes](run-modes.md) if you are not sure when to choose inline,
  pack, or YAML config runs.
- Use [Benchmark Selection](../user-guide/benchmark-selection.md) to find the
  right dataset mix.
- Use [Common Workflows](../user-guide/common-workflows.md) for copy-paste
  recipes across text and image evaluation.
