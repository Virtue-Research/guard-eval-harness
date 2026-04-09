# CLI Reference

Guard Eval Harness exposes a single `geh` CLI for running evaluations,
inspecting artifacts, and discovering available adapters and suites.

## Global Usage

```text
geh <command> [options]
```

Most commands print JSON so the output is easy to inspect manually or pipe into
other tooling.

## `geh run`

Run a benchmark and write artifacts to disk.

```text
geh run [--config PATH | --pack NAME | --dataset NAMES] [options]
```

### Source Flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | run from a YAML configuration file |
| `--pack NAME` | run a curated benchmark pack such as `core` |
| `--dataset NAMES` | run one or more comma-separated dataset adapters inline |

### Model Flags

These are required when using `--pack` or `--dataset`.

| Flag | Meaning |
| --- | --- |
| `--model ADAPTER` | model adapter alias, and for pack flows may also resolve a supported catalog slug |
| `--model-name NAME` | backend-specific model identifier |
| `--model-args JSON` | adapter-specific JSON args |

### Common Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--batch-size INT` | `1` | batch size for inference |
| `--threshold FLOAT` | `0.5` | unsafe classification threshold |
| `--limit INT` | all | max samples per dataset |
| `--output-dir PATH` | auto | override artifact output path |
| `--no-sample-cache` | `false` | disable image/media sample caching |

### Examples

```bash
geh run --dataset xstest --model mock --limit 50
geh run --dataset xstest,toxic_chat --model hf --model-name meta-llama/Llama-Guard-3-8B
geh run --pack core --model openai_moderation --limit 100
geh run --config examples/run-mock-jsonl.yaml
```

## `geh list`

Discover registered components.

```text
geh list {datasets|backends|metrics|packs|presets|plugins}
```

### What Each Target Means

| Target | What it returns |
| --- | --- |
| `datasets` | registered dataset adapter aliases |
| `backends` | registered model adapter aliases |
| `metrics` | built-in metric names |
| `packs` | curated user-facing benchmark packs |
| `presets` | code-defined reproducible benchmark suites, currently including `21x31` |
| `plugins` | the active dataset/model registry view after built-ins and entry-point plugins are loaded |

### Examples

```bash
geh list datasets
geh list backends
geh list packs
geh list presets
geh list plugins
```

`geh list plugins` is useful when you want to confirm that an installed
entry-point plugin was actually discovered by the harness.

## `geh inspect`

Inspect the stored manifest, summary, and dataset artifact layout for a prior
run.

```text
geh inspect --run-dir PATH
```

## `geh report`

Rebuild summary artifacts from an existing run directory.

```text
geh report --run-dir PATH
```

Use this when the run already exists and you want to regenerate `summary.json`
or `report.html` from stored outputs.

## `geh compare`

Compare two run directories.

```text
geh compare --run-a PATH --run-b PATH
```

Useful for model swaps, threshold changes, or regressions across the same
dataset mix.

## `geh export`

Export summary artifacts to a simpler stakeholder format.

```text
geh export --run-dir PATH --format {json|csv|xlsx} --output PATH
```

## `geh validate`

Validate a YAML config without running the full benchmark.

```text
geh validate --config PATH
```

This resolves adapters, validates local dataset paths, and confirms the config
can be materialized into a run.

## `geh cache`

Inspect or clear the normalized sample cache used for media-heavy datasets.

```text
geh cache {status|clear} [--dataset NAME]
```

Examples:

```bash
geh cache status
geh cache clear
geh cache clear --dataset unsafebench
```
