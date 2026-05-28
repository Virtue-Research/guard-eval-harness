# CLI Reference

The `geh` CLI exposes four subcommands. All commands print JSON to stdout.

```text
geh <command> [options]
```

Run `geh --help` or `geh <command> --help` for the canonical, in-tree help.

## `geh run`

Run a benchmark from a YAML config and stream artifacts to disk.

```text
geh run --config PATH [--output-dir DIR] [--threshold FLOAT]
        [--no-resume | --overwrite] [--recompute-metrics]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | required | YAML run config |
| `--output-dir DIR` | from config | Override `output.run_dir` |
| `--threshold FLOAT` | from config | Override `threshold` |
| `--no-resume` | `false` | Treat the run dir as fresh (errors if non-empty) |
| `--overwrite` | `false` | Wipe the run dir before starting |
| `--recompute-metrics` | `false` | Skip inference; recompute metrics from existing predictions |

Examples:

```bash
geh run --config examples/mock-jsonl.yaml
geh run --config my-run.yaml --overwrite
geh run --config my-run.yaml --recompute-metrics
```

## `geh list`

List registered components in JSON.

```text
geh list {datasets|guards|backends|profiles|policies|output_formats|metrics}
```

| Target | What it returns |
| --- | --- |
| `datasets` | Dataset adapters (80+) |
| `guards` | Guard implementations (`llama_guard`, `llm`, `qwen3guard`, …) |
| `backends` | Inference engines (`hf_generate`, `openai_compat`, …) |
| `profiles` | Bundled `model:` blocks (14 known-good combos) |
| `policies` | Registered policies (built-ins + dataset-scoped) |
| `output_formats` | Parsers for guard outputs (`safe_unsafe_first_line`, …) |
| `metrics` | Supported metrics |

Examples:

```bash
geh list profiles
geh list datasets
geh list backends
```

## `geh validate`

Load a config and confirm every dataset materializes — no inference.

```text
geh validate --config PATH
```

Useful as a pre-flight check before a long run.

## `geh inspect`

Read the manifest, summary, and per-dataset artifacts from a finished run.

```text
geh inspect --run-dir PATH
```
