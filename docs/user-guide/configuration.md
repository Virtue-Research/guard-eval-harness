# Configuration

Guard Eval Harness supports three ways to configure a run:

| Mode | Command | Best For |
|------|---------|----------|
| **Inline** | `geh run --dataset ... --model ...` | Quick one-off evaluations |
| **YAML Config** | `geh run --config path.yaml` | Reproducible, version-controlled runs |
| **Benchmark Pack** | `geh run --pack core --model ...` | Curated multi-dataset evaluations |

These are mutually exclusive — pick one per run.

## YAML Config Reference

```yaml title="full-config.yaml"
version: 1                           # (1)!
run_name: my-evaluation              # (2)!
threshold: 0.5                       # (3)!

model:
  adapter: hf                        # (4)!
  model_name: meta-llama/Llama-Guard-3-8B  # (5)!
  args:                              # (6)!
    apply_chat_template: true
    drop_failed_predictions: true

datasets:                            # (7)!
  - name: xstest
    adapter: xstest
    split: test
    options: {}

  - name: toxic_chat
    adapter: toxic_chat

execution:
  batch_size: 16                     # (8)!
  concurrency: 1                     # (9)!
  retries: 8                         # (10)!
  retry_backoff: 2.0                 # (11)!
  limit: null                        # (12)!
  resume: false                      # (13)!

output:
  run_dir: out/my-evaluation         # (14)!
  overwrite: false                   # (15)!
```

1. Config format version. Always `1`.
2. Human-readable name. Used in artifact directory naming.
3. Classification threshold (0.0–1.0). Scores above this are classified as `unsafe`.
4. Model backend alias. Use any registered adapter returned by `geh list backends`; most users start with `mock`, `hf`, `vllm`, `openai_moderation`, `openai_compatible`, `anthropic`, or `http`.
5. Model identifier — HuggingFace repo ID, API model name, etc.
6. Backend-specific arguments. See [Models](../models/overview.md) for each adapter's options.
7. One or more datasets to evaluate. At least one is required.
8. Batch size for inference. Use `"auto"` for adaptive sizing (local models only).
9. Parallel workers for API adapters.
10. Retry attempts on transient failures.
11. Exponential backoff multiplier between retries.
12. Max samples per dataset. `null` = use all samples.
13. Resume from cached predictions if the run directory exists.
14. Output directory (required).
15. Allow overwriting an existing run directory.

## Dataset Config

Each dataset entry supports these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **required** | Name used in artifact paths |
| `adapter` | `str` | same as `name` | Dataset adapter name |
| `path` | `str` | `null` | Local file/directory path |
| `split` | `str` | `"test"` | Dataset split |
| `id_field` | `str` | `"id"` | Sample ID column |
| `prompt_field` | `str` | `"prompt"` | Text input column |
| `response_field` | `str` | `null` | Optional response/output column |
| `messages_field` | `str` | `null` | Structured messages column |
| `label_field` | `str` | `"unsafe"` | Safety label column |
| `metadata_fields` | `list[str]` | `[]` | Extra columns to preserve |
| `options` | `dict` | `{}` | Adapter-specific options |

## Environment Variables

Config values can reference environment variables with `${VAR_NAME}` syntax:

```yaml
model:
  adapter: openai_compatible
  model_name: ${OPENAI_VISION_MODEL}
  args:
    api_key_env: OPENAI_API_KEY
```

## Validation

Validate a config file without running it:

```bash
geh validate --config my-eval.yaml
```

This checks that all adapters exist, required fields are present, and dataset/model configurations are valid.

## CLI Overrides

Several config values can be overridden from the command line:

```bash
geh run --config my-eval.yaml \
    --threshold 0.6 \
    --limit 100 \
    --output-dir out/custom-dir \
    --batch-size 32
```
