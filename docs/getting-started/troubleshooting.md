# Troubleshooting

This page focuses on the issues most likely to block a first OSS user: missing
extras, auth problems, path mistakes, and backend expectations that are easy to
miss.

## `geh` Command Not Found

Make sure you installed the project in the environment you are actively using:

```bash
pip install -e "."
geh --help
```

If that still fails, activate the right virtual environment and try again.

## Adapter Exists In Docs But Not In `geh list backends`

This usually means the required extra is missing.

Examples:

- `hf`, `hf_vlm_guard` need `pip install -e ".[hf]"`
- `vllm` needs `pip install -e ".[vllm]"`
- `openai_moderation`, `openai_compatible`, `anthropic`, and `http` need
  `pip install -e ".[api]"`

## OpenAI, Anthropic, Or Gated HuggingFace Auth Fails

Check the environment variables first:

```bash
echo "$OPENAI_API_KEY"
echo "$ANTHROPIC_API_KEY"
echo "$HF_TOKEN"
```

Common fixes:

- export the key in the current shell before running `geh`
- copy `.env.example` to `.env` and source it in your shell tooling
- make sure the selected model or dataset is actually accessible to your account

## `geh validate --config ...` Fails On A Local Path

Local dataset paths in YAML are resolved relative to the config file location,
not the shell directory you happened to run the command from.

If a config lives in `examples/`, use paths that make sense relative to that
file.

## Pack Name Confusion

These are both valid:

```bash
geh run --pack core --model mock
geh run --pack core-v1 --model mock
```

The stable shorthand resolves to the current versioned pack alias.

## Report Or Export Steps Fail After A Run

The safest way to debug is to inspect the run directory first:

```bash
geh inspect --run-dir out/my-run
```

Confirm that these files exist:

- `manifest.json`
- `summary.json`
- `datasets/<dataset>/metrics.json`

## Local Multimodal Runs Fail On File Paths

For `local_image_jsonl` and `local_image_dir`:

- verify the path exists on disk
- use absolute paths if you are debugging path resolution
- confirm the manifest fields point to files the current machine can read

## GPU Backends OOM Or Run Too Slowly

Start by reducing batch size. For local text models, prefer:

```yaml
execution:
  batch_size: auto
```

For multimodal adapters, start conservatively with batch size `1` or `2`.

## A Run Stops Partway Through

Use resume mode for long-running jobs:

```yaml
execution:
  resume: true
```

Then rerun the same config with the same `output.run_dir`.

## Need A Known-Good Starting Point

Use one of these first:

- `geh run --dataset xstest --model mock --limit 50`
- `geh run --config examples/run-mock-jsonl.yaml`
- `geh run --pack core --model openai_moderation --limit 100`
