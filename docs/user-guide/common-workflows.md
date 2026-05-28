# Common Workflows

Copy-paste recipes. Every example is a YAML config under [`examples/`](../../examples/).

## 1. Smoke-test the harness

```bash
uv run geh run --config examples/mock-jsonl.yaml
```

No GPU, no API keys. Use on a fresh machine or in CI.

## 2. Llama Guard via vLLM

```bash
# Start the model server first:
vllm serve meta-llama/Llama-Guard-3-8B --port 8000

uv run geh run --config examples/llama-guard-vllm.yaml
```

The profile `llama-guard-3-8b` defaults to local HF; the example overrides the
backend to point at the vLLM endpoint.

## 3. GPT-as-judge

```bash
export OPENAI_API_KEY=sk-...
uv run geh run --config examples/openai-judge.yaml
```

The generic `llm` guard pairs any chat model with a Policy + OutputFormat —
swap `model.backend.name` for any OpenAI / Anthropic / vLLM chat model.

## 4. One profile across several datasets

```bash
uv run geh run --config examples/profile-multi-dataset.yaml
```

Each dataset gets its own `predictions.jsonl` and `metrics.json`; the
`summary.json` rolls them up.

## 5. HF text classifier (no chat template)

```bash
uv run geh run --config examples/hf-text-classifier.yaml
```

`hf_text_classifier` is a non-generative backend — it reads the classifier
head's per-label scores directly. Use for Prompt-Guard-86M,
Llama-Prompt-Guard-22M, and any compatible HF model.

## 6. Pre-flight a config

```bash
uv run geh validate --config examples/profile-multi-dataset.yaml
```

Loads the config and confirms every dataset materializes — no inference, no
network calls beyond resolving HF datasets. Run this before any long job.

## 7. Resume / overwrite

```bash
uv run geh run --config run.yaml                  # resumes (default)
uv run geh run --config run.yaml --overwrite      # wipe + start fresh
uv run geh run --config run.yaml --recompute-metrics   # re-score predictions only
```

## 8. Inspect a finished run

```bash
uv run geh inspect --run-dir out/my-run
```

Prints the manifest + summary as JSON for piping into `jq`.
