<p align="center">
  <img src="assets/logo.png" alt="guard-eval-harness" width="200">
</p>

<h1 align="center">guard-eval-harness</h1>

<p align="center">Benchmark any safety guard against 80+ canonical datasets with a single YAML.</p>

A run is just two things: a **guard** (prompt + parser) and a **backend** (inference engine). Mix them freely — Llama Guard via vLLM, an OpenAI chat model judged by your own policy, an HF text classifier on a local GPU — all driven by the same config.

## Install

```bash
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness
uv sync --extra hf            # base + local HuggingFace inference
# or: uv sync --extra api     # base + retry/auth for hosted endpoints
# or: pip install -e ".[hf]"
```

Set API keys via env vars only for the backends you use:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`.

## 30-second demo

```bash
# No API keys, no GPU — just confirms the pipeline runs.
uv run geh run --config examples/mock-jsonl.yaml
```

## Real run via a bundled profile

```yaml title="my-run.yaml"
run_name: llama-guard-on-xstest
model:
  profile: llama-guard-3-8b      # one of 14 bundled profiles
datasets:
  - name: xstest
    limit: 100
output:
  run_dir: out/llama-guard-on-xstest
```

```bash
uv run geh run --config my-run.yaml
```

Predictions stream to disk per sample. Re-run the same config and it resumes where it stopped.

```
uv run geh list profiles     # 14 known-good guards (Llama Guard, ShieldGemma, …)
uv run geh list datasets     # 80+ safety benchmarks
uv run geh list backends     # hf_generate · openai_compat · hf_text_classifier · hf_vlm · mock
uv run geh list guards       # llama_guard · llm · md_judge · qwen3guard · …
```

## Pick a profile or roll your own

A profile is a complete `model:` block (guard + backend + args). Drop one in by slug:

```yaml
model:
  profile: granite-guardian-3.2-5b
```

Override anything you want — deep-merged onto the profile:

```yaml
model:
  profile: llama-guard-3-8b
  backend:
    kind: openai_compat              # swap local HF -> vLLM
    args:
      base_url: http://localhost:8000/v1
      api_key_env: null
```

Or build the full `model:` block from scratch — see [`examples/`](examples/).

## Examples

| File | What it shows |
|---|---|
| [`mock-jsonl.yaml`](examples/mock-jsonl.yaml) | Zero-dep smoke test (mock backend on local JSONL) |
| [`llama-guard-vllm.yaml`](examples/llama-guard-vllm.yaml) | Llama Guard 3 served over vLLM |
| [`openai-judge.yaml`](examples/openai-judge.yaml) | GPT as LLM-as-judge with a custom policy |
| [`profile-multi-dataset.yaml`](examples/profile-multi-dataset.yaml) | One profile across several datasets |
| [`hf-text-classifier.yaml`](examples/hf-text-classifier.yaml) | Prompt-Guard-86M HF classifier |

## Run artifacts

```
out/<run-name>/
  manifest.json              # run metadata + config hash
  resolved-config.json       # exact config snapshot
  summary.json               # per-dataset aggregated metrics
  datasets/
    <dataset>/
      predictions.jsonl      # one record per sample (raw output + parsed label)
      metrics.json
      dataset-manifest.json
```

```bash
uv run geh inspect --run-dir out/<run-name>
```

## Documentation

- [Quickstart](docs/getting-started/quickstart.md) — first run in 2 minutes
- [Configuration reference](docs/user-guide/configuration.md) — full YAML schema
- [Guard catalog](docs/guards/overview.md) — built-in guards & profiles
- [Backend catalog](docs/backends/overview.md) — inference engines
- [Architecture](docs/developer/architecture.md) — guard × backend split

## License

[MIT](LICENSE)
