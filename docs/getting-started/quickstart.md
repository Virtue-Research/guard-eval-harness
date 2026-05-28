# Quickstart

Go from a fresh checkout to a finished evaluation run in two minutes.

## 1. Install

```bash
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness
uv sync --extra hf            # base + local HuggingFace inference
```

`uv sync` creates `.venv/` and installs the locked deps. Use `pip install -e ".[hf]"` if you prefer pip.

## 2. Smoke test — no GPU, no keys

```bash
uv run geh run --config examples/mock-jsonl.yaml
```

This uses the bundled `mock` backend on a tiny local JSONL. It exists to confirm the pipeline runs and to show the artifact layout:

```
out/mock-demo/
  manifest.json          # run metadata + config hash (resume-keyed)
  resolved-config.json   # exact config snapshot
  summary.json           # aggregated metrics
  datasets/<name>/
    predictions.jsonl    # one row per sample
    metrics.json
    dataset-manifest.json
```

Inspect with:

```bash
uv run geh inspect --run-dir out/mock-demo
```

## 3. Discover what's available

```bash
uv run geh list profiles     # 14 known-good guards (Llama Guard, ShieldGemma, …)
uv run geh list datasets     # 80+ safety benchmarks
uv run geh list backends     # hf_generate · openai_compat · hf_text_classifier · hf_vlm · mock
uv run geh list guards       # llama_guard · llm · md_judge · qwen3guard · …
```

## 4. Real run via a bundled profile

Profiles bundle a guard + backend + sensible args under a single slug. Reference one by name:

```yaml title="run.yaml"
version: 2
run_name: granite-on-three

model:
  profile: granite-guardian-3.2-5b

datasets:
  - name: xstest
    limit: 100
  - name: toxic_chat
    limit: 100
  - name: harmbench_behaviors
    limit: 100

output:
  run_dir: out/granite-on-three
  resume: true
```

```bash
uv run geh validate --config run.yaml      # optional: load + dry-run
uv run geh run --config run.yaml
```

## 5. Swap the backend

Profiles are deep-merged with anything you put in the config. To run Llama Guard via vLLM instead of local HF:

```yaml
model:
  profile: llama-guard-3-8b
  backend:
    kind: openai_compat
    name: meta-llama/Llama-Guard-3-8B
    args:
      base_url: http://localhost:8000/v1
      api_key_env: null
```

## 6. Resume after a crash

Re-run the same config. The harness checks `manifest.json` against a SHA-256 hash of the resolved config and continues from the last completed sample.

```bash
uv run geh run --config run.yaml               # resumes automatically
uv run geh run --config run.yaml --overwrite   # wipe the run dir and start fresh
```

## What to do next

- **[Configuration](../user-guide/configuration.md)** — full YAML schema
- **[Guards](../guards/overview.md)** — bundled guards & profiles
- **[Backends](../backends/overview.md)** — supported inference engines
- **[Common workflows](../user-guide/common-workflows.md)** — copy-paste recipes
- **[Adding a guard](../developer/adding-guards.md)** — ~30 lines for a new safety model
