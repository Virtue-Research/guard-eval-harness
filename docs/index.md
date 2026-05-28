---
hide:
  - navigation
---

# Guard Eval Harness

<p style="font-size: 1.2em; color: var(--md-default-fg-color--light);">
Lightweight harness for evaluating safety guards against canonical benchmarks.
</p>

<p>
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-80+-5b6cff?style=flat-square">
  <img alt="Modalities" src="https://img.shields.io/badge/modalities-text%20%7C%20image-5b6cff?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-5b6cff?style=flat-square">
</p>

Wire any chat LLM (OpenAI, Anthropic, local HF, vLLM) — or any guard model
(Llama Guard, ShieldGemma, LlavaGuard, an HF image classifier) — to **80+
built-in safety benchmarks** with one YAML.

---

## Mental model

A run is exactly four things:

```
       Guard               Backend
   (prompt + parser)   (inference engine)
        │                    │
        ▼                    ▼
sample ──► messages ──► raw output ──► ParsedLabel ──► metrics
                                            ▲
                                            │
                                       Policy + OutputFormat
                                       (LLM-as-guard only)
```

- **Guard** owns the prompt construction + output parser. Fixed-taxonomy
  guards (Llama Guard, ShieldGemma) embed their taxonomy. The generic
  `llm` guard accepts any **Policy** + **OutputFormat** combo.
- **Backend** is the inference engine (`hf_generate`, `openai_compat`,
  `hf_vlm`, `hf_image_classifier`, `mock`, …). It knows nothing about
  safety taxonomies.
- **Dataset** loads samples in a normalized shape.
- **Runner** glues them together, streams predictions to disk, and
  computes metrics — resumable by default.

## Quick example

```yaml title="examples/llama-guard-vllm.yaml"
run_name: llama-guard-on-xstest

model:
  profile: llama-guard-3-8b
  backend:
    kind: openai_compat
    name: meta-llama/Llama-Guard-3-8B
    args:
      base_url: http://localhost:8000/v1
      api_key_env: null

datasets:
  - name: xstest
    n_samples: 100

output:
  run_dir: out/llama-guard-on-xstest
```

```bash
geh run --config examples/llama-guard-vllm.yaml
```

Streams one record per sample to
`out/llama-guard-3-on-xstest/datasets/xstest/predictions.jsonl`,
writes `metrics.json` + `manifest.json`, and resumes cleanly if
interrupted.

## Key features

<div class="grid cards" markdown>

- :material-package-variant: **Guard × Backend split**

    Plug any guard prompt into any inference engine. Llama Guard runs
    the same way on local HF, vLLM, or any OpenAI-compatible server.

- :material-database: **80+ datasets**

    Text and image safety benchmarks (XSTest, HarmBench, ToxicChat,
    BeaverTails, WildGuardMix, VLSBench, MM-SafetyBench, …) with a
    single YAML.

- :material-restore: **Resumable runs**

    Predictions stream to disk per-sample. Re-run the same config and
    it picks up where it stopped — config-hash guard prevents
    silent mixing.

- :material-chart-bar: **Per-sample outputs**

    Every run writes one JSONL row per sample with raw output, parsed
    label, ground truth, and latency. Metrics computed from disk, not
    in-memory.

- :material-source-branch: **30-line custom guards**

    Adding a new safety model is a `Guard` subclass with
    `build_messages()` + `parse()`. No catalog YAMLs, no template
    profiles.

- :material-shield-lock: **Ground-truth leak proof**

    `PredictSample` structurally excludes labels. A denylist rejects
    label-shaped metadata at the boundary.

</div>

## What's next?

- **[Installation](getting-started/installation.md)** — one-line install
- **[Quickstart](getting-started/quickstart.md)** — first evaluation in 2 minutes
- **[Configuration](user-guide/configuration.md)** — full YAML reference
- **[Guards](guards/overview.md)** — built-in guard catalog
- **[Backends](backends/overview.md)** — supported inference engines
- **[Datasets](datasets/overview.md)** — 80+ built-in benchmarks
