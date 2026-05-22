---
hide:
  - navigation
---

# Guard Eval Harness

<p style="font-size: 1.25em; color: var(--md-default-fg-color--light);">
CLI-first harness for benchmarking guardrail, moderation, and safety classification models.
</p>

<p>
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-80+-4f46e5?style=flat-square">
  <img alt="Modalities" src="https://img.shields.io/badge/modalities-text%20%7C%20image-4f46e5?style=flat-square">
</p>

Evaluate any safety model — local HuggingFace, vLLM, OpenAI, Anthropic, or custom API — against **80+ built-in safety benchmarks** with a single command.

---

## Key Features

<div class="grid cards" markdown>

- :material-console: **CLI-First**

    Run evaluations from a single command — no notebooks or scripts required.

- :material-database: **80+ Benchmarks**

    Built-in datasets covering text and image safety.

- :material-swap-horizontal: **Any Backend**

    HuggingFace, vLLM, OpenAI, Anthropic, or bring your own HTTP endpoint.

- :material-chart-bar: **Rich Metrics**

    Accuracy, precision, recall, F1, AUROC, AUPRC — computed automatically.

- :material-package-variant: **Benchmark Packs**

    Curated dataset bundles for common evaluation scenarios.

- :material-file-document: **Self-Contained Artifacts**

    Every run produces a portable directory with predictions, metrics, and HTML reports.

</div>

## Quick Example

```bash
# Install
pip install -e ".[hf]"

# Evaluate Llama Guard on XSTest
geh run --dataset xstest --model hf \
    --model-name meta-llama/Llama-Guard-3-8B

# Run a curated benchmark pack
geh run --pack core --model openai_moderation

# Compare two runs
geh compare --run-a out/run1 --run-b out/run2
```

## How It Works

```
geh run --dataset xstest --model hf --model-name meta-llama/Llama-Guard-3-8B
         │                    │                      │
         ▼                    ▼                      ▼
   Load & normalize     Instantiate adapter    Load model weights
   safety samples       (HF, vLLM, API...)     or connect to API
         │                    │                      │
         └────────────────────┼──────────────────────┘
                              ▼
                     Run inference (batched)
                              │
                              ▼
                    Compute binary metrics
                    (accuracy, F1, AUROC...)
                              │
                              ▼
                   Write artifacts to disk
                   (predictions, metrics, HTML report)
```

## What's Next?

- **[Installation](getting-started/installation.md)** — Set up your environment
- **[Quickstart](getting-started/quickstart.md)** — Run your first evaluation in 2 minutes
- **[Run Modes](getting-started/run-modes.md)** — Choose between inline, pack, and YAML flows
- **[Troubleshooting](getting-started/troubleshooting.md)** — Fix install, auth, and path issues
- **[Benchmark Selection](user-guide/benchmark-selection.md)** — Pick the right benchmark path
- **[Configuration](user-guide/configuration.md)** — Full YAML config reference
- **[Models](models/overview.md)** — Connect any safety model backend
- **[Datasets](datasets/overview.md)** — Browse 80+ built-in benchmarks
