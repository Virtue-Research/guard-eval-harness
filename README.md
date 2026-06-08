<p align="center">
  <img src="assets/polished.svg" alt="guard-eval-harness" width="550">
</p>

<p align="center">CLI-first harness for benchmarking guardrail, moderation, and safety classification models.</p>

<p align="center">
  <img src="assets/demo.gif" alt="geh demo — run a benchmark pack and export the results as a table" width="900">
</p>

<p align="center">
  <a href="https://pypi.org/project/geh/"><img alt="PyPI" src="https://img.shields.io/pypi/v/geh.svg?style=flat-square&color=4c1&logo=pypi&logoColor=white&logoSize=auto&label=pypi"></a>
  <a href="https://pypi.org/project/geh/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-2d3748?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU1Ij48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImEiIHgxPSIxMi45NTklIiB4Mj0iNzkuNjM5JSIgeTE9IjEyLjAzOSUiIHkyPSI3OC4yMDElIj48c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiMzODdFQjgiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMzNjY5OTQiLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDE9IjE5LjEyOCUiIHgyPSI5MC43NDIlIiB5MT0iMjAuNTc5JSIgeTI9Ijg4LjQyOSUiPjxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI0ZGRTA1MiIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0ZGQzMzMSIvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxwYXRoIGZpbGw9InVybCgjYSkiIGQ9Ik0xMjYuOTE2LjA3MmMtNjQuODMyIDAtNjAuNzg0IDI4LjExNS02MC43ODQgMjguMTE1bC4wNzIgMjkuMTI4aDYxLjg2OHY4Ljc0NUg0MS42MzFTLjE0NSA2MS4zNTUuMTQ1IDEyNi43N2MwIDY1LjQxNyAzNi4yMSA2My4wOTcgMzYuMjEgNjMuMDk3aDIxLjYxdi0zMC4zNTZzLTEuMTY1LTM2LjIxIDM1LjYzMi0zNi4yMWg2MS4zNjJzMzQuNDc1LjU1NyAzNC40NzUtMzMuMzE5VjMzLjk3UzE5NC42Ny4wNzIgMTI2LjkxNi4wNzJ6TTkyLjgwMiAxOS42NmExMS4xMiAxMS4xMiAwIDAgMSAxMS4xMyAxMS4xMyAxMS4xMiAxMS4xMiAwIDAgMS0xMS4xMyAxMS4xMyAxMS4xMiAxMS4xMiAwIDAgMS0xMS4xMy0xMS4xMyAxMS4xMiAxMS4xMiAwIDAgMSAxMS4xMy0xMS4xM3oiLz48cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMTI4Ljc1NyAyNTQuMTI2YzY0LjgzMiAwIDYwLjc4NC0yOC4xMTUgNjAuNzg0LTI4LjExNWwtLjA3Mi0yOS4xMjdIMTI3LjZ2LTguNzQ1aDg2LjQ0MXM0MS40ODYgNC43MDUgNDEuNDg2LTYwLjcxMmMwLTY1LjQxNi0zNi4yMS02My4wOTYtMzYuMjEtNjMuMDk2aC0yMS42MXYzMC4zNTVzMS4xNjUgMzYuMjEtMzUuNjMyIDM2LjIxaC02MS4zNjJzLTM0LjQ3NS0uNTU3LTM0LjQ3NSAzMy4zMnY1Ni4wMTNzLTUuMjM1IDMzLjg5NyA2Mi41MTggMzMuODk3em0zNC4xMTQtMTkuNTg2YTExLjEyIDExLjEyIDAgMCAxLTExLjEzLTExLjEzIDExLjEyIDExLjEyIDAgMCAxIDExLjEzLTExLjEzMSAxMS4xMiAxMS4xMiAwIDAgMSAxMS4xMyAxMS4xMyAxMS4xMiAxMS4xMiAwIDAgMS0xMS4xMyAxMS4xM3oiLz48L3N2Zz4K&logoSize=auto"></a>
  <a href="https://github.com/Virtue-Research/guard-eval-harness/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-3DA639?style=flat-square&logo=opensourceinitiative&logoColor=white&logoSize=auto"></a>
  <a href="https://github.com/Virtue-Research/guard-eval-harness/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Virtue-Research/guard-eval-harness/ci.yml?branch=main&label=CI&style=flat-square&logo=githubactions&logoColor=white&logoSize=auto"></a>
  <a href="https://virtue-research.github.io/guard-eval-harness/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-526CFE?style=flat-square&logo=materialformkdocs&logoColor=white&logoSize=auto"></a>
  <a href="https://pepy.tech/projects/geh"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/geh?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
  <img alt="Made with love" src="https://img.shields.io/badge/made%20with-%E2%9D%A4-FF1493?style=flat-square">
</p>

Evaluate any safety model — local HuggingFace, vLLM, OpenAI, Anthropic, or custom API — against 80+ built-in safety benchmarks with a single command.

## Quickstart

```bash
pip install geh

# Run a quick eval
geh run --dataset xstest --model mock --limit 50

# Run multiple datasets
geh run --dataset xstest,toxic_chat,harmful_qa --model hf \
    --model-name meta-llama/Llama-Guard-3-8B

# Run from a YAML config
geh run --config examples/run-mock-jsonl.yaml

# Use benchmark packs
geh run --pack core --model mock
```

## Installation

Requires Python 3.10+.

```bash
# Base install
pip install geh

# With HuggingFace model support
pip install "geh[hf]"

# With vLLM support
pip install "geh[vllm]"

# With API model support (OpenAI, Anthropic)
pip install "geh[api]"
```

From source (for development):

```bash
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in the API keys you need.

## Usage

### Inline mode

The fastest way to run evals — no config files needed:

```bash
geh run --dataset <dataset> --model <adapter> [--model-name <name>] [options]
```

```bash
# HuggingFace model on XSTest
geh run --dataset xstest --model hf --model-name meta-llama/Llama-Guard-3-8B

# OpenAI moderation
geh run --dataset xstest,toxic_chat --model openai_moderation

# vLLM serving
geh run --dataset harmbench_behaviors --model vllm \
    --model-name meta-llama/Llama-Guard-3-8B --batch-size 32

# Limit samples for quick smoke tests
geh run --dataset xstest --model mock --limit 10
```

### YAML config mode

For full control over model args, dataset options, execution tuning, and output:

```bash
geh run --config examples/run-mock-jsonl.yaml
```

See [`examples/`](examples/) for sample configs.

### Benchmark packs

Curated dataset bundles for common evaluation scenarios:

```bash
geh list packs
geh run --pack core --model mock
geh run --pack jailbreak --model hf --model-name meta-llama/Llama-Guard-3-8B
```

### Discovery

```bash
geh list datasets    # 80+ built-in safety benchmarks
geh list backends    # Available model adapters
geh list packs       # Curated benchmark bundles
geh list metrics     # Supported metrics
```

### Inspecting results

```bash
geh inspect --run-dir out/my-run       # View manifest, summary, artifacts
geh report --run-dir out/my-run        # Rebuild HTML report
geh compare --run-a out/run1 --run-b out/run2  # Diff two runs
geh export --run-dir out/my-run --format csv --output results.csv
```

## Run artifacts

Each run writes a self-contained directory:

```
out/my-run/
  manifest.json              # Run metadata
  resolved-config.json       # Exact config snapshot
  summary.json               # Aggregated metrics
  report.html                # Static HTML report
  datasets/
    <dataset>/
      predictions.jsonl      # Per-sample predictions
      metrics.json           # Dataset-level metrics
      dataset-manifest.json  # Dataset metadata
```

## Model adapters

| Adapter | Description |
|---------|-------------|
| `mock` | Deterministic mock for testing |
| `hf` | HuggingFace Transformers (local GPU) |
| `vllm` | vLLM inference server |
| `openai_compatible` | OpenAI-compatible APIs |
| `openai_moderation` | OpenAI Moderation endpoint |
| `anthropic` | Anthropic Claude API |
| `http` | Generic HTTP endpoint |

## Datasets

80+ built-in safety benchmarks spanning two modalities:

### Text

The core modality — evaluate text-based guardrails and moderation models across a range of safety dimensions:

- **Jailbreak / adversarial**: XSTest, HarmBench, JBB Behaviors, AdvBench, Do-Anything-Now, StrongREJECT, MaliciousInstruct, WildGuardMix
- **Toxicity**: ToxicChat, ToxiGen, Jigsaw Toxicity, Civil Comments, RealToxicityPrompts, OR-Bench
- **Hate & harassment**: HateCheck, DynaHate, ETHOS, HatExplain, Implicit Hate, Measuring Hate Speech, Social Bias Frames, ConvAbuse
- **General safety**: BeaverTails 330k, Do-Not-Answer, OpenAI Moderation (via API), GuardBench, CircleGuardBench
- **Prompt injection**: Dedicated prompt-injection benchmarks for testing input-filtering guardrails

### Image

Evaluate multimodal safety models that process image+text inputs. The harness handles image downloading, caching, and normalization automatically:

- **Unsafe content detection**: UnsafeBench (8k+ images across safety categories), HoliSafeBench (holistic image safety with fine-grained risk types)
- **Visual jailbreaks**: JailbreakV (adversarial images designed to bypass vision-language model safeguards)
- **Image edit safety**: Safe-vs-Unsafe Image Edits (detecting harmful image manipulation requests)
- **Cross-modal attacks**: VLSBench, MSTS (text+image multimodal safety evaluation)
- **Benign baselines**: ImageNet-1k safe subset (measuring false positive rates on benign images)
- **Local image data**: Load from local directories or JSONL manifests with image paths/URLs

### Local files

Bring your own data in any modality:

- `local_jsonl` — text samples from a JSONL file
- `local_csv` — text samples from a CSV file
- `local_image_jsonl` — image+text samples from a JSONL manifest with image paths/URLs
- `local_image_dir` — image samples from a directory of images

Run `geh list datasets` for the full list.

## About

`guard-eval-harness` is built and maintained by the research team at
**[Virtue AI](https://www.virtueai.com)** — one security solution for your entire AI stack.

## License

[MIT](LICENSE)
