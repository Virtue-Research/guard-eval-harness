# Installation

Recommended path: use [uv](https://docs.astral.sh/uv/) to manage a
project-local virtualenv. `pip` works too if you prefer.

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (or `pip` if you don't have uv)
- A shell environment for API keys (only if you use hosted models)

## Install — uv (recommended)

```bash
# clone
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness

# base install (mock backend + dataset normalization + CLI)
uv sync

# add extras you actually need
uv sync --extra dev                 # tests + lint
uv sync --extra hf                  # local HuggingFace inference
uv sync --extra api                 # tenacity retry for hosted endpoints
uv sync --extra hf --extra dev      # combine extras

# run anything inside the venv automatically
uv run geh list guards
uv run pytest
```

`uv sync` creates `.venv/` in the project root and installs the
locked dependency set from `uv.lock`. `uv run <cmd>` executes
`<cmd>` inside that venv without needing `source .venv/bin/activate`.

## Install — pip

```bash
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness

python -m venv .venv
source .venv/bin/activate
pip install -e .                     # base
pip install -e ".[hf]"               # add local HF backend
pip install -e ".[dev]"              # add tests + lint
```

## Install from git (no clone)

If you only want to *use* `geh` and not develop on it, install
straight from GitHub:

```bash
pip install "git+https://github.com/Virtue-Research/guard-eval-harness.git"

# with extras
pip install "guard-eval-harness[hf] @ git+https://github.com/Virtue-Research/guard-eval-harness.git"
```

Or install as a standalone CLI tool (isolated venv, available on
`$PATH`):

```bash
uv tool install "guard-eval-harness @ git+https://github.com/Virtue-Research/guard-eval-harness.git"
# or
pipx install "git+https://github.com/Virtue-Research/guard-eval-harness.git"
```

## Optional Extras

| Extra | What it pulls in | When you need it |
|---|---|---|
| `hf` | `torch`, `transformers`, `accelerate`, `Pillow`, `sentencepiece` | Local HuggingFace guards (Llama Guard, ShieldGemma, image classifier, VLM) |
| `api` | `tenacity` | Robust retry for hosted OpenAI / vLLM-server / LiteLLM endpoints |
| `dev` | `pytest`, `ruff`, `build` | Running tests + lint, building wheels |

## Environment Setup

For hosted models you'll typically need API keys:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
```

You only need the variables required by the backends you actually run.

## Verify The Install

```bash
uv run geh list guards     # llm, llama_guard, shieldgemma, ...
uv run geh list backends   # mock, hf_generate, hf_vlm, openai_compat, ...
uv run geh run --config examples/mock-jsonl.yaml
```

You should see JSON output with the created `run_dir`,
`manifest_path`, and `summary_path`.

## Next Steps

- [Quickstart](quickstart.md) for a 2-minute first run
- [Configuration](../user-guide/configuration.md) for the full YAML reference
- [Troubleshooting](troubleshooting.md) if installs, auth, or dataset downloads fail
