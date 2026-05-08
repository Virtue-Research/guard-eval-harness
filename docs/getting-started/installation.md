# Installation

Use the base install when you only need the CLI, configs, dataset
normalization, and core metrics. Add extras only for the backends and features
you plan to use.

## Requirements

- Python `3.10+`
- `pip` or another PEP 517-compatible installer
- a working shell environment for API keys if you use hosted models

## Base Install

```bash
git clone https://github.com/Virtue-Research/guard-eval-harness.git
cd guard-eval-harness
pip install -e "."
```

The base install is enough for:

- the `geh` CLI
- config validation
- built-in dataset loading
- the `mock` model adapter
- artifact inspection and comparison flows

## Optional Extras

=== "HuggingFace"

    ```bash
    pip install -e ".[hf]"
    ```

    Use this for local text, image, and specialized multimodal HuggingFace adapters.

=== "vLLM"

    ```bash
    pip install -e ".[vllm]"
    ```

    Use this for high-throughput local inference with the `vllm` adapter.

=== "API"

    ```bash
    pip install -e ".[api]"
    ```

    Use this for `openai_moderation`, `openai_compatible`, `anthropic`, and `http`.

=== "Audio"

    ```bash
    pip install -e ".[audio]"
    ```

    Add this when working with native audio datasets or adapters.

=== "Reports"

    ```bash
    pip install -e ".[report]"
    ```

    Useful when you want HTML or spreadsheet-friendly reporting dependencies available explicitly.

=== "Everything"

    ```bash
    pip install -e ".[hf,vllm,api,audio,report,dev]"
    ```

## Environment Setup

Copy the example environment file if you plan to use hosted models or gated
HuggingFace assets:

```bash
cp .env.example .env
```

```bash title=".env"
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

You only need to set the variables required by the adapters you actually run.

## Verify The Install

```bash
geh list backends
geh list packs
geh run --dataset xstest --model mock --limit 10
```

You should see JSON output with the created `run_dir`, `manifest_path`, and
`summary_path`.

## Next Steps

- [Quickstart](quickstart.md) for a 2-minute first run
- [Run Modes](run-modes.md) to choose inline, pack, or YAML config flows
- [Troubleshooting](troubleshooting.md) if installs, auth, or dataset downloads fail
