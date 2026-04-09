# Common Workflows

These recipes use the configs and commands already present in the repository,
so they are good starting points for repeatable runs.

## 1. Smoke Test The Harness

```bash
pip install -e "."
geh run --dataset xstest --model mock --limit 50
```

Use this first on a fresh machine or CI job.

## 2. Run A Local HuggingFace Model Against Local JSONL

Config file:

```text
examples/run-hf-mock-jsonl.yaml
```

Run it with:

```bash
pip install -e ".[hf]"
geh run --config examples/run-hf-mock-jsonl.yaml
```

This is a good bridge between the simplest `mock` setup and a real local
backend.

## 3. Run OpenAI Moderation On An Image-Safety Dataset

Config file:

```text
examples/openai-moderation-safe-vs-unsafe-image-edits.yaml
```

Run it with:

```bash
pip install -e ".[api]"
export OPENAI_API_KEY=sk-...
geh run --config examples/openai-moderation-safe-vs-unsafe-image-edits.yaml
```

Use this when you want a hosted multimodal moderation baseline quickly.

## 4. Run A Local Image JSONL Workflow

Hosted OpenAI-compatible path:

```bash
export LOCAL_IMAGE_JSONL_PATH=/abs/path/to/images.jsonl
export OPENAI_API_KEY=sk-...
export OPENAI_VISION_MODEL=gpt-4.1-mini
geh run --config examples/openai-compatible-local-image-jsonl.yaml
```

Local HuggingFace VLM path:

```bash
export LOCAL_IMAGE_JSONL_PATH=/abs/path/to/images.jsonl
geh run --config examples/llavaguard-local-image-jsonl.yaml
```

## 5. Run A Curated Pack Instead Of Picking Datasets Manually

```bash
geh list packs
geh run --pack core --model mock
geh run --pack jailbreak --model hf --model-name meta-llama/Llama-Guard-3-8B
```

Choose this when the question is "how does this model do on a standard starter
suite?" rather than "how does it do on one dataset?"

## 6. Run Code Vulnerability Evaluation

Config file:

```text
examples/code-vuln/run-vulnllm-r-openai.yaml
```

Run it with:

```bash
pip install -e ".[api]"
export OPENAI_API_KEY=sk-...
geh run --config examples/code-vuln/run-vulnllm-r-openai.yaml
```

This is the cleanest starting point for repository-level and function-level
code security benchmarks.

## 7. Validate A Config Before Spending GPU Or API Time

```bash
geh validate --config examples/run-mock-jsonl.yaml
```

This is especially useful when you are working with local paths or
adapter-specific args.
