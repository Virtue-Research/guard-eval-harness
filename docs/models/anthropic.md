# Anthropic Adapter

The `anthropic` adapter evaluates safety using the Anthropic Messages API with Claude models.

## Requirements

```bash
pip install -e ".[api]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```bash
geh run --dataset xstest --model anthropic \
    --model-name claude-sonnet-4-20250514
```

## Configuration

```yaml
model:
  adapter: anthropic
  model_name: claude-sonnet-4-20250514
  args:
    api_key_env: ANTHROPIC_API_KEY
    concurrency: 4
    max_tokens: 16
    temperature: 0.0
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `str` | **required** | Claude model name |
| `api_key_env` | `str` | `"ANTHROPIC_API_KEY"` | Env var for API key |
| `concurrency` | `int` | `1` | Parallel API requests |
| `max_tokens` | `int` | `16` | Maximum response tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |

## Multimodal Support

The Anthropic adapter supports image inputs for vision-capable models:

```yaml
model:
  adapter: anthropic
  model_name: claude-sonnet-4-20250514
  args:
    api_key_env: ANTHROPIC_API_KEY
    concurrency: 4

datasets:
  - name: unsafebench
    adapter: unsafebench
```

## Token Accounting

The adapter reports token usage in prediction metadata, useful for cost tracking:

```json
{
  "metadata": {
    "input_tokens": 142,
    "output_tokens": 3
  }
}
```

## Capabilities

| Capability | Supported |
|-----------|:---------:|
| Probability scores | No (text-based) |
| Batching | No |
| Concurrency | Yes |
| Token accounting | Yes |
| Category outputs | No |
| Input modalities | Text, Image |
