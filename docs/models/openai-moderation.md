# OpenAI Moderation Adapter

The `openai_moderation` adapter uses the [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) for content safety classification. It supports text and image inputs with per-category scores.

## Requirements

```bash
pip install -e ".[api]"
export OPENAI_API_KEY=sk-...
```

## Quick Start

```bash
geh run --dataset xstest,toxic_chat --model openai_moderation
```

## Configuration

```yaml
model:
  adapter: openai_moderation
  model_name: omni-moderation-latest
  args:
    api_key_env: OPENAI_API_KEY
    concurrency: 8
    categories:             # Filter to specific categories
      - sexual
      - violence
      - violence/graphic
      - self-harm
      - self-harm/intent
      - self-harm/instructions
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_name` | `str` | `"omni-moderation-latest"` | Moderation model to use |
| `api_key_env` | `str` | `"OPENAI_API_KEY"` | Environment variable containing the API key |
| `url` | `str` | OpenAI default | Custom moderation endpoint URL |
| `concurrency` | `int` | `1` | Parallel API requests |
| `categories` | `list[str]` | all | Filter to specific safety categories |

## Category Scores

The OpenAI Moderation API returns per-category confidence scores. The adapter maps these to `NormalizedPrediction` with:

- `unsafe_score` — Maximum score across all (filtered) categories
- `category_scores` — Per-category breakdown
- `predicted_categories` — Categories exceeding the threshold

## Multimodal Support

The `omni-moderation-latest` model supports image inputs:

```yaml title="examples/openai-moderation-safe-vs-unsafe-image-edits.yaml"
version: 1
run_name: openai-moderation-images
threshold: 0.5

model:
  adapter: openai_moderation
  model_name: omni-moderation-latest
  args:
    api_key_env: OPENAI_API_KEY

datasets:
  - name: safe_vs_unsafe_image_edits
    adapter: safe_vs_unsafe_image_edits
    split: train

execution:
  concurrency: 8
```

## Capabilities

| Capability | Supported |
|-----------|:---------:|
| Probability scores | Yes |
| Batching | No |
| Concurrency | Yes |
| Cost estimation | No |
| Category outputs | Yes |
| Input modalities | Text, Image |
