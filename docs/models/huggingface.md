# HuggingFace Adapter

The `hf` adapter runs safety models locally using HuggingFace Transformers. It supports Llama Guard, Granite Guardian, ShieldGemma, and any text-classification or text-generation model on the Hub.

## Requirements

```bash
pip install -e ".[hf]"
```

## Quick Start

```bash
geh run --dataset xstest --model hf \
    --model-name meta-llama/Llama-Guard-3-8B \
    --batch-size 16
```

## Configuration

```yaml
model:
  adapter: hf
  model_name: meta-llama/Llama-Guard-3-8B
  args:
    apply_chat_template: true       # Use model's chat template
    drop_failed_predictions: true   # Skip samples that fail inference
    task: text-classification       # Pipeline task (auto-detected if omitted)
    label_score_aggregation: max    # How to aggregate multi-label scores
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_name` | `str` | **required** | HuggingFace model ID or local path |
| `apply_chat_template` | `bool` | `false` | Apply the model's chat template to format inputs |
| `add_generation_prompt` | `bool` | `false` | Add generation prompt after chat template |
| `drop_failed_predictions` | `bool` | `false` | Skip failed samples instead of raising errors |
| `task` | `str` | auto | Pipeline task type (`text-classification`, etc.) |
| `label_score_aggregation` | `str` | `"max"` | Aggregation for multi-label scores |
| `pretrained` | `dict` | `{}` | Extra kwargs for `from_pretrained()` |

## Supported Models

The adapter includes specialized handling for:

### Llama Guard

```yaml
model:
  adapter: hf
  model_name: meta-llama/Llama-Guard-3-8B
  args:
    apply_chat_template: true
    add_generation_prompt: true
```

!!! note
    Llama Guard models require `HF_TOKEN` for gated access. Set it in your `.env` file.

### Granite Guardian

```yaml
model:
  adapter: hf
  model_name: ibm-granite/granite-guardian-3.1-8b
  args:
    apply_chat_template: true
```

### Text Classification Models

```yaml
model:
  adapter: hf
  model_name: unitary/toxic-bert
  args:
    task: text-classification
    label_score_aggregation: max
```

## Batch Size

For local GPU inference, batch size significantly affects throughput and memory:

```yaml
execution:
  batch_size: 16        # Fixed batch size
  # or
  batch_size: auto      # Adaptive — backs off on OOM
```

!!! tip
    Use `batch_size: auto` if you're unsure about GPU memory limits. The harness will start with the configured size and reduce it on OOM errors.

## Capabilities

| Capability | Supported |
|-----------|:---------:|
| Probability scores | Yes |
| Batching | Yes |
| Concurrency | No |
| Category outputs | Model-dependent |
| Input modalities | Text |

## Vision-Language Models

For image+text models, use the specialized adapters:

- **`hf_vlm_guard`** — LlavaGuard and similar VLM guard models
- **`hf_image_classifier`** — Image classification pipelines
- **`hf_shieldgemma2`** — ShieldGemma2 multimodal
- **`hf_safeqwen_vlm`** — SafeQWen VLM

See [Image Benchmarks](../datasets/image.md) for dataset pairing.
