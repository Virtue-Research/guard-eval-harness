# vLLM Adapter

The `vllm` adapter provides high-throughput local inference using the [vLLM](https://docs.vllm.ai/) engine. Ideal for evaluating large batches on GPU.

## Requirements

```bash
pip install -e ".[vllm]"
```

Requires vLLM 0.8.0–0.17.x and a CUDA-capable GPU.

## Quick Start

```bash
geh run --dataset xstest --model vllm \
    --model-name meta-llama/Llama-Guard-3-8B \
    --batch-size 512
```

## Configuration

```yaml title="examples/run-vllm-llama-guard.yaml"
version: 1
run_name: vllm-llama-guard
threshold: 0.5

model:
  adapter: vllm
  model_name: meta-llama/Llama-Guard-3-8B
  args:
    apply_chat_template: true
    add_generation_prompt: true
    max_new_tokens: 16
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    text_score_mapping:
      safe: 0.0
      unsafe: 1.0

datasets:
  - name: xstest
    adapter: xstest

execution:
  batch_size: 512

output:
  run_dir: out/vllm-llama-guard
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_name` | `str` | **required** | HuggingFace model ID |
| `max_new_tokens` | `int` | `16` | Maximum tokens to generate |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory to use |
| `apply_chat_template` | `bool` | `false` | Use model's chat template |
| `add_generation_prompt` | `bool` | `false` | Add generation prompt |
| `text_score_mapping` | `dict` | `{}` | Map text outputs to numeric scores |

### Text Score Mapping

Most safety models output text like `"safe"` or `"unsafe"`. The `text_score_mapping` converts these to numeric scores:

```yaml
args:
  text_score_mapping:
    safe: 0.0
    unsafe: 1.0
```

## Batch Size

vLLM handles batching internally, so you can use very large batch sizes:

```yaml
execution:
  batch_size: 512    # vLLM manages GPU memory efficiently
```

The adapter uses auto-batching with capacity backoff to handle memory pressure.

## Multi-GPU

For models that don't fit on a single GPU:

```yaml
model:
  adapter: vllm
  model_name: meta-llama/Llama-Guard-3-8B
  args:
    tensor_parallel_size: 2    # Shard across 2 GPUs
```

## Capabilities

| Capability | Supported |
|-----------|:---------:|
| Probability scores | Yes |
| Batching | Yes |
| Concurrency | No |
| Category outputs | No |
| Input modalities | Text |
