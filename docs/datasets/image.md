# Image Benchmarks

Evaluate multimodal safety models that process image+text inputs. The harness handles image downloading, caching, and normalization automatically.

## Available Benchmarks

| Dataset | Adapter | Description | Images |
|---------|---------|-------------|:------:|
| UnsafeBench | `unsafebench` | Safety categories across 8k+ images | 8,000+ |
| HoliSafeBench | `holisafe_bench` | Holistic safety with fine-grained risk types | - |
| JailbreakV 28k | `jailbreakv_28k` | Adversarial images bypassing VLM safeguards | 28,000 |
| Safe vs Unsafe Edits | `safe_vs_unsafe_image_edits` | Harmful image manipulation detection | - |
| VLSBench | `vlsbench` | Vision-language safety | - |
| MSTS | `msts` | Multimodal safety evaluation | - |
| ImageNet-1k Safe | `imagenet1k_val_safe` | Benign baseline for false-positive calibration | 50,000 |

## Compatible Model Adapters

| Adapter | Models |
|---------|--------|
| `openai_moderation` | `omni-moderation-latest` |
| `openai_compatible` | GPT-4o, GPT-4.1, etc. |
| `anthropic` | Claude Sonnet, Claude Opus |
| `hf_vlm_guard` | LlavaGuard |
| `hf_shieldgemma2` | ShieldGemma2 |
| `hf_image_classifier` | Image classification pipelines |

## Image Caching

Images are downloaded once and cached locally using content-addressed storage (SHA256). The cache is shared across runs.

```bash
# Check cache status
geh cache status

# Clear cache for a specific dataset
geh cache clear --dataset unsafebench

# Clear all cached images
geh cache clear
```

!!! tip
    Use `--no-sample-cache` to disable image caching for a run if you want fresh downloads.

## Usage Examples

### OpenAI Moderation on Image Datasets

```yaml title="examples/openai-moderation-safe-vs-unsafe-image-edits.yaml"
version: 1
run_name: openai-moderation-images
threshold: 0.5

model:
  adapter: openai_moderation
  model_name: omni-moderation-latest
  args:
    api_key_env: OPENAI_API_KEY
    categories:
      - sexual
      - violence
      - self-harm

datasets:
  - name: safe_vs_unsafe_image_edits
    adapter: safe_vs_unsafe_image_edits
    split: train
    options:
      variant: batch1

execution:
  concurrency: 8
```

### ShieldGemma2 on Local Images

```yaml title="examples/shieldgemma2-local-image-dir.yaml"
version: 1
run_name: shieldgemma2-local-images

model:
  adapter: hf_shieldgemma2
  model_name: google/shieldgemma-2-4b-it
  args:
    device_map: auto
    torch_dtype: bfloat16
    policies:
      - sexual
      - dangerous
      - violence
    emit_categories: true

datasets:
  - name: local_image_dir
    adapter: local_image_dir
    path: ${LOCAL_IMAGE_DIR_ROOT}

execution:
  batch_size: 8
```

### LlavaGuard on Image JSONL

```yaml title="examples/llavaguard-local-image-jsonl.yaml"
version: 1
run_name: llavaguard-images

model:
  adapter: hf_vlm_guard
  model_name: AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf
  args:
    flow: llavaguard
    device_map: auto
    torch_dtype: bfloat16
    max_new_tokens: 200
    emit_categories: true

datasets:
  - name: local_image_jsonl
    adapter: local_image_jsonl
    path: ${LOCAL_IMAGE_JSONL_PATH}

execution:
  batch_size: 2
```
