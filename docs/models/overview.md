# Model Adapters

Model adapters are the contract between Guard Eval Harness and the system being
evaluated. Every adapter turns backend-specific outputs into the same
`NormalizedPrediction` schema so runs can be compared consistently.

## Where Most Users Should Start

| Adapter | Inputs | Best fit | Start here? |
| --- | --- | --- | --- |
| [`mock`](#mock) | text | smoke tests, docs examples, CI | Yes |
| [`hf`](huggingface.md) | text | local HuggingFace safety models | Yes |
| [`vllm`](vllm.md) | text, image | high-throughput local inference | Yes |
| [`openai_moderation`](openai-moderation.md) | text, image | hosted moderation baseline | Yes |
| [`openai_compatible`](openai-compatible.md) | text, image | hosted or self-hosted OpenAI-style APIs | Yes |
| [`anthropic`](anthropic.md) | text, image | Claude-based classifier flows | Situational |
| [`http`](http.md) | text | custom REST moderation endpoint | Situational |

## Specialized Local Adapters

These adapters are registered publicly, but they are more specialized than the
first-line adapters above. Reach for them when you already know the exact model
family or modality you want.

| Adapter | Inputs | Positioning | Notes |
| --- | --- | --- | --- |
| `hf_vlm_guard` | text, image | advanced | vision-language guard adapters such as LlavaGuard-style models |
| `hf_gemma4_vlm` | text, image | advanced | Gemma 4 VLM safety workflows |
| `hf_safeqwen_vlm` | text, image | advanced | SafeQwen VLM safety-head workflows |
| `hf_image_classifier` | image | specialized | image-only classification pipelines |
| `hf_shieldgemma2` | image | specialized | ShieldGemma2 image moderation |

## Choosing An Adapter

Use this rough decision tree:

```text
Need the fastest first run?
  -> mock

Need a local text model?
  -> hf

Need local throughput at scale?
  -> vllm

Need hosted moderation quickly?
  -> openai_moderation

Need an OpenAI-style chat/completions endpoint?
  -> openai_compatible

Need your own REST API?
  -> http

Need image specialized local models?
  -> one of the specialized HF adapters
```

## Capability Shape

Each adapter declares an `AdapterCapabilities` record used by the runner and
stored in the run manifest:

| Field | Meaning |
| --- | --- |
| `adapter_name` | stable public alias |
| `probability_scores` | whether the adapter emits a score in `[0, 1]` |
| `batching` | whether it supports batched prediction |
| `concurrency` | whether parallel requests make sense |
| `cost_estimation` | whether cost metadata is tracked |
| `token_accounting` | whether token usage is reported |
| `supported_input_modalities` | valid input types such as text or image |
| `supports_category_outputs` | whether category-level outputs can be preserved |
| `notes` | adapter-specific hints or specialization tags |

## Mock Adapter {#mock}

The `mock` adapter is the recommended first run because it is deterministic and
does not require a GPU or API key.

```bash
geh run --dataset xstest --model mock --limit 10
```

Useful strategies:

- `label_echo` for perfect label mirroring in tests
- `keyword` for a minimal heuristic baseline

## Related References

- [HuggingFace](huggingface.md)
- [vLLM](vllm.md)
- [OpenAI Moderation](openai-moderation.md)
- [OpenAI Compatible](openai-compatible.md)
- [Datasets Overview](../datasets/overview.md)
