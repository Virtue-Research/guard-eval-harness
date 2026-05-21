# Backends

A **Backend** is the inference engine — it knows how to talk to one
runtime (a local HF model, an OpenAI-compatible HTTP endpoint, a vLLM
server, …). Backends know nothing about safety taxonomies; that's the
Guard's job.

## Contract

Two flavors:

```python
class GenerationBackend(Backend):
    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]: ...

class ClassifierBackend(Backend):
    def classify(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[dict[str, float]]: ...
```

The runner dispatches on `guard.backend_kind` ("generate" vs "classify").

## Built-in backends

| Backend | Kind | What it does |
|---|---|---|
| `mock` | generate | Trigger-word heuristic for tests and the bundled demo. Zero deps. |
| `mock_classifier` | classify | Trigger-word heuristic returning `{"unsafe": p, "safe": 1-p}`. |
| `hf_generate` | generate | Local transformers `AutoTokenizer` + `AutoModelForCausalLM`. Applies the model's chat template automatically. |
| `hf_vlm` | generate | Local transformers `AutoProcessor` + `AutoModelForVision2Seq` for image+text → text VLMs (LlavaGuard, etc.). |
| `hf_image_classifier` | classify | Local transformers `AutoImageProcessor` + `AutoModelForImageClassification`. Returns softmax probs. |
| `openai_compat` | generate | OpenAI / vLLM-server / LiteLLM / any chat-completions endpoint. httpx POST with exponential-backoff retry. Handles image content parts. |

## Choosing a backend

| Where the model lives | Use |
|---|---|
| Local HF weights, text-only | `hf_generate` |
| Local HF weights, vision-language | `hf_vlm` |
| Local HF weights, image classifier (NSFW etc.) | `hf_image_classifier` |
| OpenAI / Anthropic / Bedrock / any hosted chat API | `openai_compat` |
| Local vLLM with OpenAI server mode | `openai_compat` (target the vLLM server URL) |
| Local sglang / TGI / LMDeploy with OpenAI mode | `openai_compat` |
| Tests / first-run demo | `mock` |

We deliberately do **not** ship a standalone vLLM in-process backend —
vLLM's own `/v1/chat/completions` server is the path of least resistance
and works through `openai_compat`.

## YAML examples

**Local HF for Llama Guard:**

```yaml
backend:
  kind: hf_generate
  name: meta-llama/Llama-Guard-3-8B
  args:
    device: auto
    dtype: bf16
    trust_remote_code: false
```

**OpenAI:**

```yaml
backend:
  kind: openai_compat
  name: gpt-4o-mini
  args:
    api_key_env: OPENAI_API_KEY
```

**vLLM OpenAI server (no auth):**

```yaml
backend:
  kind: openai_compat
  name: meta-llama/Llama-3.1-8B-Instruct
  args:
    base_url: http://localhost:8000/v1
    api_key_env: null
```

**Image classifier:**

```yaml
backend:
  kind: hf_image_classifier
  name: Falconsai/nsfw_image_detection
  args:
    dtype: fp16
```

## Custom backends

See [Adding a Backend](../developer/adding-backends.md).
