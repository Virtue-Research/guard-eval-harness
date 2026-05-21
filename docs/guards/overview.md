# Guards

A **Guard** owns one safety model's prompt construction and output
parser. The same `Guard` runs unchanged on any backend that exposes
the right shape (`generate` or `classify`).

## Contract

```python
class Guard(ABC):
    name: str
    backend_kind: Literal["generate", "classify"]
    accepts_policy: bool
    accepts_output_format: bool

    def build_messages(self, sample, *, policy=None, output_format=None) -> list[Message]:
        ...

    def parse(self, output) -> ParsedLabel:
        ...
```

Every guard accepts `policy=` and `output_format=` kwargs. Fixed-taxonomy
guards ignore them with an info log; the runner always passes them so
the call shape is uniform.

## Built-in guards

| Guard | `backend_kind` | `accepts_policy` | What it does |
|---|---|---|---|
| `llm` | `generate` | ✓ | Generic chat-LLM-as-guard. Templates a policy + output_format into the system prompt, forwards user/assistant turns. Pairs with any chat model (gpt-4o, claude, local HF). |
| `llama_guard` | `generate` | ✗ | Llama Guard 2 / 3 / 4 family. Forwards the conversation; the tokenizer's chat template embeds the MLCommons taxonomy. Parses `safe` / `unsafe\nS1,S3`. |
| `shieldgemma` | `generate` | ✗ | ShieldGemma. Wraps the last user turn in the fixed "policy expert" template, parses the leading `Yes` / `No` token. |
| `llavaguard` | `generate` | ✗ | LlavaGuard 9-category vision safety. Prepends the JSON-rating system prompt, parses `{rating, category, rationale}`. |
| `hf_image_classifier` | `classify` | ✗ | Generic image-classifier guard. Maps a label distribution to `unsafe_score` via `unsafe_labels` or `label_score_mapping`. |

## Choosing a guard

| If you want to evaluate… | Use |
|---|---|
| A general-purpose chat LLM as guard | `llm` + a policy + an output_format |
| Llama Guard checkpoints (any variant) | `llama_guard` |
| ShieldGemma | `shieldgemma` |
| LlavaGuard / vision JSON-rating models | `llavaguard` |
| An NSFW / unsafe-image discriminator | `hf_image_classifier` + `unsafe_labels` |

## YAML examples

**Llama Guard 3 via vLLM OpenAI-compatible server:**

```yaml
model:
  guard: llama_guard
  backend:
    kind: openai_compat
    name: meta-llama/Llama-Guard-3-8B
    args:
      base_url: http://localhost:8000/v1
      api_key_env: null
```

**Generic LLM-as-guard with custom policy:**

```yaml
model:
  guard: llm
  policy: general_safety       # registered preset
  output_format: safe_unsafe_first_line
  backend:
    kind: openai_compat
    name: gpt-4o-mini
    args:
      api_key_env: OPENAI_API_KEY
```

**Image classifier guard:**

```yaml
model:
  guard: hf_image_classifier
  guard_args:
    unsafe_labels: ["nsfw", "porn"]
  backend:
    kind: hf_image_classifier
    name: Falconsai/nsfw_image_detection
```

## Custom guards

Adding a new safety model is a `Guard` subclass with two methods.
See [Adding a Guard](../developer/adding-guards.md).
