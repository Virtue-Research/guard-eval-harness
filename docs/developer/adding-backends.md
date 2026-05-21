# Adding a Backend

A backend is a `GenerationBackend` or `ClassifierBackend` subclass that
talks to one inference engine. Backends are model-agnostic — they
know how to call something, not what safety taxonomy means.

## Example: minimal generation backend

```python
# src/guard_eval_harness/backends/my_runtime.py
from typing import Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    GenerationBackend,
    backend_registry,
)
from guard_eval_harness.schemas import Message


@backend_registry.register("my_runtime")
class MyRuntimeBackend(GenerationBackend):
    """Call my custom inference HTTP endpoint."""

    kind = "my_runtime"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if config.model is None:
            raise ValueError("my_runtime requires backend.name")
        self.model_name = config.model
        self.endpoint = config.args.get("endpoint", "http://localhost:9000")
        self.timeout = float(config.args.get("timeout", 30.0))

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        import httpx

        outputs: list[str] = []
        with httpx.Client(timeout=self.timeout) as client:
            for messages in batch:
                response = client.post(
                    f"{self.endpoint}/generate",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": m.role, "content": m.text_content}
                            for m in messages
                        ],
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
                response.raise_for_status()
                outputs.append(response.json()["text"])
        return outputs
```

Register by adding it to the side-effect imports in
`backends/__init__.py`.

## Use it from YAML

```yaml
model:
  guard: llama_guard
  backend:
    kind: my_runtime
    name: my-llama-guard-checkpoint
    args:
      endpoint: http://localhost:9000
      timeout: 60
```

## Multimodal backends

If your backend supports image content parts, handle `MediaPart` in
the serialization step. See `backends/openai_compat.py` for the
canonical pattern (text parts → `{"type": "text", ...}`,
image media → `{"type": "image_url", ...}`).

## Classifier backends

For discriminative models that return per-label probabilities, subclass
`ClassifierBackend` and implement `classify(batch) -> list[dict[str, float]]`.
See `backends/hf_image_classifier.py`.

## Conventions

- **Lazy load.** Don't load model weights in `__init__`; do it lazily on
  the first `generate()` call. Tests instantiate every backend without
  network or GPU access.
- **Fail clearly on missing config.** `args.api_key_env` unset, no
  `name`, etc. — raise `RuntimeError`/`ValueError` with the exact env
  var name and what to do.
- **Don't introduce safety logic.** Backends only run inference. If you
  find yourself parsing labels in a backend, that belongs in a Guard.
- **Document `args`.** List every supported key in the docstring with
  defaults. `OpenAICompatibleBackend` is a good reference.
