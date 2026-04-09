# Adding a Model Adapter

This guide walks through adding a new model adapter that plugs into the public
model registry and returns canonical `NormalizedPrediction` objects.

## 1. Create The Adapter File

Create a file in `src/guard_eval_harness/models/`:

```python title="src/guard_eval_harness/models/my_model.py"
from __future__ import annotations

from collections.abc import Sequence

from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.models.templates import sample_to_text
from guard_eval_harness.registry.core import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    NormalizedSample,
)


@model_registry.register("my_model")
class MyModelAdapter(ModelAdapter):
    adapter_name = "my_model"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=False,
            concurrency=True,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text",),
            supports_category_outputs=False,
            notes=("example-adapter",),
        )

    def predict_batch(
        self,
        samples: Sequence[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        predictions: list[NormalizedPrediction] = []
        for sample in samples:
            text = sample_to_text(sample)
            score = self._classify(text)
            predictions.append(
                NormalizedPrediction(
                    sample_id=sample.id,
                    unsafe_score=score,
                    unsafe_label=score >= threshold,
                    threshold=threshold,
                    latency_ms=0.0,
                )
            )
        return predictions
```

## 2. Declare Capabilities Honestly

The runner relies on `AdapterCapabilities`, so treat it as behavior, not
marketing.

Important fields:

- `adapter_name`
- `probability_scores`
- `batching`
- `concurrency`
- `supported_input_modalities`
- `supports_category_outputs`
- `notes`

## 3. Handle Inputs Through The Shared Schema

Use helpers such as `sample_to_text()` when possible, and only branch on
modality when the backend truly needs it.

For multimodal adapters, inspect the sample's normalized message content rather
than relying on raw dataset-specific fields.

## 4. Read Adapter Args From Resolved Config

Adapter-specific values come from `self.config.args`:

```python
self.timeout = float(self.config.args.get("timeout", 30.0))
self.api_key_env = self.config.args.get("api_key_env", "MY_API_KEY")
```

## 5. Register The Adapter

The decorator registers public aliases:

```python
@model_registry.register("my_model", "my-model")
class MyModelAdapter(ModelAdapter):
    ...
```

For external plugins, expose the adapter through the
`guard_eval_harness.models` entry-point group.

## 6. Add Tests

Keep tests close to the normalized contract:

```python title="tests/test_models_my_model.py"
def test_predict_batch_returns_normalized_predictions():
    adapter = MyModelAdapter(config=...)
    samples = [...]
    predictions = adapter.predict_batch(samples, threshold=0.5)

    assert len(predictions) == len(samples)
    assert all(0.0 <= p.unsafe_score <= 1.0 for p in predictions)
    assert all(p.unsafe_label == (p.unsafe_score >= p.threshold) for p in predictions)
```

## 7. Verify Registration

```bash
geh list backends
```

Your new alias should appear in the JSON output.

## Helpful Patterns

- Use `allow_partial_predictions` only when dropping failed samples is a real
  and acceptable behavior for the backend.
- Preserve category outputs when the backend exposes them naturally.
- Keep backend-specific parsing in the adapter, not in the shared runner.
