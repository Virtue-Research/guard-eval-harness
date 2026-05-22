# Adding a Dataset Adapter

This guide walks through adding a dataset adapter that emits canonical
`NormalizedSample` objects.

## 1. Start With The Right Base Class

Use:

- `DatasetAdapter` for simple local or bespoke loaders
- `SourceBackedDatasetAdapter` for built-in datasets backed by a public source
- `MultimodalDatasetAdapter` for image content

## 2. Create A Simple Adapter

```python title="src/guard_eval_harness/datasets/my_dataset.py"
from __future__ import annotations

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.registry.core import dataset_registry
from guard_eval_harness.schemas import Message, NormalizedSample, UnsafeLabel


@dataset_registry.register("my_dataset")
class MyDatasetAdapter(DatasetAdapter):
    def load(self) -> list[NormalizedSample]:
        rows = self._load_rows()
        samples: list[NormalizedSample] = []
        for index, row in enumerate(rows):
            sample = NormalizedSample(
                id=self._make_sample_id(row, index),
                dataset=self.config.name,
                split=self.config.split,
                messages=[Message(role="user", content=row["prompt"])],
                label=UnsafeLabel(unsafe=bool(row["unsafe"])),
                metadata={"source": row.get("source", "unknown")},
            )
            samples.append(sample)
        return samples
```

## 3. Prefer Shared Helpers

The base classes already handle useful behavior:

- `_make_sample_id()` for deterministic IDs
- `_messages_from_mapping()` for common prompt/messages field mappings
- `_finalize_samples()` in source-backed flows
- automatic metadata shaping through `describe()`

## 4. Source-Backed Built-Ins

If the dataset is a built-in public source, `SourceBackedDatasetAdapter` is
usually the best starting point:

```python
from guard_eval_harness.datasets.source_backed import SourceBackedDatasetAdapter


@dataset_registry.register("my_source_dataset")
class MySourceDataset(SourceBackedDatasetAdapter):
    display_name = "My Source Dataset"
    source_uri = "https://example.com/dataset"
    license_name = "MIT"
    supported_splits = ("test",)

    def load_source_rows(self):
        ...
```

The source-backed base class handles split validation, source metadata, and
sample finalization for you.

## 5. Multimodal Datasets

For image datasets, use `MultimodalDatasetAdapter`:

```python
from guard_eval_harness.datasets.multimodal_base import MultimodalDatasetAdapter


@dataset_registry.register("my_image_dataset")
class MyImageDataset(MultimodalDatasetAdapter):
    def load(self) -> list[NormalizedSample]:
        image_ref = self.resolve_image("/abs/path/to/image.png")
        sample = self.normalize_multimodal_row(
            {"image": "image.png"},
            row_number=0,
            text="Is this unsafe?",
            image_ref=image_ref,
            unsafe=True,
        )
        return [sample]
```

Useful helpers:

- `resolve_image()`
- `build_multimodal_message()`
- `normalize_multimodal_row()`

## 6. Register And Verify

```python
@dataset_registry.register("my_dataset", "my-dataset")
class MyDatasetAdapter(DatasetAdapter):
    ...
```

Then verify with:

```bash
geh list datasets
```

## 7. Test The Normalized Contract

```python title="tests/test_datasets_my_dataset.py"
def test_my_dataset_loads_normalized_samples():
    adapter = MyDatasetAdapter(config=...)
    samples = adapter.load()

    assert samples
    assert all(sample.messages for sample in samples)
    assert all(sample.label.unsafe in (True, False) for sample in samples)
```

## 8. Prefer Clear Labels Over Clever Mapping

The most valuable dataset adapters are the ones where future readers can easily
understand:

- what counts as unsafe
- which split is being used
- what metadata is preserved
- how multimodal content is represented
