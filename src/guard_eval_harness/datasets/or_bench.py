"""Source-backed OR-Bench over-refusal dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"
_SUPPORTED_SUBSETS = {
    "or-bench-80k",
    "or-bench-hard-1k",
    "or-bench-toxic",
}


@dataset_registry.register("or_bench")
class ORBenchDataset(DatasetAdapter):
    """Load the OR-Bench over-refusal benchmark.

    The dataset has three configs selectable via
    ``options.subset``:

    * ``or-bench-80k`` – 80K seemingly-toxic but safe prompts
      (default)
    * ``or-bench-hard-1k`` – harder ambiguous subset (safe)
    * ``or-bench-toxic`` – genuinely toxic prompts (unsafe)
    """

    display_name = "OR-Bench"
    source_uri = (
        "https://huggingface.co/datasets/bench-llm/or-bench"
    )
    languages = ("en",)
    categories = (
        "Over-Refusal",
        "Toxicity",
    )
    metadata_fields_to_preserve = (
        "category",
        "subset",
    )
    label_mapping_note = (
        "or-bench-80k and or-bench-hard-1k prompts are "
        "safe (tests over-refusal); or-bench-toxic prompts "
        "are unsafe"
    )
    supported_splits = ("train",)

    def _subset(self) -> str:
        """Return the configured OR-Bench subset name."""
        subset = str(
            self.config.options.get(
                "subset", "or-bench-80k"
            )
        ).strip()
        if subset not in _SUPPORTED_SUBSETS:
            supported = ", ".join(
                sorted(_SUPPORTED_SUBSETS)
            )
            raise ValueError(
                f"or_bench options.subset must be one of: "
                f"{supported}"
            )
        return subset

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured OR-Bench subset."""
        subset = self._subset()
        is_toxic = subset == "or-bench-toxic"

        rows = load_hf_rows(
            "bench-llm/or-bench",
            subset=subset,
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"or-bench-{index:06d}",
                "prompt": row["prompt"],
                "unsafe": is_toxic,
                "category": row.get("category"),
                "subset": subset,
            }
            for index, row in enumerate(rows, start=1)
            if str(row.get("prompt", "")).strip()
        ]
