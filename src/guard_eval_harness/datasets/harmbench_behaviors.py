"""Source-backed HarmBench Behaviors dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/centerforaisafety/HarmBench/raw/"
    "1751dd591e3be4bb52cab4d926977a61e304aba5/data/behavior_datasets"
)
_FILENAMES = {
    "val": "harmbench_behaviors_text_val.csv",
    "test": "harmbench_behaviors_text_test.csv",
}


@dataset_registry.register("harmbench_behaviors")
class HarmBenchBehaviorsDataset(SourceBackedDatasetAdapter):
    """Load the real HarmBench Behaviors source files."""

    display_name = "HarmBench Behaviors"
    source_uri = (
        "https://github.com/centerforaisafety/HarmBench/tree/"
        "1751dd591e3be4bb52cab4d926977a61e304aba5/data/behavior_datasets"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Cybercrime & Unauthorized Intrusion",
        "Chemical & Biological Weapons/Drugs",
        "Copyright Violations",
        "Misinformation & Disinformation",
        "Harassment & Bullying",
        "Illegal Activities",
        "General Harm",
    )
    metadata_fields_to_preserve = (
        "category",
        "functional_category",
        "tags",
    )
    label_mapping_note = "all HarmBench behavior prompts are unsafe"
    supported_splits = ("val", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the selected split from the upstream CSV file."""
        filename = _FILENAMES[self.config.split]
        rows = load_csv_rows(
            alias="harmbench_behaviors",
            url=f"{_BASE_URL}/{filename}",
            filename=filename,
        )
        normalized_rows: list[dict[str, object]] = []
        for row in rows:
            prompt = row["Behavior"]
            if row["FunctionalCategory"] == "contextual" and row["ContextString"]:
                prompt = f"{prompt}\n\n{row['ContextString']}"
            normalized_rows.append(
                {
                    "id": row["BehaviorID"],
                    "prompt": prompt,
                    "unsafe": True,
                    "category": row["SemanticCategory"],
                    "functional_category": row["FunctionalCategory"],
                    "tags": row["Tags"],
                }
            )
        return normalized_rows
