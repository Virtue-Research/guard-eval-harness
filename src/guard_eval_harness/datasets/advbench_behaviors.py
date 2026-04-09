"""Source-backed AdvBench Behaviors dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/llm-attacks/llm-attacks/raw/"
    "0f505d82e25c15a83b6954db28191b69927a255d/data/advbench"
)
_FILENAME = "harmful_behaviors.csv"


@dataset_registry.register("advbench_behaviors")
class AdvBenchBehaviorsDataset(SourceBackedDatasetAdapter):
    """Load the original AdvBench harmful behaviors CSV."""

    display_name = "AdvBench Behaviors"
    source_uri = "https://github.com/llm-attacks/llm-attacks"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "profanity",
        "graphic depictions",
        "threats",
        "misinformation",
        "discrimination",
        "cybercrime",
        "dangerous activities",
        "illegal activities",
    )
    label_mapping_note = "all AdvBench behavior prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream AdvBench CSV file."""
        rows = load_csv_rows(
            alias="advbench_behaviors",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        return [
            {
                "id": f"advbench-behavior-{index}",
                "prompt": row["goal"],
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
