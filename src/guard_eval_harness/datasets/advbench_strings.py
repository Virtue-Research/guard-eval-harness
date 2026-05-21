"""Source-backed AdvBench Strings dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/llm-attacks/llm-attacks/raw/"
    "0f505d82e25c15a83b6954db28191b69927a255d/data/advbench"
)
_FILENAME = "harmful_strings.csv"


@dataset_registry.register("advbench_strings")
class AdvBenchStringsDataset(DatasetAdapter):
    """Load the AdvBench harmful string set."""

    display_name = "AdvBench Strings"
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
    label_mapping_note = "all AdvBench strings are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream AdvBench strings CSV file."""
        rows = load_csv_rows(
            alias="advbench_strings",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"advbench-string-{index}",
                "prompt": row["target"],
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
