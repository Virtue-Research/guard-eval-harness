"""Source-backed Do Anything Now Questions dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/verazuo/jailbreak_llms/raw/"
    "09ab5ecb9632087119af25a04b88b0f496c989be/data/forbidden_question"
)
_FILENAME = "forbidden_question_set.csv"


@dataset_registry.register("do_anything_now_questions")
class DoAnythingNowQuestionsDataset(SourceBackedDatasetAdapter):
    """Load the Do Anything Now harmful question set."""

    display_name = "Do Anything Now Questions"
    source_uri = "https://github.com/verazuo/jailbreak_llms"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Illegal Activity",
        "Hate Speech",
        "Malware Generation",
        "Physical Harm",
        "Fraud",
        "Pornography",
        "Political Lobbying",
        "Privacy Violence",
        "Legal Opinion",
        "Financial Advice",
        "Health Consultation",
        "Government Decision",
    )
    label_mapping_note = "all Do Anything Now questions are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream Do Anything Now CSV file."""
        rows = load_csv_rows(
            alias="do_anything_now_questions",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"do-anything-now-{index}",
                "prompt": row["question"],
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
