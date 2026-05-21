"""Source-backed HarmfulQA Questions dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry


_REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"


@dataset_registry.register("harmful_qa_questions")
class HarmfulQAQuestionsDataset(DatasetAdapter):
    """Load the HarmfulQA prompt-only question set."""

    display_name = "HarmfulQA Questions"
    source_uri = "https://huggingface.co/datasets/declare-lab/HarmfulQA"
    license_name = "Apache 2.0"
    languages = ("en",)
    supported_splits = ("train",)
    metadata_fields_to_preserve = ("topic", "subtopic")
    label_mapping_note = "all HarmfulQA question prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load prompt-only HarmfulQA questions from Hugging Face."""
        rows = load_hf_rows(
            "declare-lab/HarmfulQA",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"harmful-qa-question-{index}",
                "prompt": row["question"],
                "unsafe": True,
                "topic": row.get("topic"),
                "subtopic": row.get("subtopic"),
            }
            for index, row in enumerate(rows, start=1)
        ]
