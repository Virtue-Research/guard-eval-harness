"""Source-backed TweetEval Hate dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"


@dataset_registry.register("tweet_eval_hate")
class TweetEvalHateDataset(DatasetAdapter):
    """Load the TweetEval hate speech detection subset."""

    display_name = "TweetEval Hate"
    source_uri = (
        "https://huggingface.co/datasets/tweet_eval"
    )
    license_name = "See TweetEval per-subset terms"
    languages = ("en",)
    categories = ("Hate Speech",)
    label_mapping_note = (
        "label 1 maps to unsafe (hate) and "
        "label 0 maps to safe (non-hate)"
    )
    supported_splits = ("train", "validation", "test")

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the TweetEval hate split from Hugging Face."""
        rows = load_hf_rows(
            "tweet_eval",
            subset="hate",
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"tweeteval-hate-{index}",
                "prompt": row["text"],
                "unsafe": row["label"] in {1, "1", True},
            }
            for index, row in enumerate(rows, start=1)
        ]
