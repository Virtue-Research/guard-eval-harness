"""Source-backed Sorry-Bench dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"


@dataset_registry.register("sorry_bench")
class SorryBenchDataset(DatasetAdapter):
    """Load the Sorry-Bench unsafe instruction benchmark.

    By default loads the 202406 version (45 categories,
    450 base instructions).  Set ``options.version`` to
    ``"202503"`` for the newer 44-category / 440-row
    release (gated).
    """

    display_name = "Sorry-Bench"
    source_uri = (
        "https://huggingface.co/datasets/"
        "sorry-bench/sorry-bench-202406"
    )
    languages = ("en",)
    categories = (
        "Hate Speech",
        "Crimes",
        "Inappropriate Topics",
        "Unqualified Advice",
    )
    metadata_fields_to_preserve = (
        "category",
        "question_id",
    )
    label_mapping_note = (
        "all Sorry-Bench instructions are unsafe by design"
    )
    supported_splits = ("train",)

    def _repo_id(self) -> str:
        """Return the HF repo ID based on version option."""
        version = str(
            self.config.options.get("version", "202406")
        ).strip()
        return f"sorry-bench/sorry-bench-{version}"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load Sorry-Bench base questions."""
        rows = load_hf_rows(
            self._repo_id(),
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"sorry-bench-{index}-q{row.get('question_id', '')}",
                "prompt": (
                    row["turns"][0]
                    if isinstance(row.get("turns"), list)
                    and row["turns"]
                    else str(row.get("turns", ""))
                ),
                "unsafe": True,
                "category": row.get("category"),
                "question_id": row.get("question_id"),
            }
            for index, row in enumerate(rows, start=1)
            if (
                isinstance(row.get("turns"), list)
                and row["turns"]
                and row["turns"][0] is not None
                and str(row["turns"][0]).strip()
            )
            or (
                not isinstance(row.get("turns"), list)
                and str(row.get("turns", "")).strip()
            )
        ]
