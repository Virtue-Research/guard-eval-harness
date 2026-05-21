"""Source-backed HEx-PHI dataset adapter."""

from typing import Any

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry

_REPO_ID = "LLM-Tuning-Safety/HEx-PHI"

_CATEGORY_FILES = {
    "Category_1_Illegal_Activity": "category_1.csv",
    "Category_3_Hate_Harass_Violence": "category_3.csv",
    "Category_4_Malware": "category_4.csv",
    "Category_5_Physical_Harm": "category_5.csv",
    "Category_6_Economic_Harm": "category_6.csv",
    "Category_7_Fraud_Deception": "category_7.csv",
    "Category_8_Adult_Content": "category_8.csv",
    "Category_9_Political_Campaigning": "category_9.csv",
    "Category_10_Privacy_Violation_Activity": "category_10.csv",
    "Category_11_Tailored_Financial_Advice": "category_11.csv",
}

_CATEGORY_SPLITS = tuple(_CATEGORY_FILES)


def _load_category_csv(
    category: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a single HEx-PHI category CSV by file name."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "source-backed datasets require the 'datasets' package"
        ) from exc

    filename = _CATEGORY_FILES[category]
    split = "train"
    if limit is not None:
        split = f"train[:{limit}]"
    dataset = load_dataset(
        _REPO_ID,
        data_files=filename,
        split=split,
    )
    if hasattr(dataset, "to_list"):
        return list(dataset.to_list())
    return [dict(row) for row in dataset]


@dataset_registry.register("hex_phi")
class HExPHIDataset(DatasetAdapter):
    """Load the HEx-PHI harmful question set from Hugging Face."""

    display_name = "HEx-PHI"
    source_uri = (
        "https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = ("harmful_instructions",)
    metadata_fields_to_preserve = ("category",)
    label_mapping_note = "all HEx-PHI prompts are unsafe"
    supported_splits = _CATEGORY_SPLITS

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load all HEx-PHI category splits and merge."""
        normalized: list[dict[str, object]] = []
        per_category = (
            max(1, self.execution_limit // len(_CATEGORY_SPLITS))
            if self.execution_limit
            else None
        )
        for category_split in _CATEGORY_SPLITS:
            rows = _load_category_csv(
                category_split,
                limit=per_category,
            )
            category = category_split.replace("_", " ")
            for row in rows:
                # Column name varies across CSVs; grab first text column
                prompt = row.get("prompt") or next(iter(row.values()))
                index = len(normalized) + 1
                normalized.append(
                    {
                        "id": f"hex-phi-{index}",
                        "prompt": str(prompt),
                        "unsafe": True,
                        "category": category,
                    }
                )
        if self.execution_limit:
            normalized = normalized[: self.execution_limit]
        return normalized
