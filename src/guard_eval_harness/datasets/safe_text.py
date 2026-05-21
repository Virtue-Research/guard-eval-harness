"""Source-backed SafeText dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/sharonlevy/SafeText/raw/"
    "25bb7a8c4a00634bc0ea1b5e85bec85cf32f36aa"
)
_FILENAME = "SafeText.json"


def _safe_text_limit(
    rows: list[dict[str, object]],
    *,
    execution_limit: int | None,
    keep_unsafe: bool,
) -> list[dict[str, object]]:
    """Limit one class of SafeText while preserving both labels in smoke runs."""
    if execution_limit is None:
        return rows
    safe_limit = execution_limit - (execution_limit // 2)
    unsafe_limit = execution_limit // 2
    limit = unsafe_limit if keep_unsafe else safe_limit
    return rows[:limit]


@dataset_registry.register("safe_text")
class SafeTextDataset(DatasetAdapter):
    """Load the SafeText commonsense physical safety set."""

    display_name = "SafeText"
    source_uri = "https://github.com/sharonlevy/SafeText"
    license_name = "MIT"
    languages = ("en",)
    categories = ("Physical safety",)
    label_mapping_note = (
        "SafeText safe prompts map to safe and unsafe prompts map to unsafe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream SafeText JSON file."""
        payload = load_json_payload(
            alias="safe_text",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        safe_rows: list[dict[str, object]] = []
        unsafe_rows: list[dict[str, object]] = []
        index = 1
        for category, samples in payload.items():
            for text in samples["safe"]:
                safe_rows.append(
                    {
                        "id": f"safe-text-{index}",
                        "prompt": f"{category} {text}",
                        "unsafe": False,
                    }
                )
                index += 1
            for text in samples["unsafe"]:
                unsafe_rows.append(
                    {
                        "id": f"safe-text-{index}",
                        "prompt": f"{category} {text}",
                        "unsafe": True,
                    }
                )
                index += 1
        balanced_safe_rows = _safe_text_limit(
            safe_rows,
            execution_limit=self.execution_limit,
            keep_unsafe=False,
        )
        balanced_unsafe_rows = _safe_text_limit(
            unsafe_rows,
            execution_limit=self.execution_limit,
            keep_unsafe=True,
        )
        return (
            balanced_safe_rows + balanced_unsafe_rows
            if self.execution_limit is not None
            else safe_rows + unsafe_rows
        )
