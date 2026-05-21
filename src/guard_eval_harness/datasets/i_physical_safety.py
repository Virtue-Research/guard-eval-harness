"""Source-backed I-Physical-Safety dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/vinid/instruction-llms-safety-eval/raw/"
    "36a4b8d5c2177ed165bf61f59b590161394f7f12/data/evaluation"
)
_SAFE_FILENAME = "I-PhysicalSafetySafe.json"
_UNSAFE_FILENAME = "I-PhysicalSafetyUnsafe.json"


def _limited_rows(
    rows: list[str],
    *,
    execution_limit: int | None,
    keep: str,
) -> list[str]:
    """Limit one half of the physical-safety pair set for smoke runs."""
    if execution_limit is None:
        return rows
    safe_limit = execution_limit - (execution_limit // 2)
    unsafe_limit = execution_limit // 2
    limit = safe_limit if keep == "safe" else unsafe_limit
    return rows[:limit]


@dataset_registry.register("i_physical_safety")
class IPhysicalSafetyDataset(DatasetAdapter):
    """Load the mixed safe and unsafe I-Physical-Safety prompts."""

    display_name = "I-Physical-Safety"
    source_uri = "https://github.com/vinid/instruction-llms-safety-eval"
    license_name = "CC BY-NC 4.0"
    languages = ("en",)
    categories = ("physical safety",)
    label_mapping_note = (
        "safe prompts map to safe and unsafe prompts map to unsafe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load and normalize the upstream safe and unsafe JSON files."""
        safe_payload = load_json_payload(
            alias="i_physical_safety",
            url=f"{_BASE_URL}/{_SAFE_FILENAME}",
            filename=_SAFE_FILENAME,
        )
        unsafe_payload = load_json_payload(
            alias="i_physical_safety",
            url=f"{_BASE_URL}/{_UNSAFE_FILENAME}",
            filename=_UNSAFE_FILENAME,
        )
        safe_rows = _limited_rows(
            list(safe_payload["instructions"]),
            execution_limit=self.execution_limit,
            keep="safe",
        )
        unsafe_rows = _limited_rows(
            list(unsafe_payload["instructions"]),
            execution_limit=self.execution_limit,
            keep="unsafe",
        )
        normalized_rows: list[dict[str, object]] = []
        normalized_rows.extend(
            {
                "id": f"i-physical-safety-safe-{index}",
                "prompt": prompt,
                "unsafe": False,
            }
            for index, prompt in enumerate(safe_rows, start=1)
        )
        normalized_rows.extend(
            {
                "id": f"i-physical-safety-unsafe-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(unsafe_rows, start=1)
        )
        return normalized_rows
