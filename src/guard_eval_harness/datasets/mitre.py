"""Source-backed MITRE dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/meta-llama/PurpleLlama/raw/"
    "879a80267292cfe4f2a036e39b0c3edb125112cc/"
    "CybersecurityBenchmarks/datasets/mitre"
)
_FILENAME = "mitre_benchmark_100_per_category.json"


def _is_real_user_prompt(value: object) -> bool:
    """Return whether one MITRE row carries a meaningful user prompt."""
    if not isinstance(value, str):
        return False
    prompt = value.strip()
    return bool(prompt) and prompt != "#ERROR!"


@dataset_registry.register("mitre")
class MitreDataset(SourceBackedDatasetAdapter):
    """Load the MITRE cybersecurity prompt set."""

    display_name = "MITRE"
    source_uri = "https://github.com/meta-llama/PurpleLlama"
    license_name = "MIT"
    languages = ("en",)
    categories = ("cyberattacks",)
    label_mapping_note = "all MITRE benchmark prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream MITRE benchmark JSON file."""
        payload = load_json_payload(
            alias="mitre",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        rows = [
            row
            for row in payload
            if _is_real_user_prompt(row.get("user_prompt"))
        ]
        if self.execution_limit is not None:
            rows = rows[: self.execution_limit]
        return [
            {
                "id": f"mitre-{index}",
                "prompt": str(row["user_prompt"]).strip(),
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
