"""Source-backed TDCRedTeaming dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/centerforaisafety/tdc2023-starter-kit/raw/"
    "f7ddd7d1c7bda82c6609ecfaf1484565d08d754a/red_teaming/data"
)
_FILENAME = "behaviors.json"


@dataset_registry.register("tdc_red_teaming")
class TDCRedTeamingDataset(SourceBackedDatasetAdapter):
    """Load the TDCRedTeaming prompt sets."""

    display_name = "TDCRedTeaming"
    source_uri = "https://github.com/centerforaisafety/tdc2023-starter-kit"
    license_name = "MIT"
    languages = ("en",)
    supported_splits = ("val", "test")
    label_mapping_note = "all TDCRedTeaming prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the configured TDCRedTeaming split from GitHub."""
        split_dir = "dev" if self.config.split == "val" else "test"
        prompts = list(
            load_json_payload(
                alias="tdc_red_teaming",
                url=f"{_BASE_URL}/{split_dir}/{_FILENAME}",
                filename=f"{self.config.split}-{_FILENAME}",
            )
        )
        if self.execution_limit is not None:
            prompts = prompts[: self.execution_limit]
        return [
            {
                "id": f"tdc-red-teaming-{self.config.split}-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(prompts, start=1)
        ]
