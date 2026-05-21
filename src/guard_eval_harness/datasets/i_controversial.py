"""Source-backed I-Controversial dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/vinid/instruction-llms-safety-eval/raw/"
    "36a4b8d5c2177ed165bf61f59b590161394f7f12/data/evaluation"
)
_FILENAME = "I-Controversial.json"


@dataset_registry.register("i_controversial")
class IControversialDataset(DatasetAdapter):
    """Load the I-Controversial instruction set."""

    display_name = "I-Controversial"
    source_uri = "https://github.com/vinid/instruction-llms-safety-eval"
    license_name = "CC BY-NC 4.0"
    languages = ("en",)
    categories = ("controversial topics",)
    label_mapping_note = "all I-Controversial instructions are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream I-Controversial JSON file."""
        payload = load_json_payload(
            alias="i_controversial",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        instructions = list(payload["instructions"])
        if self.execution_limit is not None:
            instructions = instructions[: self.execution_limit]
        return [
            {
                "id": f"i-controversial-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(instructions, start=1)
        ]
