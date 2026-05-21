"""Source-backed I-CoNa dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/vinid/instruction-llms-safety-eval/raw/"
    "36a4b8d5c2177ed165bf61f59b590161394f7f12/data/evaluation"
)
_FILENAME = "I-CoNa.json"


@dataset_registry.register("i_cona")
class ICoNaDataset(DatasetAdapter):
    """Load the I-CoNa unsafe instruction set."""

    display_name = "I-CoNa"
    source_uri = "https://github.com/vinid/instruction-llms-safety-eval"
    license_name = "CC BY-NC 4.0"
    languages = ("en",)
    categories = ("hate speech",)
    label_mapping_note = "all I-CoNa instructions are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream I-CoNa JSON file."""
        payload = load_json_payload(
            alias="i_cona",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
        )
        instructions = list(payload["instructions"])
        if self.execution_limit is not None:
            instructions = instructions[: self.execution_limit]
        return [
            {
                "id": f"i-cona-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(instructions, start=1)
        ]
