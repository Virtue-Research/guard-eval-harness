"""Source-backed HarmfulQ dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/SALT-NLP/chain-of-thought-bias/raw/"
    "c49f77dd8bf71b65847875aedc7c4a2d02008f28/data/dangerous-q"
)
_FILENAME = "toxic_outs.json"


@dataset_registry.register("harmful_q")
class HarmfulQDataset(DatasetAdapter):
    """Load the HarmfulQ prompt set."""

    display_name = "HarmfulQ"
    source_uri = "https://github.com/SALT-NLP/chain-of-thought-bias"
    license_name = "CC BY SA 4.0"
    languages = ("en",)
    categories = (
        "racism",
        "stereotypes",
        "sexism",
        "illegal activity",
        "toxicity",
        "harmful content",
    )
    label_mapping_note = "all HarmfulQ prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream HarmfulQ JSON file."""
        prompts = list(
            load_json_payload(
                alias="harmful_q",
                url=f"{_BASE_URL}/{_FILENAME}",
                filename=_FILENAME,
            )
        )
        if self.execution_limit is not None:
            prompts = prompts[: self.execution_limit]
        return [
            {
                "id": f"harmful-q-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(prompts, start=1)
        ]
