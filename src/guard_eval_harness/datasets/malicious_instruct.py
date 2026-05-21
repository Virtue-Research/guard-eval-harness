"""Source-backed MaliciousInstruct dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_text_lines,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/Princeton-SysML/Jailbreak_LLM/raw/"
    "1e490531c0ef4bc13430178bf1fbc8f813e34b1a/data"
)
_FILENAME = "MaliciousInstruct.txt"


@dataset_registry.register("malicious_instruct")
class MaliciousInstructDataset(DatasetAdapter):
    """Load the MaliciousInstruct text prompt list."""

    display_name = "MaliciousInstruct"
    source_uri = "https://github.com/Princeton-SysML/Jailbreak_LLM"
    license_name = "MIT"
    languages = ("en",)
    categories = ("malicious instructions",)
    label_mapping_note = "all MaliciousInstruct prompts are unsafe"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream MaliciousInstruct text file."""
        prompts = load_text_lines(
            alias="malicious_instruct",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"malicious-instruct-{index}",
                "prompt": prompt,
                "unsafe": True,
            }
            for index, prompt in enumerate(prompts, start=1)
        ]
