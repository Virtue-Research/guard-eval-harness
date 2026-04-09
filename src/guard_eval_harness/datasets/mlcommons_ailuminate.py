"""Source-backed MLCommons AILuminate dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "d7c115ec83a7e3c447eade38f6120f84cbccb674"
_BASE_URL = (
    "https://raw.githubusercontent.com/mlcommons/ailuminate/"
    f"{_REVISION}/"
    "airr_official_1.0_demo_en_us_prompt_set_release.csv"
)
_FILENAME = "airr_official_1.0_demo_en_us_prompt_set_release.csv"

_HAZARD_LABELS = {
    "vcr": "Violent Crimes",
    "ncr": "Non-Violent Crimes",
    "src": "Sex-Related Crimes",
    "cse": "Child Sexual Exploitation",
    "dfm": "Defamation",
    "sxc_prn": "Sexual Content",
    "prv": "Privacy",
    "ipv": "IP Violations",
    "iwp": "Indiscriminate Weapons",
    "hte": "Hate",
    "ssh": "Suicide & Self-Harm",
    "spc_ele": "Specialized Advice (Elections)",
    "spc_fin": "Specialized Advice (Finance)",
    "spc_hlt": "Specialized Advice (Health)",
}


@dataset_registry.register("mlcommons_ailuminate")
class MLCommonsAILuminateDataset(SourceBackedDatasetAdapter):
    """Load the MLCommons AILuminate demo prompt set."""

    display_name = "MLCommons AILuminate"
    source_uri = "https://github.com/mlcommons/ailuminate"
    license_name = "CC BY 4.0"
    languages = ("en",)
    categories = tuple(_HAZARD_LABELS.values())
    metadata_fields_to_preserve = ("hazard", "persona")
    label_mapping_note = (
        "all AILuminate prompts are unsafe adversarial stimuli"
    )
    supported_splits = ("train",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the AILuminate demo CSV from GitHub."""
        rows = load_csv_rows(
            alias="mlcommons-ailuminate",
            url=_BASE_URL,
            filename=_FILENAME,
        )
        if self.execution_limit is not None:
            rows = rows[: self.execution_limit]
        return [
            {
                "id": row.get(
                    "release_prompt_id",
                    row.get(
                        "Release_prompt_id",
                        f"ailuminate-{index}",
                    ),
                ),
                "prompt": (
                    row.get("prompt_text")
                    or row.get("Prompt_text")
                ),
                "unsafe": True,
                "hazard": (
                    row.get("hazard") or row.get("Hazard")
                ),
                "persona": (
                    row.get("persona") or row.get("Persona")
                ),
            }
            for index, row in enumerate(rows, start=1)
            if str(
                row.get("prompt_text")
                or row.get("Prompt_text", "")
            ).strip()
        ]
