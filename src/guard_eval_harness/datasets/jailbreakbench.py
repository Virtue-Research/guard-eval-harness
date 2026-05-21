"""Source-backed JailbreakBench dataset adapter."""

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    load_json_payload,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"
_ARTIFACT_BASE_URL = (
    "https://github.com/JailbreakBench/artifacts/raw/"
    "909e68c01d94222b8ad2e397a017e2e12e2adb73/attack-artifacts"
)
_DEFAULT_ARTIFACT_METHOD = "PAIR"
_DEFAULT_ARTIFACT_ATTACK_TYPE = "black_box"
_DEFAULT_ARTIFACT_MODEL = "vicuna-13b-v1.5"


@dataset_registry.register("jailbreakbench")
class JailbreakBenchDataset(DatasetAdapter):
    """Load JailbreakBench attack artifacts by default."""

    display_name = "JailbreakBench Artifacts"
    source_uri = (
        "https://github.com/JailbreakBench/artifacts"
    )
    license_name = "MIT"
    languages = ("en",)
    categories = ("jailbreak",)
    metadata_fields_to_preserve = (
        "goal",
        "behavior",
        "category",
        "attack_method",
        "attack_type",
        "artifact_model",
        "jailbroken",
        "jailbroken_llama_guard1",
        "queries_to_jailbreak",
        "number_of_queries",
    )
    label_mapping_note = (
        "JailbreakBench attack artifact prompts are unsafe; set "
        "options.source='behaviors' for the legacy JBB-Behaviors target list"
    )
    supported_splits = ("harmful",)

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load JailbreakBench attack artifacts or legacy behaviors."""
        if str(self.config.options.get("source", "artifacts")) == "behaviors":
            return self._load_behavior_rows()

        method = str(
            self.config.options.get(
                "artifact_method",
                _DEFAULT_ARTIFACT_METHOD,
            )
        )
        attack_type = str(
            self.config.options.get(
                "artifact_attack_type",
                _DEFAULT_ARTIFACT_ATTACK_TYPE,
            )
        )
        artifact_model = str(
            self.config.options.get(
                "artifact_model",
                _DEFAULT_ARTIFACT_MODEL,
            )
        )
        filename = f"{method}-{attack_type}-{artifact_model}.json"
        payload = load_json_payload(
            alias="jailbreakbench",
            url=(
                f"{_ARTIFACT_BASE_URL}/{method}/{attack_type}/"
                f"{artifact_model}.json"
            ),
            filename=filename,
        )
        jailbreaks = list(payload.get("jailbreaks", []))
        if self.execution_limit is not None:
            jailbreaks = jailbreaks[: self.execution_limit]
        return [
            {
                "id": (
                    "jailbreakbench-artifact-"
                    f"{method.lower()}-{attack_type}-{artifact_model}-"
                    f"{row['index']}"
                ),
                "prompt": row["prompt"],
                "unsafe": True,
                "goal": row.get("goal"),
                "behavior": row.get("behavior"),
                "category": row.get("category"),
                "attack_method": method,
                "attack_type": attack_type,
                "artifact_model": artifact_model,
                "jailbroken": row.get("jailbroken"),
                "jailbroken_llama_guard1": row.get(
                    "jailbroken_llama_guard1"
                ),
                "queries_to_jailbreak": row.get("queries_to_jailbreak"),
                "number_of_queries": row.get("number_of_queries"),
            }
            for row in jailbreaks
            if row.get("prompt") is not None
            and str(row.get("prompt", "")).strip()
        ]

    def _load_behavior_rows(self) -> list[dict[str, object]]:
        """Load the legacy JailbreakBench behavior-target view."""
        rows = load_hf_rows(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="harmful",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"jailbreakbench-{index}",
                "prompt": row.get("Goal", row.get("Behavior", "")),
                "unsafe": True,
                "category": row.get("Category", ""),
            }
            for index, row in enumerate(rows, start=1)
        ]
