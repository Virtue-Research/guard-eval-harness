"""Source-backed AgentHarm dataset adapter."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "main"
_SUPPORTED_CONFIGS = {
    "harmful",
    "harmless_benign",
    "chat",
}


@dataset_registry.register("agent_harm")
class AgentHarmDataset(SourceBackedDatasetAdapter):
    """Load the UK AISI AgentHarm benchmark.

    By default loads the ``harmful`` config.  Set
    ``options.config`` to ``"harmless_benign"`` or
    ``"chat"`` for alternate views.
    """

    display_name = "AgentHarm"
    source_uri = (
        "https://huggingface.co/datasets/"
        "ai-safety-institute/AgentHarm"
    )
    languages = ("en",)
    categories = (
        "Copyright",
        "Cybercrime",
        "Disinformation",
        "Drugs",
        "Fraud",
        "Harassment",
        "Hate",
        "Sexual",
    )
    metadata_fields_to_preserve = (
        "name",
        "category",
        "config",
    )
    label_mapping_note = (
        "harmful config rows are unsafe; harmless_benign "
        "rows are safe; chat config rows are unsafe"
    )
    supported_splits = ("test_public", "validation")

    def _hf_config(self) -> str:
        """Return the configured AgentHarm HF config."""
        config = str(
            self.config.options.get("config", "harmful")
        ).strip()
        if config not in _SUPPORTED_CONFIGS:
            supported = ", ".join(
                sorted(_SUPPORTED_CONFIGS)
            )
            raise ValueError(
                f"agent_harm options.config must be one "
                f"of: {supported}"
            )
        return config

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load AgentHarm rows from Hugging Face."""
        hf_config = self._hf_config()
        is_unsafe = hf_config != "harmless_benign"

        rows = load_hf_rows(
            "ai-safety-institute/AgentHarm",
            subset=hf_config,
            split=self.config.split,
            revision=_REVISION,
            limit=self.execution_limit,
        )
        return [
            {
                "id": str(row.get("id", f"agent-harm-{index}")),
                "prompt": row["prompt"],
                "unsafe": is_unsafe,
                "name": row.get("name"),
                "category": row.get("category"),
                "config": hf_config,
            }
            for index, row in enumerate(rows, start=1)
            if str(row.get("prompt", "")).strip()
        ]
