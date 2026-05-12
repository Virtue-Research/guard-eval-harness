"""Source-backed WildJailbreak dataset adapter."""

from __future__ import annotations

from typing import Any

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry

_SUPPORTED_DATA_TYPES = {
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
}
_HF_SPLIT_BY_CONFIG = {
    "train": "train",
    "eval": "train",
}


@dataset_registry.register("wildjailbreak")
class WildJailbreakDataset(SourceBackedDatasetAdapter):
    """Load AllenAI WildJailbreak jailbreak + benign prompts."""

    display_name = "WildJailbreak"
    source_uri = (
        "https://huggingface.co/datasets/"
        "allenai/wildjailbreak"
    )
    license_name = "AI2 ImpACT License (Low Risk)"
    languages = ("en",)
    categories = (
        "Jailbreak",
        "Over-Refusal",
    )
    metadata_fields_to_preserve = ()
    label_mapping_note = (
        "data_type values ending in '_harmful' map to "
        "unsafe; values ending in '_benign' map to safe. "
        "The options.data_type filter selects a subset."
    )
    supported_splits = ("train", "eval")

    def load_source_rows(self) -> list[dict[str, Any]]:
        """Load WildJailbreak rows from Hugging Face."""
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "source-backed datasets require the "
                "'datasets' package"
            ) from exc

        data_type_filter = self.config.options.get(
            "data_type"
        )
        if (
            data_type_filter is not None
            and data_type_filter
            not in _SUPPORTED_DATA_TYPES
        ):
            supported = ", ".join(
                sorted(_SUPPORTED_DATA_TYPES)
            )
            raise ValueError(
                f"wildjailbreak options.data_type must be "
                f"one of: {supported}"
            )

        hf_config = self.config.split
        hf_split = _HF_SPLIT_BY_CONFIG[hf_config]

        ds = load_dataset(
            "allenai/wildjailbreak",
            hf_config,
            split=hf_split,
            delimiter="\t",
            keep_default_na=False,
        )
        rows = (
            list(ds.to_list())
            if hasattr(ds, "to_list")
            else list(ds)
        )

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            data_type = str(
                row.get("data_type", "")
            ).strip()
            if not data_type:
                continue
            if (
                data_type_filter is not None
                and data_type != data_type_filter
            ):
                continue

            adversarial = str(
                row.get("adversarial", "")
            ).strip()
            vanilla = str(
                row.get("vanilla", "")
            ).strip()
            prompt = adversarial if adversarial else vanilla
            if not prompt:
                continue

            normalized.append(
                {
                    "id": (
                        f"wildjailbreak-{index:06d}"
                    ),
                    "prompt": prompt,
                    "unsafe": data_type.endswith(
                        "_harmful"
                    ),
                    "data_type": data_type,
                }
            )
            if (
                self.execution_limit is not None
                and len(normalized)
                >= self.execution_limit
            ):
                break
        return normalized
