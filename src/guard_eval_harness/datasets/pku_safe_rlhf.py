"""Source-backed PKU-SafeRLHF dataset adapter."""

from __future__ import annotations

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"


def _harm_categories(category_mapping: dict[str, bool]) -> tuple[str, ...]:
    """Return the active harm categories for one PKU-SafeRLHF response."""
    return tuple(
        key for key, enabled in sorted(category_mapping.items()) if enabled
    )


@dataset_registry.register("pku_safe_rlhf")
class PKUSafeRLHFDataset(SourceBackedDatasetAdapter):
    """Load PKU-SafeRLHF prompt/response pairs as assistant-turn samples."""

    display_name = "PKU-SafeRLHF"
    source_uri = "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF"
    license_name = "Apache 2.0"
    languages = ("en",)
    metadata_fields_to_preserve = (
        "prompt_source",
        "response_source",
        "target_role",
    )
    label_mapping_note = (
        "each response is a separate sample and is unsafe when the source marks "
        "that response as not safe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load both response candidates from the PKU-SafeRLHF test split."""
        rows = load_hf_rows(
            "PKU-Alignment/PKU-SafeRLHF",
            split="test",
            revision=_REVISION,
            limit=self.execution_limit,
        )
        normalized_rows: list[dict[str, object]] = []
        for row_index, row in enumerate(rows, start=1):
            for index in (0, 1):
                response_key = f"response_{index}"
                safe_key = f"is_response_{index}_safe"
                source_key = f"response_{index}_source"
                category_key = f"response_{index}_harm_category"
                response = str(row[response_key]).strip()
                if not response:
                    continue
                normalized_rows.append(
                    {
                        "id": (
                            f"pku-safe-rlhf-{row_index:05d}-{index}-"
                            f"{row[f'response_{index}_sha256']}"
                        ),
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": response},
                        ],
                        "unsafe": not bool(row[safe_key]),
                        "active_categories": _harm_categories(row[category_key]),
                        "prompt_source": row["prompt_source"],
                        "response_source": row[source_key],
                        "target_role": "assistant",
                    }
                )
        return normalized_rows
