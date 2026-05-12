"""Source-backed BeaverTails 330k dataset adapter."""

from __future__ import annotations

from collections import defaultdict

from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry

_REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"


def _active_categories(category_mapping: dict[str, bool]) -> tuple[str, ...]:
    """Return the set of active BeaverTails categories."""
    return tuple(
        key for key, enabled in sorted(category_mapping.items()) if enabled
    )


@dataset_registry.register("beaver_tails_330k")
class BeaverTailsDataset(SourceBackedDatasetAdapter):
    """Load the BeaverTails 330k test split and deduplicate prompt/response pairs."""

    display_name = "BeaverTails 330k"
    source_uri = "https://huggingface.co/datasets/PKU-Alignment/BeaverTails"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Animal Abuse",
        "Child Abuse",
        "Controversial Topics, Politics",
        "Discrimination, Stereotype, Injustice",
        "Drug Abuse, Weapons, Banned Substance",
        "Financial Crime, Property Crime, Theft",
        "Hate Speech, Offensive Language",
        "Misinformation Regarding Ethics, Laws, and Safety",
        "Non-Violent Unethical Behavior",
        "Privacy Violation",
        "Self-Harm",
        "Sexually Explicit, Adult Content",
        "Terrorism, Organized Crime",
        "Violence, Aiding and Abetting, Incitement",
    )
    metadata_fields_to_preserve = ("target_role",)
    label_mapping_note = (
        "assistant responses are unsafe when BeaverTails marks the pair as not safe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load and deduplicate the BeaverTails test split."""
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]] = (
            defaultdict(list)
        )
        if self.execution_limit is None:
            rows = load_hf_rows(
                "PKU-Alignment/BeaverTails",
                split="330k_test",
                revision=_REVISION,
            )
            for row in rows:
                grouped_rows[(row["prompt"], row["response"])].append(row)
        else:
            offset = 0
            fetch_size = max(self.execution_limit * 8, 32)
            while len(grouped_rows) < self.execution_limit:
                rows = load_hf_rows(
                    "PKU-Alignment/BeaverTails",
                    split="330k_test",
                    revision=_REVISION,
                    limit=fetch_size,
                    offset=offset,
                )
                if not rows:
                    break
                for row in rows:
                    grouped_rows[(row["prompt"], row["response"])].append(row)
                offset += len(rows)

        normalized_rows: list[dict[str, object]] = []
        for index, ((prompt, response), group) in enumerate(
            grouped_rows.items(),
            start=1,
        ):
            safe_votes = sum(bool(item["is_safe"]) for item in group)
            unsafe = (safe_votes / len(group)) < 0.5
            normalized_rows.append(
                {
                    "id": f"beavertails-{index}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "unsafe": unsafe,
                    "active_categories": _active_categories(group[0]["category"]),
                    "target_role": "assistant",
                }
            )
            if (
                self.execution_limit is not None
                and len(normalized_rows) >= self.execution_limit
            ):
                break
        return normalized_rows
