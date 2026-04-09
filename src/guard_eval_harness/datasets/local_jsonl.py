"""Local JSONL dataset adapter skeleton."""

from __future__ import annotations

import json
from typing import Any

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample


@dataset_registry.register("local_jsonl")
class LocalJsonlDataset(DatasetAdapter):
    """Load normalized or mappable rows from a local JSONL file."""

    source_suffixes = (".jsonl", ".ndjson")

    def load(self) -> list[NormalizedSample]:
        """Load and normalize JSONL rows."""
        path = self._resolve_source_path()
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row {line_number} at {path} must be an object"
                    )
                rows.append(row)
        return self._finalize_samples(rows)
