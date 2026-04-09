"""Local CSV dataset adapter skeleton."""

from __future__ import annotations

import csv
from typing import Any

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample


@dataset_registry.register("local_csv")
class LocalCsvDataset(DatasetAdapter):
    """Load rows from a local CSV file and normalize them."""

    source_suffixes = (".csv",)

    def load(self) -> list[NormalizedSample]:
        """Load and normalize CSV rows."""
        path = self._resolve_source_path()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header row: {path}")
            rows: list[dict[str, Any]] = []
            for row in reader:
                if self._row_is_empty(row):
                    continue
                rows.append(row)
        return self._finalize_samples(rows)

    @staticmethod
    def _row_is_empty(row: dict[str, Any]) -> bool:
        """Return True when a CSV row contains only empty values."""
        return all(value in (None, "") for value in row.values())
