"""Tests for summary export formats."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.exports.summary import export_summary


class ExportSummaryTest(unittest.TestCase):
    """Validate CSV summary export fields."""

    def test_csv_export_includes_metrics_count(self) -> None:
        summary = {
            "run_name": "demo",
            "status": "completed",
            "threshold": 0.5,
            "datasets": [
                {
                    "name": "dataset-a",
                    "metrics": {
                        "count": 8,
                        "accuracy": 0.9,
                        "auroc": 0.8,
                        "auprc": 0.7,
                        "precision": 0.6,
                        "recall": 0.5,
                        "f1": 0.4,
                        "fpr": 0.3,
                        "fnr": 0.2,
                        "tp": 4,
                        "tn": 3,
                        "fp": 1,
                        "fn": 0,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.csv"
            with patch(
                "guard_eval_harness.exports.summary.load_or_build_summary",
                return_value=summary,
            ):
                export_summary(
                    tmpdir,
                    fmt="csv",
                    output_path=output_path.as_posix(),
                )
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(["dataset-a"], [row["name"] for row in rows])
        self.assertEqual(["8"], [row["count"] for row in rows])


if __name__ == "__main__":
    unittest.main()
