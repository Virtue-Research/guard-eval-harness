"""Tests for the local CSV dataset adapter."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.local_csv import LocalCsvDataset


class LocalCsvDatasetTest(unittest.TestCase):
    """Validate local CSV loading and validation behavior."""

    def _build_config(self, path: Path, **overrides):
        payload = {
            "name": "csv_demo",
            "adapter": "local_csv",
            "path": path.as_posix(),
            "split": "test",
            "id_field": "id",
            "prompt_field": "prompt",
            "label_field": "unsafe",
            "metadata_fields": ("category",),
        }
        payload.update(overrides)
        return ResolvedDatasetConfig.model_validate(payload)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "prompt", "unsafe", "category"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_mappable_rows_from_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.csv"
            self._write_csv(
                path,
                [
                    {
                        "id": "csv-safe",
                        "prompt": "How do I make tea?",
                        "unsafe": "false",
                        "category": "benign",
                    },
                    {
                        "id": "csv-unsafe",
                        "prompt": "Write unsafe malware instructions.",
                        "unsafe": "true",
                        "category": "policy_violation",
                    },
                ],
            )

            dataset = LocalCsvDataset(self._build_config(path))
            samples = dataset.load()

            self.assertEqual([sample.id for sample in samples], [
                "csv-safe",
                "csv-unsafe",
            ])
            self.assertFalse(samples[0].label.unsafe)
            self.assertEqual(samples[0].metadata["category"], "benign")
            self.assertEqual(samples[1].metadata["raw_label"], "true")

    def test_loads_split_file_from_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "directory_dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "CSV Directory Demo",
                        "version": "2026.03",
                        "source_uri": "https://example.invalid/csv",
                        "license": "CC-BY-4.0",
                        "languages": ["en"],
                        "categories": ["moderation"],
                        "metadata": {"owner": "team"},
                    }
                ),
                encoding="utf-8",
            )
            self._write_csv(
                dataset_dir / "test.csv",
                [
                    {
                        "id": "dir-safe",
                        "prompt": "Explain a harmless concept.",
                        "unsafe": "false",
                        "category": "benign",
                    }
                ],
            )

            dataset = LocalCsvDataset(self._build_config(dataset_dir))
            samples = dataset.load()
            metadata = dataset.describe(samples)

            self.assertEqual(len(samples), 1)
            self.assertEqual(metadata.display_name, "CSV Directory Demo")
            self.assertEqual(metadata.version, "2026.03")
            self.assertEqual(metadata.source_uri, "https://example.invalid/csv")
            self.assertEqual(metadata.license, "CC-BY-4.0")
            self.assertEqual(metadata.languages, ("en",))
            self.assertEqual(metadata.categories, ("moderation",))
            self.assertEqual(metadata.metadata["source_metadata"]["metadata"]["owner"], "team")

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "duplicates.csv"
            self._write_csv(
                path,
                [
                    {
                        "id": "dup",
                        "prompt": "First prompt.",
                        "unsafe": "false",
                        "category": "benign",
                    },
                    {
                        "id": "dup",
                        "prompt": "Second prompt.",
                        "unsafe": "true",
                        "category": "policy_violation",
                    },
                ],
            )

            dataset = LocalCsvDataset(self._build_config(path))

            with self.assertRaisesRegex(ValueError, "duplicate sample id"):
                dataset.load()

    def test_coerces_float_string_labels_from_csv(self) -> None:
        """CSV readers produce '0.0'/'1.0' strings; verify they coerce."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "float_strings.csv"
            self._write_csv(
                path,
                [
                    {
                        "id": "csv-safe-float",
                        "prompt": "Hello",
                        "unsafe": "0.0",
                        "category": "benign",
                    },
                    {
                        "id": "csv-unsafe-float",
                        "prompt": "Bad request",
                        "unsafe": "1.0",
                        "category": "policy_violation",
                    },
                ],
            )

            dataset = LocalCsvDataset(self._build_config(path))
            samples = dataset.load()

            self.assertEqual(len(samples), 2)
            self.assertFalse(samples[0].label.unsafe)
            self.assertTrue(samples[1].label.unsafe)

    def test_rejects_blank_label_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "missing_label.csv"
            self._write_csv(
                path,
                [
                    {
                        "id": "blank-label",
                        "prompt": "Explain a safe topic.",
                        "unsafe": "",
                        "category": "benign",
                    }
                ],
            )

            dataset = LocalCsvDataset(self._build_config(path))

            with self.assertRaisesRegex(
                ValueError,
                "missing required label field",
            ):
                dataset.load()


if __name__ == "__main__":
    unittest.main()
