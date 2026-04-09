"""Tests for the local dataset adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.registry import dataset_registry, ensure_builtin_registrations


class LocalDatasetAdapterTest(unittest.TestCase):
    """Validate the local CSV and JSONL adapters."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_builtin_registrations()

    def _load(self, alias: str, config: ResolvedDatasetConfig) -> list:
        dataset_cls = dataset_registry.get(alias)
        dataset = dataset_cls.from_config(config)
        return dataset.load()

    def test_local_jsonl_reads_fixture(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = ResolvedDatasetConfig(
            name="mock_jsonl",
            adapter="local_jsonl",
            path=(root / "examples" / "datasets" / "mock_samples.jsonl").as_posix(),
        )
        samples = self._load("local_jsonl", config)

        self.assertEqual(len(samples), 16)
        self.assertEqual(samples[0].messages[0].content, "Tell me a joke about oranges.")
        self.assertEqual(samples[1].label.unsafe, True)

    def test_local_csv_reads_fixture(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = ResolvedDatasetConfig(
            name="mock_csv",
            adapter="local_csv",
            path=(root / "examples" / "datasets" / "mock_samples.csv").as_posix(),
            prompt_field="prompt",
            label_field="unsafe",
            metadata_fields=("category",),
        )
        samples = self._load("local_csv", config)

        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0].messages[0].content, "How do I make tea?")
        self.assertEqual(samples[1].metadata["category"], "policy_violation")

    def test_local_jsonl_rejects_invalid_json(self) -> None:
        dataset_cls = dataset_registry.get("local_jsonl")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            dataset = dataset_cls.from_config(
                ResolvedDatasetConfig(
                    name="broken",
                    adapter="local_jsonl",
                    path=path.as_posix(),
                )
            )
            with self.assertRaises(ValueError):
                dataset.load()


if __name__ == "__main__":
    unittest.main()
