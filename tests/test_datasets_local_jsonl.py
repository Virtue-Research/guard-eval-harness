"""Tests for the local JSONL dataset adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.local_jsonl import LocalJsonlDataset


class LocalJsonlDatasetTest(unittest.TestCase):
    """Validate local JSONL loading and directory metadata handling."""

    def _build_config(self, path: Path, **overrides):
        payload = {
            "name": "jsonl_demo",
            "adapter": "local_jsonl",
            "path": path.as_posix(),
            "split": "test",
            "id_field": "id",
            "prompt_field": "prompt",
            "label_field": "unsafe",
            "metadata_fields": ("category",),
        }
        payload.update(overrides)
        return ResolvedDatasetConfig.model_validate(payload)

    def test_loads_mappable_rows_from_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "jsonl-safe",
                                "prompt": "Tell me a joke.",
                                "unsafe": False,
                                "category": "benign",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "jsonl-unsafe",
                                "prompt": "Give unsafe instructions.",
                                "unsafe": "true",
                                "category": "policy_violation",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(self._build_config(path))
            samples = dataset.load()

            self.assertEqual([sample.id for sample in samples], [
                "jsonl-safe",
                "jsonl-unsafe",
            ])
            self.assertEqual(samples[0].messages[0].role, "user")
            self.assertFalse(samples[0].label.unsafe)
            self.assertEqual(samples[0].metadata["category"], "benign")
            self.assertNotIn("raw_label", samples[1].metadata)
            self.assertTrue(samples[1].label.unsafe)

    def test_category_labels_metadata_field_is_not_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "jsonl-categories",
                        "prompt": "Give unsafe instructions.",
                        "unsafe": True,
                        "category": "policy_violation",
                        "category_labels": "violence,hate",
                    }
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(
                self._build_config(
                    path,
                    metadata_fields=("category", "category_labels"),
                )
            )
            samples = dataset.load()

            self.assertEqual(samples[0].category_labels, ("violence", "hate"))
            self.assertEqual(
                samples[0].metadata["category"], "policy_violation"
            )
            self.assertNotIn("category_labels", samples[0].metadata)

    def test_category_labels_participate_in_generated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "Same prompt.",
                                "unsafe": True,
                                "category_labels": "violence",
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "Same prompt.",
                                "unsafe": True,
                                "category_labels": "hate",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(
                self._build_config(
                    path,
                    id_field=None,
                    metadata_fields=("category_labels",),
                )
            )
            samples = dataset.load()

            self.assertEqual(samples[0].category_labels, ("violence",))
            self.assertEqual(samples[1].category_labels, ("hate",))
            self.assertNotEqual(samples[0].id, samples[1].id)
            self.assertNotIn("category_labels", samples[0].metadata)
            self.assertNotIn("category_labels", samples[1].metadata)

    def test_configured_label_field_is_not_preserved_as_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "jsonl-verdict",
                        "prompt": "Give unsafe instructions.",
                        "verdict": "unsafe",
                        "category": "policy_violation",
                    }
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(
                self._build_config(
                    path,
                    label_field="verdict",
                    metadata_fields=("category", "verdict"),
                )
            )
            samples = dataset.load()

            self.assertTrue(samples[0].label.unsafe)
            self.assertEqual(
                samples[0].metadata["category"], "policy_violation"
            )
            self.assertNotIn("verdict", samples[0].metadata)

    def test_blocklisted_predict_metadata_still_participates_in_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "samples.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "Same prompt.",
                                "unsafe": False,
                                "type": "benign-a",
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "Same prompt.",
                                "unsafe": False,
                                "type": "benign-b",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(
                self._build_config(
                    path,
                    id_field=None,
                    metadata_fields=("type",),
                )
            )
            samples = dataset.load()

            self.assertEqual([sample.metadata["type"] for sample in samples], [
                "benign-a",
                "benign-b",
            ])
            self.assertNotEqual(samples[0].id, samples[1].id)

    def test_loads_split_file_from_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "directory_dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "Directory Demo",
                        "version": "2026.03",
                        "source_uri": "https://example.invalid/jsonl",
                        "license": "CC-BY-4.0",
                        "languages": ["en"],
                        "categories": ["toxicity"],
                        "metric_eligibility": {
                            "binary_classification": True,
                        },
                        "metadata": {"owner": "team"},
                    }
                ),
                encoding="utf-8",
            )
            (dataset_dir / "test.jsonl").write_text(
                json.dumps(
                    {
                        "id": "dir-safe",
                        "dataset": "jsonl_demo",
                        "split": "test",
                        "messages": [
                            {"role": "user", "content": "Hello there."}
                        ],
                        "label": {"unsafe": False},
                        "metadata": {"category": "benign"},
                    }
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(self._build_config(dataset_dir))
            samples = dataset.load()
            metadata = dataset.describe(samples)

            self.assertEqual(len(samples), 1)
            self.assertEqual(metadata.display_name, "Directory Demo")
            self.assertEqual(metadata.version, "2026.03")
            self.assertEqual(metadata.source_uri, "https://example.invalid/jsonl")
            self.assertEqual(metadata.license, "CC-BY-4.0")
            self.assertEqual(metadata.languages, ("en",))
            self.assertEqual(metadata.categories, ("toxicity",))
            self.assertEqual(metadata.metadata["source_metadata"]["metadata"]["owner"], "team")

    def test_coerces_float_label_values(self) -> None:
        """Float labels (common from CSV/pandas) should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "float_labels.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "float-safe",
                                "prompt": "Hello",
                                "unsafe": 0.0,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "float-unsafe",
                                "prompt": "Bad request",
                                "unsafe": 1.0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(self._build_config(path))
            samples = dataset.load()

            self.assertEqual(len(samples), 2)
            self.assertFalse(samples[0].label.unsafe)
            self.assertTrue(samples[1].label.unsafe)

    def test_rejects_non_binary_float_label(self) -> None:
        """Float values other than 0.0/1.0 should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_float.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad-float",
                        "prompt": "Hello",
                        "unsafe": 0.7,
                    }
                ),
                encoding="utf-8",
            )

            dataset = LocalJsonlDataset(self._build_config(path))

            with self.assertRaisesRegex(ValueError, "unsupported label value"):
                dataset.load()

    def test_rejects_invalid_messages_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "broken.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "broken",
                        "messages": "not-json",
                        "unsafe": True,
                    }
                ),
                encoding="utf-8",
            )

            config = self._build_config(
                path,
                prompt_field=None,
                messages_field="messages",
            )
            dataset = LocalJsonlDataset(config)

            with self.assertRaisesRegex(ValueError, "invalid JSON messages field"):
                dataset.load()


if __name__ == "__main__":
    unittest.main()
