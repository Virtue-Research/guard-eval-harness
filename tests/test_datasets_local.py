"""Tests for the local dataset adapters (jsonl / csv / image_jsonl / image_dir)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.config import ResolvedDatasetConfig
from guard_eval_harness.datasets.local_csv import LocalCsvDataset
from guard_eval_harness.datasets.local_image_dir import LocalImageDirDataset
from guard_eval_harness.datasets.local_image_jsonl import LocalImageJsonlDataset
from guard_eval_harness.datasets.local_jsonl import LocalJsonlDataset
from guard_eval_harness.registry import (
    dataset_registry,
    ensure_builtin_registrations,
)


# ---------------------------------------------------------------------------
# Fixture-driven smoke tests against bundled examples/datasets/
# ---------------------------------------------------------------------------


class LocalDatasetAdapterTest(unittest.TestCase):
    """Validate the local CSV and JSONL adapters against bundled fixtures."""

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
        self.assertEqual(
            samples[0].messages[0].content, "Tell me a joke about oranges."
        )
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


# ---------------------------------------------------------------------------
# local_csv adapter
# ---------------------------------------------------------------------------


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
            self.assertEqual(
                metadata.metadata["source_metadata"]["metadata"]["owner"],
                "team",
            )

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


# ---------------------------------------------------------------------------
# local_jsonl adapter
# ---------------------------------------------------------------------------


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
            self.assertEqual(samples[1].metadata["raw_label"], "true")

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
            self.assertEqual(
                metadata.metadata["source_metadata"]["metadata"]["owner"],
                "team",
            )

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

    def test_metadata_fields_affect_ids_but_not_predict_metadata_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metadata_ids.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "same prompt",
                                "unsafe": False,
                                "type": "a",
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "same prompt",
                                "unsafe": False,
                                "type": "b",
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

        self.assertEqual(len(samples), 2)
        self.assertNotEqual(samples[0].id, samples[1].id)
        self.assertEqual(samples[0].metadata["type"], "a")
        self.assertEqual(samples[0].to_predict_sample().metadata, {})

    def test_predict_metadata_fields_allow_benign_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predict_metadata.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "prompt": "hello",
                        "unsafe": False,
                        "type": "routing",
                    }
                ),
                encoding="utf-8",
            )
            config = self._build_config(
                path,
                metadata_fields=("type",),
                predict_metadata_fields=("type",),
            )
            dataset = LocalJsonlDataset(config)

            sample = dataset.load()[0]

        self.assertEqual(
            sample.to_predict_sample(
                predict_metadata_fields=config.predict_metadata_fields,
            ).metadata,
            {"type": "routing"},
        )


# ---------------------------------------------------------------------------
# local_image_jsonl adapter
# ---------------------------------------------------------------------------


class LocalImageJsonlDatasetTest(unittest.TestCase):
    """Validate image JSONL loading and multimodal message assembly."""

    def test_loads_relative_image_paths(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "images" / "sample.png"
            image_path.parent.mkdir()
            Image.new("RGB", (20, 20), color="red").save(image_path)
            jsonl_path = root / "test.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "img-safe",
                        "prompt": "Describe this image.",
                        "image": "images/sample.png",
                        "unsafe": False,
                        "category": "benign",
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalImageJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_image_jsonl",
                    adapter="local_image_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    metadata_fields=("category",),
                    options={"category_field": "category"},
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples), 1)
        self.assertFalse(samples[0].label.unsafe)
        self.assertEqual(samples[0].category_labels, ("benign",))
        self.assertEqual(
            samples[0].messages[0].text_content,
            "Describe this image.",
        )

    def test_messages_field_gets_image_attached(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "sample.png"
            Image.new("RGB", (20, 20), color="blue").save(image_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "with-messages",
                        "messages": [
                            {"role": "user", "content": "Look at this"}
                        ],
                        "image_path": "sample.png",
                        "unsafe": True,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalImageJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_image_jsonl",
                    adapter="local_image_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    prompt_field=None,
                    messages_field="messages",
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples[0].messages[0].image_refs), 1)
        self.assertEqual(samples[0].messages[0].text_content, "Look at this")

    def test_load_writes_sample_cache_entry(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "sample-cache"
            image_path = root / "sample.png"
            Image.new("RGB", (20, 20), color="green").save(image_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "cache-image",
                        "prompt": "Check this image.",
                        "image_path": "sample.png",
                        "unsafe": False,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalImageJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_image_jsonl",
                    adapter="local_image_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    options={
                        "sample_cache_dir": cache_dir.as_posix(),
                    },
                )
            )

            samples = dataset.load()

            self.assertEqual(len(samples), 1)
            self.assertEqual(
                len(list(cache_dir.glob("local_image_jsonl/*/samples.jsonl"))),
                1,
            )


# ---------------------------------------------------------------------------
# local_image_dir adapter
# ---------------------------------------------------------------------------


class LocalImageDirDatasetTest(unittest.TestCase):
    """Validate directory-based safe/unsafe image loading."""

    def test_loads_safe_and_unsafe_directories(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_dir = root / "test" / "safe"
            unsafe_dir = root / "test" / "unsafe"
            safe_dir.mkdir(parents=True)
            unsafe_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="white").save(safe_dir / "a.png")
            Image.new("RGB", (16, 16), color="black").save(unsafe_dir / "b.png")
            (unsafe_dir / "b.txt").write_text(
                "unsafe caption", encoding="utf-8"
            )
            dataset = LocalImageDirDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_image_dir",
                    adapter="local_image_dir",
                    path=root.as_posix(),
                    split="test",
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples), 2)
        unsafe_samples = [sample for sample in samples if sample.label.unsafe]
        self.assertEqual(len(unsafe_samples), 1)
        self.assertEqual(
            unsafe_samples[0].messages[0].text_content, "unsafe caption"
        )

    def test_false_string_options_disable_recursion_and_sidecars(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsafe_root = root / "test" / "unsafe"
            nested = unsafe_root / "nested"
            unsafe_root.mkdir(parents=True)
            nested.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="black").save(
                unsafe_root / "top.png"
            )
            Image.new("RGB", (16, 16), color="black").save(
                nested / "nested.png"
            )
            (unsafe_root / "top.txt").write_text(
                "should be ignored",
                encoding="utf-8",
            )
            dataset = LocalImageDirDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_image_dir",
                    adapter="local_image_dir",
                    path=root.as_posix(),
                    split="test",
                    options={
                        "recursive": "false",
                        "caption_sidecars": "false",
                    },
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].messages[0].text_content, "")


if __name__ == "__main__":
    unittest.main()
