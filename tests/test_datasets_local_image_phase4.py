"""Tests for local Phase 4 multimodal dataset adapters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.local_image_dir import LocalImageDirDataset
from guard_eval_harness.datasets.local_image_jsonl import LocalImageJsonlDataset


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
        for sample in samples:
            self.assertNotIn("label_directory", sample.metadata)
            self.assertNotIn(
                "label_directory",
                sample.to_predict_sample().metadata,
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
