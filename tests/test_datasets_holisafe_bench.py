"""Tests for the HoliSafe-Bench dataset adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.config.models import (
    ResolvedDatasetConfig,
)
from guard_eval_harness.datasets.holisafe_bench import (
    HoliSafeBenchDataset,
    _is_unsafe_by_risk_type,
)
from guard_eval_harness.registry import (
    dataset_registry,
    ensure_builtin_registrations,
)


class HoliSafeBenchRiskTypeTest(unittest.TestCase):
    """Validate risk type to unsafe label mapping."""

    def test_sss_is_safe(self) -> None:
        self.assertFalse(_is_unsafe_by_risk_type("SSS"))

    def test_uuu_is_unsafe(self) -> None:
        self.assertTrue(_is_unsafe_by_risk_type("UUU"))

    def test_suu_is_unsafe(self) -> None:
        self.assertTrue(_is_unsafe_by_risk_type("SUU"))

    def test_usu_is_unsafe(self) -> None:
        self.assertTrue(_is_unsafe_by_risk_type("USU"))

    def test_ssu_is_unsafe(self) -> None:
        self.assertTrue(_is_unsafe_by_risk_type("SSU"))

    def test_case_insensitive(self) -> None:
        self.assertFalse(_is_unsafe_by_risk_type("sss"))
        self.assertTrue(_is_unsafe_by_risk_type("uuu"))


class HoliSafeBenchRegistrationTest(unittest.TestCase):
    """Validate registry integration."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_builtin_registrations()

    def test_registered_in_dataset_registry(
        self,
    ) -> None:
        adapter_cls = dataset_registry.get(
            "holisafe_bench"
        )
        self.assertIs(adapter_cls, HoliSafeBenchDataset)


class HoliSafeBenchLoadTest(unittest.TestCase):
    """Validate loading and normalization logic."""

    def _make_adapter(
        self, **options: object
    ) -> HoliSafeBenchDataset:
        config = ResolvedDatasetConfig(
            name="holisafe_bench",
            adapter="holisafe_bench",
            split="test",
            options=dict(options),
        )
        return HoliSafeBenchDataset.from_config(config)

    def test_load_normalizes_rows(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (32, 32), "red").save(
                img_path
            )
            mock_rows = [
                {
                    "id": 1,
                    "image": img_path.as_posix(),
                    "query": "Describe this image.",
                    "category": "hate",
                    "subcategory": "race",
                    "type": "UUU",
                    "image_safe": False,
                    "image_safety_label": 2,
                },
                {
                    "id": 2,
                    "image": img_path.as_posix(),
                    "query": "What is in this picture?",
                    "category": "violence",
                    "subcategory": "terrorism",
                    "type": "SSS",
                    "image_safe": True,
                    "image_safety_label": 0,
                },
            ]

            adapter = self._make_adapter(
                media_cache_dir=tmpdir,
                no_sample_cache=True,
            )
            with patch.object(
                adapter,
                "_load_hf_rows",
                return_value=mock_rows,
            ):
                samples = adapter.load()

        self.assertEqual(len(samples), 2)
        # UUU => unsafe
        self.assertTrue(samples[0].label.unsafe)
        self.assertEqual(
            samples[0].category_labels, ("hate",)
        )
        self.assertEqual(
            samples[0].metadata["subcategory"], "race"
        )
        # SSS => safe
        self.assertFalse(samples[1].label.unsafe)
        self.assertEqual(
            samples[1].category_labels, ("violence",)
        )

    def test_skips_rows_without_image(self) -> None:
        mock_rows = [
            {
                "id": 1,
                "image": None,
                "query": "No image here.",
                "category": "hate",
                "subcategory": "race",
                "type": "UUU",
                "image_safe": False,
                "image_safety_label": 2,
            },
        ]
        adapter = self._make_adapter(no_sample_cache=True)
        with patch.object(
            adapter,
            "_load_hf_rows",
            return_value=mock_rows,
        ):
            samples = adapter.load()
        self.assertEqual(len(samples), 0)

    def test_rejects_unsupported_split(self) -> None:
        config = ResolvedDatasetConfig(
            name="holisafe_bench",
            adapter="holisafe_bench",
            split="train",
        )
        adapter = HoliSafeBenchDataset.from_config(
            config
        )
        with self.assertRaises(ValueError):
            adapter.load()

    def test_metadata_fields_preserved(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (16, 16), "blue").save(
                img_path
            )
            mock_rows = [
                {
                    "id": 42,
                    "image": img_path.as_posix(),
                    "query": "Hello",
                    "category": "privacy",
                    "subcategory": (
                        "facial_data_exposure"
                    ),
                    "type": "USU",
                    "image_safe": False,
                    "image_safety_label": 8,
                },
            ]
            adapter = self._make_adapter(
                media_cache_dir=tmpdir,
                no_sample_cache=True,
            )
            with patch.object(
                adapter,
                "_load_hf_rows",
                return_value=mock_rows,
            ):
                samples = adapter.load()

        meta = samples[0].metadata
        self.assertEqual(meta["id"], 42)
        self.assertEqual(meta["type"], "USU")
        self.assertNotIn("image_safety_label", meta)
        self.assertNotIn("image_safe", meta)


if __name__ == "__main__":
    unittest.main()
