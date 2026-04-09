"""Tests for media caching utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.datasets.media_cache import (
    compute_sha256,
    image_dimensions,
    resolve_local_image,
    resolve_pil_image,
)


class ComputeSha256Test(unittest.TestCase):
    """Validate SHA-256 computation."""

    def test_deterministic_hash(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        digest = compute_sha256(path)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, compute_sha256(path))
        path.unlink()

    def test_different_content_different_hash(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"aaa")
            path_a = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"bbb")
            path_b = Path(f.name)
        self.assertNotEqual(
            compute_sha256(path_a), compute_sha256(path_b)
        )
        path_a.unlink()
        path_b.unlink()


class ImageDimensionsTest(unittest.TestCase):
    """Validate image dimension extraction."""

    def test_returns_width_height(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as f:
            img = Image.new("RGB", (120, 80))
            img.save(f, format="PNG")
            path = Path(f.name)
        w, h = image_dimensions(path)
        self.assertEqual(w, 120)
        self.assertEqual(h, 80)
        path.unlink()


class ResolvePilImageTest(unittest.TestCase):
    """Validate PIL image resolution to cache."""

    def test_saves_and_hashes(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            img = Image.new("RGB", (10, 10), color="red")
            path, digest = resolve_pil_image(img, cache_dir=cache)
            self.assertTrue(path.exists())
            self.assertEqual(len(digest), 64)
            self.assertTrue(path.name.endswith(".png"))
            w, h = image_dimensions(path)
            self.assertEqual(w, 10)
            self.assertEqual(h, 10)

    def test_idempotent_caching(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            img = Image.new("RGB", (5, 5), color="blue")
            path1, d1 = resolve_pil_image(img, cache_dir=cache)
            path2, d2 = resolve_pil_image(img, cache_dir=cache)
            self.assertEqual(d1, d2)
            self.assertEqual(path1, path2)

    def test_converts_cmyk_images_before_png_save(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            img = Image.new("CMYK", (6, 4), color=(0, 128, 128, 0))
            path, digest = resolve_pil_image(img, cache_dir=cache)
            self.assertTrue(path.exists())
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.suffix, ".png")
            w, h = image_dimensions(path)
            self.assertEqual((w, h), (6, 4))

    def test_preserves_png_native_high_bit_depth_mode(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            img = Image.new("I;16", (2, 2))
            img.putdata([0, 1, 65535, 32768])
            path, _digest = resolve_pil_image(img, cache_dir=cache)
            with Image.open(path) as reopened:
                self.assertEqual(reopened.mode, "I;16")
                pixels = reopened.load()
                self.assertIsNotNone(pixels)
                self.assertEqual(pixels[0, 0], 0)
                self.assertEqual(pixels[1, 0], 1)
                self.assertEqual(pixels[0, 1], 65535)
                self.assertEqual(pixels[1, 1], 32768)


class ResolveLocalImageTest(unittest.TestCase):
    """Validate local image path resolution."""

    def test_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            path = Path(f.name)
        result_path, digest = resolve_local_image(path)
        self.assertEqual(result_path, path)
        self.assertEqual(len(digest), 64)
        path.unlink()

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_local_image(Path("/nonexistent/image.png"))


if __name__ == "__main__":
    unittest.main()
