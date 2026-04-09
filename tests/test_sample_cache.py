"""Tests for the NormalizedSample disk cache."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import wave

from guard_eval_harness.datasets.sample_cache import (
    clear_sample_cache,
    compute_cache_key,
    load_cached_samples,
    write_sample_cache,
)
from guard_eval_harness.schemas import (
    MediaPart,
    MediaRef,
    Message,
    NormalizedSample,
    TextPart,
    UnsafeLabel,
)


def _text_sample(
    sample_id: str = "s1",
    text: str = "hello",
) -> NormalizedSample:
    return NormalizedSample(
        id=sample_id,
        dataset="test_ds",
        split="train",
        messages=[Message(role="user", content=text)],
        label=UnsafeLabel(unsafe=False),
    )


def _image_sample(
    image_path: Path,
    sample_id: str = "img1",
) -> NormalizedSample:
    ref = MediaRef(
        modality="image",
        uri=image_path.as_posix(),
        sha256="a" * 64,
        width=10,
        height=10,
    )
    return NormalizedSample(
        id=sample_id,
        dataset="test_ds",
        split="train",
        messages=[
            Message(
                role="user",
                content=[
                    TextPart(text="describe"),
                    MediaPart(media=ref),
                ],
            )
        ],
        label=UnsafeLabel(unsafe=True),
    )


def _write_wav(path: Path) -> None:
    """Write a tiny mono WAV fixture."""
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


def _audio_sample(
    audio_path: Path,
    sample_id: str = "aud1",
) -> NormalizedSample:
    ref = MediaRef(
        modality="audio",
        uri=audio_path.as_posix(),
        sha256="b" * 64,
        mime_type="audio/wav",
        duration_seconds=0.01,
        sample_rate_hz=16000,
        channels=1,
    )
    return NormalizedSample(
        id=sample_id,
        dataset="test_ds",
        split="train",
        messages=[
            Message(
                role="user",
                content=[
                    TextPart(text="listen"),
                    MediaPart(media=ref),
                ],
            )
        ],
        label=UnsafeLabel(unsafe=False),
    )


class ComputeCacheKeyTest(unittest.TestCase):
    """Validate cache key computation."""

    def test_deterministic(self) -> None:
        parts = {"adapter": "foo", "split": "train", "rev": "abc"}
        self.assertEqual(
            compute_cache_key(parts),
            compute_cache_key(parts),
        )

    def test_different_inputs(self) -> None:
        a = compute_cache_key({"adapter": "foo", "split": "train"})
        b = compute_cache_key({"adapter": "foo", "split": "test"})
        self.assertNotEqual(a, b)

    def test_none_preserved_in_key(self) -> None:
        """None values must produce a different key from omitted keys.

        This prevents a limited run (execution_limit=None stripped)
        from sharing a cache entry with a full run.
        """
        with_none = compute_cache_key(
            {"adapter": "foo", "limit": None}
        )
        without = compute_cache_key({"adapter": "foo"})
        self.assertNotEqual(with_none, without)

    def test_none_vs_int_limit_differ(self) -> None:
        """execution_limit=None (full) vs =1 must not collide."""
        full = compute_cache_key(
            {"adapter": "ds", "execution_limit": None}
        )
        limited = compute_cache_key(
            {"adapter": "ds", "execution_limit": 1}
        )
        self.assertNotEqual(full, limited)


class WriteAndLoadTest(unittest.TestCase):
    """Validate round-trip write → load."""

    def test_text_only_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            samples = [_text_sample("s1"), _text_sample("s2")]
            write_sample_cache(cache, "adapter_a", "key1", samples)
            loaded = load_cached_samples(cache, "adapter_a", "key1")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].id, "s1")
            self.assertEqual(loaded[1].id, "s2")

    def test_image_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            img_path = Path(tmpdir) / "test.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            samples = [_image_sample(img_path)]
            write_sample_cache(cache, "adapter_b", "key2", samples)
            loaded = load_cached_samples(cache, "adapter_b", "key2")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                loaded[0].messages[0].image_refs[0].uri,
                img_path.as_posix(),
            )

    def test_empty_samples_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            write_sample_cache(cache, "adapter_c", "key3", [])
            loaded = load_cached_samples(cache, "adapter_c", "key3")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, [])

    def test_audio_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            audio_path = Path(tmpdir) / "test.wav"
            _write_wav(audio_path)
            samples = [_audio_sample(audio_path)]
            write_sample_cache(cache, "adapter_audio", "key_audio", samples)
            loaded = load_cached_samples(
                cache,
                "adapter_audio",
                "key_audio",
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                loaded[0].messages[0].audio_refs[0].uri,
                audio_path.as_posix(),
            )


class CacheMissTest(unittest.TestCase):
    """Validate cache miss scenarios."""

    def test_returns_none_on_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_cached_samples(
                Path(tmpdir), "no_adapter", "no_key"
            )
            self.assertIsNone(result)

    def test_returns_none_on_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            missing = Path("/nonexistent/image.png")
            samples = [_image_sample(missing)]
            write_sample_cache(cache, "adapter_d", "key4", samples)
            loaded = load_cached_samples(
                cache, "adapter_d", "key4"
            )
            self.assertIsNone(loaded)

    def test_returns_none_on_corrupt_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            entry = cache / "adapter_e" / "key5"
            entry.mkdir(parents=True)
            (entry / "samples.jsonl").write_text(
                "not valid json\n"
            )
            loaded = load_cached_samples(
                cache, "adapter_e", "key5"
            )
            self.assertIsNone(loaded)


class ClearCacheTest(unittest.TestCase):
    """Validate cache clearing."""

    def test_clear_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            write_sample_cache(
                cache, "a", "k1", [_text_sample()]
            )
            write_sample_cache(
                cache, "b", "k2", [_text_sample()]
            )
            removed = clear_sample_cache(cache_dir=cache)
            self.assertEqual(removed, 2)
            self.assertFalse((cache / "a").exists())
            self.assertFalse((cache / "b").exists())

    def test_clear_single_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            write_sample_cache(
                cache, "keep", "k1", [_text_sample()]
            )
            write_sample_cache(
                cache, "remove", "k2", [_text_sample()]
            )
            removed = clear_sample_cache(
                cache_dir=cache, adapter_name="remove"
            )
            self.assertEqual(removed, 1)
            self.assertTrue((cache / "keep").exists())
            self.assertFalse((cache / "remove").exists())

    def test_clear_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            removed = clear_sample_cache(
                cache_dir=Path(tmpdir) / "nope"
            )
            self.assertEqual(removed, 0)

    def test_clear_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            with self.assertRaises(ValueError):
                clear_sample_cache(
                    cache_dir=cache,
                    adapter_name="../../etc",
                )


class AtomicWriteTest(unittest.TestCase):
    """Validate no temp files are left behind."""

    def test_no_tmp_files_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            write_sample_cache(
                cache, "adapter_f", "key6", [_text_sample()]
            )
            entry = cache / "adapter_f" / "key6"
            tmp_files = list(entry.glob("_tmp_*"))
            self.assertEqual(tmp_files, [])
            self.assertTrue((entry / "samples.jsonl").exists())


class TextOnlySamplesTest(unittest.TestCase):
    """Text-only samples should always pass media validation."""

    def test_text_only_passes_media_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            samples = [_text_sample("t1"), _text_sample("t2")]
            write_sample_cache(
                cache, "adapter_g", "key7", samples
            )
            loaded = load_cached_samples(
                cache, "adapter_g", "key7"
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
