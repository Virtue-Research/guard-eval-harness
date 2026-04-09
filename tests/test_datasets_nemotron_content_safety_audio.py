"""Tests for the Nemotron Content Safety Audio dataset adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import wave

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.nemotron_content_safety_audio import (
    NemotronContentSafetyAudioDataset,
)


def _write_wav(path: Path) -> None:
    """Write a tiny mono WAV fixture."""
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)


class NemotronContentSafetyAudioDatasetTest(unittest.TestCase):
    """Validate Nemotron dataset normalization."""

    def test_load_normalizes_audio_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            _write_wav(audio_path)
            dataset = NemotronContentSafetyAudioDataset.from_config(
                ResolvedDatasetConfig(
                    name="nemotron_content_safety_audio",
                    adapter="nemotron_content_safety_audio",
                    split="test",
                )
            )
            rows = [
                {
                    "id": "row-1",
                    "audio_filename": "clip.wav",
                    "audio_duration_seconds": 0.1,
                    "prompt": "Threatening speech",
                    "prompt_label": "unsafe",
                    "response": "Reference response",
                    "violated_categories": "violence, harassment",
                }
            ]
            with (
                patch.object(dataset, "load_source_rows", return_value=rows),
                patch(
                    "guard_eval_harness.datasets.nemotron_content_safety_audio.cached_download",
                    return_value=audio_path,
                ),
            ):
                samples = dataset.load()

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].label.unsafe)
        self.assertEqual(
            samples[0].category_labels,
            ("violence", "harassment"),
        )
        self.assertEqual(samples[0].messages[0].text_content, "")
        self.assertEqual(len(samples[0].messages[0].audio_refs), 1)
        self.assertEqual(
            samples[0].metadata["reference_transcript"],
            "Threatening speech",
        )
        self.assertEqual(
            samples[0].metadata["reference_response"],
            "Reference response",
        )

    def test_load_skips_invalid_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            _write_wav(audio_path)
            dataset = NemotronContentSafetyAudioDataset.from_config(
                ResolvedDatasetConfig(
                    name="nemotron_content_safety_audio",
                    adapter="nemotron_content_safety_audio",
                    split="test",
                )
            )
            rows = [
                {
                    "id": "bad-row",
                    "audio_filename": "bad.wav",
                    "prompt_label": "unsafe",
                },
                {
                    "id": "good-row",
                    "audio_filename": "clip.wav",
                    "prompt_label": "safe",
                },
            ]

            def fake_download(*, filename: str, **_: object) -> Path:
                if filename == "bad.wav":
                    return Path(tmpdir) / "missing.wav"
                return audio_path

            with (
                patch.object(dataset, "load_source_rows", return_value=rows),
                patch(
                    "guard_eval_harness.datasets.nemotron_content_safety_audio.cached_download",
                    side_effect=fake_download,
                ),
            ):
                samples = dataset.load()

        self.assertEqual([sample.id for sample in samples], ["good-row"])
        self.assertEqual(
            dataset._source_metadata["skipped_invalid_row_count"],
            1,
        )
        self.assertEqual(
            dataset._source_metadata["skipped_invalid_rows"][0]["sample_id"],
            "bad-row",
        )

    def test_load_rejects_unsafe_upstream_audio_filename(self) -> None:
        dataset = NemotronContentSafetyAudioDataset.from_config(
            ResolvedDatasetConfig(
                name="nemotron_content_safety_audio",
                adapter="nemotron_content_safety_audio",
                split="test",
                options={"skip_invalid_rows": False},
            )
        )
        rows = [
            {
                "id": "bad-row",
                "audio_filename": "../escape.wav",
                "prompt_label": "unsafe",
            }
        ]

        with (
            patch.object(dataset, "load_source_rows", return_value=rows),
            patch(
                "guard_eval_harness.datasets.nemotron_content_safety_audio.cached_download"
            ) as cached_download_mock,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "unsafe audio_filename path",
            ):
                dataset.load()

        cached_download_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
