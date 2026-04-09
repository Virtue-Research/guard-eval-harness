"""Tests for the local audio JSONL dataset adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import wave

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.local_audio_jsonl import (
    LocalAudioJsonlDataset,
)


def _write_wav(path: Path) -> None:
    """Write a tiny mono WAV fixture."""
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)


class LocalAudioJsonlDatasetTest(unittest.TestCase):
    """Validate audio JSONL loading and multimodal message assembly."""

    def test_loads_relative_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio" / "sample.wav"
            audio_path.parent.mkdir()
            _write_wav(audio_path)
            jsonl_path = root / "test.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "audio-safe",
                        "prompt": "Classify this audio.",
                        "audio": "audio/sample.wav",
                        "unsafe": False,
                        "category": "benign",
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
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
            "Classify this audio.",
        )
        self.assertEqual(len(samples[0].messages[0].audio_refs), 1)

    def test_messages_field_gets_audio_attached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "with-messages",
                        "messages": [
                            {"role": "user", "content": "Listen carefully"}
                        ],
                        "audio_path": "sample.wav",
                        "unsafe": True,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    prompt_field=None,
                    messages_field="messages",
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples[0].messages[0].audio_refs), 1)
        self.assertEqual(
            samples[0].messages[0].text_content,
            "Listen carefully",
        )

    def test_load_writes_sample_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "sample-cache"
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "cache-audio",
                        "prompt": "Classify this audio.",
                        "audio_path": "sample.wav",
                        "unsafe": False,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
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
                len(list(cache_dir.glob("local_audio_jsonl/*/samples.jsonl"))),
                1,
            )

    def test_messages_field_reuses_existing_audio_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "with-audio-in-message",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Listen carefully"},
                                    {
                                        "type": "audio",
                                        "audio_url": "sample.wav",
                                    },
                                ],
                            }
                        ],
                        "audio_path": "sample.wav",
                        "unsafe": False,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    prompt_field=None,
                    messages_field="messages",
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples[0].messages[0].audio_refs), 1)
        self.assertEqual(
            samples[0].messages[0].audio_refs[0].uri,
            audio_path.as_posix(),
        )
        self.assertEqual(
            samples[0].messages[0].text_content,
            "Listen carefully",
        )

    def test_messages_field_targets_later_user_turn_with_existing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "later-user-audio",
                        "messages": [
                            {"role": "user", "content": "Intro text"},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Actual audio turn"},
                                    {
                                        "type": "audio",
                                        "audio_url": "sample.wav",
                                    },
                                ],
                            },
                        ],
                        "audio_path": "sample.wav",
                        "unsafe": False,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    prompt_field=None,
                    messages_field="messages",
                )
            )

            samples = dataset.load()

        all_audio_refs = [
            audio_ref
            for message in samples[0].messages
            for audio_ref in message.audio_refs
        ]
        self.assertEqual(len(all_audio_refs), 1)
        self.assertEqual(all_audio_refs[0].uri, audio_path.as_posix())
        self.assertEqual(samples[0].messages[0].text_content, "Intro text")
        self.assertEqual(samples[0].messages[1].text_content, "Actual audio turn")

    def test_messages_field_removes_audio_from_non_target_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "id": "multiple-audio-turns",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Intro"},
                                    {
                                        "type": "audio",
                                        "audio_url": "sample.wav",
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "Response"},
                                    {
                                        "type": "audio",
                                        "audio_url": "sample.wav",
                                    },
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Actual audio turn"},
                                    {
                                        "type": "audio",
                                        "audio_url": "sample.wav",
                                    },
                                ],
                            },
                        ],
                        "audio_path": "sample.wav",
                        "unsafe": False,
                    }
                ),
                encoding="utf-8",
            )
            dataset = LocalAudioJsonlDataset.from_config(
                ResolvedDatasetConfig(
                    name="local_audio_jsonl",
                    adapter="local_audio_jsonl",
                    path=jsonl_path.as_posix(),
                    split="test",
                    prompt_field=None,
                    messages_field="messages",
                )
            )

            samples = dataset.load()

        self.assertEqual(len(samples[0].messages[0].audio_refs), 0)
        self.assertEqual(len(samples[0].messages[1].audio_refs), 0)
        self.assertEqual(len(samples[0].messages[2].audio_refs), 1)
        self.assertEqual(
            samples[0].messages[2].audio_refs[0].uri,
            audio_path.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
