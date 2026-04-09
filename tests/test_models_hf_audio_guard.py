"""Tests for the Hugging Face audio guard adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.hf_audio_guard import (
    HuggingFaceAudioGuardAdapter,
)
from guard_eval_harness.schemas import NormalizedPrediction, NormalizedSample


def _audio_sample(sample_id: str = "audio-1") -> NormalizedSample:
    """Build a one-audio normalized sample."""
    return NormalizedSample(
        id=sample_id,
        dataset="audio-demo",
        split="train",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "media",
                        "media": {
                            "modality": "audio",
                            "uri": "/tmp/test.wav",
                        },
                    }
                ],
            }
        ],
        label={"unsafe": True},
    )


class HuggingFaceAudioGuardAdapterTest(unittest.TestCase):
    """Validate audio guard parsing and flow inference."""

    def test_prediction_from_json_output(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_audio_guard",
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
        )
        adapter = HuggingFaceAudioGuardAdapter.from_config(config)
        prediction = adapter._prediction_from_output(
            _audio_sample(),
            '{"unsafe": true, "unsafe_score": 0.9, "categories": ["violence"]}',
            threshold=0.5,
            latency_ms=12.0,
            flow="qwen2_audio",
        )

        self.assertIsInstance(prediction, NormalizedPrediction)
        self.assertTrue(prediction.unsafe_label)
        self.assertEqual(prediction.predicted_categories, ("violence",))

    def test_prediction_falls_back_to_text_parse(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_audio_guard",
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
        )
        adapter = HuggingFaceAudioGuardAdapter.from_config(config)
        prediction = adapter._prediction_from_output(
            _audio_sample(),
            "unsafe",
            threshold=0.5,
            latency_ms=12.0,
            flow="qwen2_audio",
        )

        self.assertEqual(prediction.unsafe_score, 1.0)
        self.assertTrue(prediction.unsafe_label)

    def test_flow_name_inference(self) -> None:
        expectations = {
            "Qwen/Qwen2-Audio-7B-Instruct": "qwen2_audio",
            "nvidia/audio-flamingo-3-hf": "audio_flamingo_3",
            "microsoft/Phi-4-multimodal-instruct": "phi4_multimodal",
            "Qwen/Qwen2.5-Omni-7B": "qwen2_5_omni",
        }
        for model_name, expected in expectations.items():
            with self.subTest(model_name=model_name):
                adapter = HuggingFaceAudioGuardAdapter.from_config(
                    ResolvedModelConfig(
                        adapter="hf_audio_guard",
                        model_name=model_name,
                    )
                )
                self.assertEqual(adapter._flow_name(), expected)

    def test_predict_batch_drops_failures_when_partial_allowed(self) -> None:
        adapter = HuggingFaceAudioGuardAdapter.from_config(
            ResolvedModelConfig(
                adapter="hf_audio_guard",
                model_name="Qwen/Qwen2-Audio-7B-Instruct",
                args={"drop_failed_predictions": True},
            )
        )
        bad = _audio_sample("bad")
        good = _audio_sample("good")
        good_prediction = NormalizedPrediction(
            sample_id="good",
            unsafe_score=0.0,
            unsafe_label=False,
            threshold=0.5,
        )
        with patch.object(
            adapter,
            "_predict_qwen2_audio",
            side_effect=[ValueError("broken"), good_prediction],
        ):
            predictions = adapter.predict_batch(
                [bad, good],
                threshold=0.5,
            )

        self.assertEqual(
            [prediction.sample_id for prediction in predictions],
            ["good"],
        )

    def test_model_loader_raises_clear_error_for_missing_transformers_class(
        self,
    ) -> None:
        adapter = HuggingFaceAudioGuardAdapter.from_config(
            ResolvedModelConfig(
                adapter="hf_audio_guard",
                model_name="Qwen/Qwen2-Audio-7B-Instruct",
            )
        )

        with self.assertRaisesRegex(
            ImportError,
            "Qwen2AudioForConditionalGeneration.*transformers==4.41.0",
        ):
            adapter._model_loader(
                SimpleNamespace(__version__="4.41.0")
            )


if __name__ == "__main__":
    unittest.main()
