"""Tests for the Hugging Face VLM guard adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.hf_vlm_guard import HuggingFaceVLMGuardAdapter
from guard_eval_harness.models.multimodal import (
    parse_llama_guard_output,
    parse_llavaguard_output,
)
from guard_eval_harness.schemas import NormalizedSample


class _FakeBatch(dict):
    """Simple processor batch wrapper with a no-op `.to()`."""

    def to(self, device):  # noqa: ANN001
        return self


def _image_sample(
    path: Path,
    *,
    sample_id: str = "sample-1",
    text: str = "Check this image",
) -> NormalizedSample:
    """Build a one-image multimodal sample."""
    return NormalizedSample(
        id=sample_id,
        dataset="image-demo",
        split="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "media",
                        "media": {
                            "modality": "image",
                            "uri": path.as_posix(),
                        },
                    },
                ],
            }
        ],
        label={"unsafe": True},
    )


class HuggingFaceVLMGuardParsingTest(unittest.TestCase):
    """Validate multimodal guard output parsing."""

    def test_parse_llama_guard_output_extracts_category(self) -> None:
        parsed = parse_llama_guard_output("unsafe\nS12")

        self.assertEqual(parsed.unsafe_score, 1.0)
        self.assertEqual(parsed.predicted_categories, ("S12",))
        self.assertEqual(parsed.category_scores["S12"], 1.0)

    def test_parse_llavaguard_output_accepts_markdown_json(self) -> None:
        parsed = parse_llavaguard_output(
            """```json
            {"rating":"Unsafe","category":"O3: Sexual Content",
             "rationale":"Contains explicit nudity."}
            ```"""
        )

        self.assertEqual(parsed.unsafe_score, 1.0)
        self.assertEqual(
            parsed.predicted_categories,
            ("O3: Sexual Content",),
        )
        self.assertEqual(
            parsed.metadata["rationale"],
            "Contains explicit nudity.",
        )


class HuggingFaceVLMGuardAdapterTest(unittest.TestCase):
    """Validate VLM batching and flow-specific preprocessing."""

    def test_internvl_question_template_defaults_missing_context(self) -> None:
        sample = NormalizedSample(
            id="image-only",
            dataset="image-demo",
            split="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "media",
                            "media": {
                                "modality": "image",
                                "uri": "/tmp/sample.png",
                            },
                        }
                    ],
                }
            ],
            label={"unsafe": True},
        )
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="example-org/example-vlm-guard",
            args={"flow": "internvl_chat"},
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)

        question = adapter._internvl_question_for_sample(sample)

        self.assertIn("safe or unsafe", question.lower())
        self.assertIn("No additional text context provided.", question)

    def test_internvl_chat_flow_uses_batch_chat(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        sample_a = NormalizedSample(
            id="sample-a",
            dataset="image-demo",
            split="test",
            messages=[{"role": "user", "content": "first"}],
            label={"unsafe": True},
        )
        sample_b = NormalizedSample(
            id="sample-b",
            dataset="image-demo",
            split="test",
            messages=[{"role": "user", "content": "second"}],
            label={"unsafe": False},
        )
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="example-org/example-vlm-guard",
            args={"flow": "internvl_chat"},
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)

        class _FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

        class _FakeModel:
            device = torch.device("cpu")
            dtype = torch.float32

            def __init__(self) -> None:
                self.captured_args = None

            def eval(self) -> None:
                return None

            def batch_chat(self, tokenizer, pixel_values, **kwargs):  # noqa: ANN001
                self.captured_args = {
                    "tokenizer": tokenizer,
                    "pixel_values_shape": tuple(pixel_values.shape),
                    **kwargs,
                }
                return ["unsafe", "safe"]

        fake_model = _FakeModel()
        with (
            patch.object(
                adapter,
                "_get_internvl_tokenizer",
                return_value=_FakeTokenizer(),
            ),
            patch.object(
                adapter,
                "_get_vlm_model",
                return_value=(fake_model, None),
            ),
            patch.object(
                adapter,
                "_prepare_internvl_batch",
                return_value=(
                    [sample_a, sample_b],
                    torch.ones((3, 3, 8, 8), dtype=torch.float32),
                    [1, 2],
                    ["Question A", "Question B"],
                ),
            ),
        ):
            predictions = adapter.predict_batch(
                [sample_a, sample_b],
                threshold=0.5,
            )

        self.assertEqual(len(predictions), 2)
        self.assertTrue(predictions[0].unsafe_label)
        self.assertFalse(predictions[1].unsafe_label)
        self.assertEqual(
            fake_model.captured_args["questions"],
            ["Question A", "Question B"],
        )
        self.assertEqual(
            fake_model.captured_args["num_patches_list"],
            [1, 2],
        )
        self.assertEqual(
            fake_model.captured_args["pixel_values_shape"],
            (3, 3, 8, 8),
        )

    def test_llama_guard_4_uses_dynamic_cache_by_default(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="meta-llama/Llama-Guard-4-12B",
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)
        fake_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                eos_token_id=0,
            )
        )

        generation_kwargs = adapter._generation_kwargs(fake_processor)

        self.assertEqual(
            generation_kwargs["cache_implementation"],
            "dynamic",
        )
        self.assertFalse(generation_kwargs["use_cache"])

    def test_llama_guard_4_flow_uses_inline_image_paths(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 32), color="red").save(image_path)
            unsafe_sample = _image_sample(image_path, sample_id="unsafe")
            safe_sample = _image_sample(image_path, sample_id="safe")
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name="meta-llama/Llama-Guard-4-12B",
            )
            adapter = HuggingFaceVLMGuardAdapter.from_config(config)

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured_conversations = None
                    self.captured_kwargs = None

                def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN001
                    self.captured_conversations = conversations
                    self.captured_kwargs = kwargs
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (len(conversations), 4),
                                dtype=torch.long,
                            ),
                            "attention_mask": torch.ones(
                                (len(conversations), 4),
                                dtype=torch.long,
                            ),
                        }
                    )

                def batch_decode(self, tokens, skip_special_tokens=True):  # noqa: ANN001
                    return ["unsafe\nS12", "safe"]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(self, **kwargs):
                    batch_size = kwargs["input_ids"].shape[0]
                    return torch.ones((batch_size, 6), dtype=torch.long)

            fake_processor = _FakeProcessor()
            fake_model = _FakeModel()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_processor,
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(fake_model, None),
                ),
            ):
                predictions = adapter.predict_batch(
                    [unsafe_sample, safe_sample],
                    threshold=0.5,
                )

        self.assertEqual(len(predictions), 2)
        self.assertTrue(predictions[0].unsafe_label)
        self.assertEqual(predictions[0].predicted_categories, ("S12",))
        self.assertFalse(predictions[1].unsafe_label)
        content = fake_processor.captured_conversations[0][0]["content"]
        self.assertEqual(content[0]["text"], "Check this image")
        self.assertEqual(content[1]["path"], image_path.as_posix())
        self.assertTrue(fake_processor.captured_kwargs["tokenize"])
        self.assertTrue(fake_processor.captured_kwargs["return_dict"])

    def test_llama_guard_4_flow_keeps_image_only_turns_renderable(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (22, 22), color="red").save(image_path)
            sample = NormalizedSample(
                id="image-only",
                dataset="image-demo",
                split="test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "media",
                                "media": {
                                    "modality": "image",
                                    "uri": image_path.as_posix(),
                                },
                            }
                        ],
                    }
                ],
                label={"unsafe": True},
            )
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name="meta-llama/Llama-Guard-4-12B",
            )
            adapter = HuggingFaceVLMGuardAdapter.from_config(config)

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured_conversations = None

                def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN001
                    self.captured_conversations = conversations
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones((1, 3), dtype=torch.long),
                            "attention_mask": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                        }
                    )

                def batch_decode(self, tokens, skip_special_tokens=True):  # noqa: ANN001
                    return ["safe"]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(self, **kwargs):
                    return torch.ones((1, 5), dtype=torch.long)

            fake_processor = _FakeProcessor()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_processor,
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(
            fake_processor.captured_conversations[0][0]["content"],
            [
                {"type": "image", "path": image_path.as_posix()},
                {"type": "text", "text": ""},
            ],
        )

    def test_llama_guard_3_vision_uses_placeholder_images(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (24, 20), color="blue").save(image_path)
            sample = _image_sample(image_path)
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name="meta-llama/Llama-Guard-3-11B-Vision",
            )
            adapter = HuggingFaceVLMGuardAdapter.from_config(config)

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured_conversations = None
                    self.captured_inputs = None

                def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN001
                    self.captured_conversations = conversations
                    return ["prompt-1"]

                def __call__(self, **kwargs):  # noqa: ANN001
                    self.captured_inputs = kwargs
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones((1, 3), dtype=torch.long),
                            "attention_mask": torch.ones(
                                (1, 3), dtype=torch.long
                            ),
                        }
                    )

                def batch_decode(self, tokens, skip_special_tokens=True):  # noqa: ANN001
                    return ["unsafe\nS9"]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(self, **kwargs):
                    return torch.ones((1, 5), dtype=torch.long)

            fake_processor = _FakeProcessor()
            fake_model = _FakeModel()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_processor,
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(fake_model, None),
                ),
            ):
                predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].predicted_categories, ("S9",))
        content = fake_processor.captured_conversations[0][0]["content"]
        self.assertEqual(content[1], {"type": "image"})
        self.assertEqual(len(fake_processor.captured_inputs["images"]), 1)
        self.assertEqual(
            fake_processor.captured_inputs["images"][0].size, (24, 20)
        )

    def test_llavaguard_flow_parses_json_output(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (18, 18), color="purple").save(image_path)
            sample = _image_sample(image_path)
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
            )
            adapter = HuggingFaceVLMGuardAdapter.from_config(config)

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured_conversations = None

                def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN001
                    self.captured_conversations = conversations
                    return ["prompt-1"]

                def __call__(self, **kwargs):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones((1, 2), dtype=torch.long),
                            "attention_mask": torch.ones(
                                (1, 2), dtype=torch.long
                            ),
                        }
                    )

                def batch_decode(self, tokens, skip_special_tokens=True):  # noqa: ANN001
                    return [
                        """```json
                        {"rating":"Unsafe",
                         "category":"O3: Sexual Content",
                         "rationale":"The image is explicit."}
                        ```"""
                    ]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(self, **kwargs):
                    return torch.ones((1, 4), dtype=torch.long)

            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=(_processor := _FakeProcessor()),
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertTrue(predictions[0].unsafe_label)
        self.assertEqual(
            predictions[0].predicted_categories,
            ("O3: Sexual Content",),
        )
        self.assertEqual(
            predictions[0].metadata["rationale"],
            "The image is explicit.",
        )
        llavaguard_text = _processor.captured_conversations[0][0]["content"][1][
            "text"
        ]
        self.assertIn('Return strict JSON with keys "rating"', llavaguard_text)
        self.assertIn("Conversation context:", llavaguard_text)

    def test_llavaguard_custom_taxonomy_text_keeps_literal_braces(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
            args={
                "taxonomy_text": (
                    'Return JSON like {"rating":"Safe"}.\n'
                    "Context:\n{conversation_text}"
                )
            },
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)

        rendered = adapter._render_llavaguard_taxonomy_text("demo context")

        self.assertIn('{"rating":"Safe"}', rendered)
        self.assertIn("demo context", rendered)

    def test_malformed_outputs_are_skipped_when_partial_predictions_enabled(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)
        sample_a = NormalizedSample(
            id="sample-a",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "hello"}],
            label={"unsafe": False},
        )
        sample_b = sample_a.model_copy(update={"id": "sample-b"})

        with patch.object(
            adapter,
            "_parse_output",
            side_effect=[
                ValueError("bad verdict"),
                (1.0, ("O3",), {"O3": 1.0}, {"raw_output": "unsafe"}),
            ],
        ):
            predictions = adapter._predictions_from_texts(
                [sample_a, sample_b],
                ["bad", "good"],
                threshold=0.5,
                latency_ms=1.0,
            )

        self.assertEqual([p.sample_id for p in predictions], ["sample-b"])

    def test_malformed_outputs_raise_when_partial_predictions_disabled(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
            args={"drop_failed_predictions": False},
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-a",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "hello"}],
            label={"unsafe": False},
        )

        with patch.object(
            adapter,
            "_parse_output",
            side_effect=ValueError("bad verdict"),
        ):
            with self.assertRaisesRegex(ValueError, "bad verdict"):
                adapter._predictions_from_texts(
                    [sample],
                    ["bad"],
                    threshold=0.5,
                    latency_ms=1.0,
                )


if __name__ == "__main__":
    unittest.main()
