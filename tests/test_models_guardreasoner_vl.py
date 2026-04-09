"""Tests for GuardReasoner-VL flow in hf_vlm_guard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import (
    ResolvedModelConfig,
)
from guard_eval_harness.models.hf_vlm_guard import (
    HuggingFaceVLMGuardAdapter,
)
from guard_eval_harness.models.multimodal import (
    parse_guardreasoner_vl_output,
)
from guard_eval_harness.schemas import NormalizedSample


class _FakeBatch(dict):
    """Processor batch wrapper with a no-op ``.to()``."""

    def to(self, device):  # noqa: ANN001
        return self


def _image_sample(
    path: Path,
    *,
    sample_id: str = "sample-1",
    text: str = "Check this image",
) -> NormalizedSample:
    """Build a single-image multimodal sample."""
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


class GuardReasonerVLParsingTest(unittest.TestCase):
    """Validate GuardReasoner-VL output parsing."""

    def test_parse_harmful_request(self) -> None:
        output = (
            "<think>The user asks for something "
            "dangerous.</think>\n"
            "<result>\nRequest: harmful\n"
            "Response: unharmful\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)
        self.assertTrue(
            parsed.metadata["request_harmful"]
        )
        self.assertFalse(
            parsed.metadata["response_harmful"]
        )

    def test_parse_harmful_response(self) -> None:
        output = (
            "<think>The response is harmful."
            "</think>\n"
            "<result>\nRequest: unharmful\n"
            "Response: harmful\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)
        self.assertFalse(
            parsed.metadata["request_harmful"]
        )
        self.assertTrue(
            parsed.metadata["response_harmful"]
        )

    def test_parse_both_unharmful(self) -> None:
        output = (
            "<think>Everything looks fine.</think>\n"
            "<result>\nRequest: unharmful\n"
            "Response: unharmful\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 0.0)

    def test_parse_both_harmful(self) -> None:
        output = (
            "<think>Both are bad.</think>\n"
            "<result>\nRequest: harmful\n"
            "Response: harmful\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)

    def test_parse_without_result_tags(self) -> None:
        output = "Request: harmful\nResponse: unharmful"
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)

    def test_parse_none_treated_as_unharmful(
        self,
    ) -> None:
        output = (
            "<result>\nRequest: none\n"
            "Response: none\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 0.0)

    def test_parse_fallback_mixed_harmful_unharmful(
        self,
    ) -> None:
        """Regression: mixed 'harmful'+'unharmful' in
        fallback text must resolve to unsafe."""
        output = (
            "The request is harmful but the "
            "response is unharmful."
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)

    def test_parse_fallback_only_harmful(self) -> None:
        output = "This content is harmful."
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)

    def test_parse_fallback_only_unharmful(self) -> None:
        output = "This content is unharmful."
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 0.0)

    def test_parse_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_guardreasoner_vl_output("")

    def test_parse_case_insensitive(self) -> None:
        output = (
            "<result>\nRequest: Harmful\n"
            "Response: Unharmful\n</result>"
        )
        parsed = parse_guardreasoner_vl_output(output)
        self.assertEqual(parsed.unsafe_score, 1.0)


class GuardReasonerVLFlowTest(unittest.TestCase):
    """Validate GuardReasoner-VL adapter flow."""

    def test_flow_auto_detection(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name=(
                "yueliu1999/GuardReasoner-VL-7B"
            ),
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(
            config
        )
        self.assertEqual(
            adapter._flow_name(), "guardreasoner_vl"
        )

    def test_generation_kwargs_defaults(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name=(
                "yueliu1999/GuardReasoner-VL-7B"
            ),
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(
            config
        )
        fake_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                eos_token_id=0,
            )
        )
        gen_kwargs = adapter._generation_kwargs(
            fake_processor
        )
        self.assertEqual(
            gen_kwargs["max_new_tokens"], 4096
        )
        self.assertFalse(gen_kwargs["do_sample"])

    def test_single_sample_prediction(self) -> None:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (32, 32), "red").save(
                img_path
            )
            sample = _image_sample(img_path)
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name=(
                    "yueliu1999/GuardReasoner-VL-7B"
                ),
            )
            adapter = (
                HuggingFaceVLMGuardAdapter.from_config(
                    config
                )
            )

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured_conversations = None

                def apply_chat_template(
                    self, conversations, **kwargs
                ):  # noqa: ANN001
                    self.captured_conversations = (
                        conversations
                    )
                    return ["prompt-text"]

                def __call__(
                    self, **kwargs
                ):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (1, 3),
                                dtype=torch.long,
                            ),
                            "attention_mask": torch.ones(
                                (1, 3),
                                dtype=torch.long,
                            ),
                        }
                    )

                def batch_decode(
                    self,
                    tokens,
                    skip_special_tokens=True,
                ):  # noqa: ANN001
                    return [
                        "<think>Thinking...</think>\n"
                        "<result>\nRequest: harmful\n"
                        "Response: unharmful\n"
                        "</result>"
                    ]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(
                    self, **kwargs
                ):  # noqa: ANN001
                    return torch.ones(
                        (1, 5), dtype=torch.long
                    )

            fake_proc = _FakeProcessor()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_proc,
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                preds = adapter.predict_batch(
                    [sample], threshold=0.5
                )

        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0].unsafe_label)
        self.assertEqual(preds[0].unsafe_score, 1.0)
        # Verify system message was included
        conv = fake_proc.captured_conversations[0]
        self.assertEqual(conv[0]["role"], "system")
        self.assertIn("classifier", conv[0]["content"][0]["text"])

    def test_batch_prediction_varied_lengths(
        self,
    ) -> None:
        """Batch of 2 samples with different text."""
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_a = Path(tmpdir) / "a.png"
            img_b = Path(tmpdir) / "b.png"
            Image.new("RGB", (24, 24), "red").save(
                img_a
            )
            Image.new("RGB", (48, 48), "blue").save(
                img_b
            )
            short = _image_sample(
                img_a,
                sample_id="short",
                text="bad?",
            )
            long = _image_sample(
                img_b,
                sample_id="long",
                text="Please check this "
                "photo for potential violence.",
            )
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name=(
                    "yueliu1999/GuardReasoner-VL-7B"
                ),
            )
            adapter = (
                HuggingFaceVLMGuardAdapter.from_config(
                    config
                )
            )

            class _FakeProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def apply_chat_template(
                    self, conversations, **kwargs
                ):  # noqa: ANN001
                    return [
                        "prompt-1",
                        "prompt-2",
                    ]

                def __call__(
                    self, **kwargs
                ):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (2, 5),
                                dtype=torch.long,
                            ),
                            "attention_mask": torch.ones(
                                (2, 5),
                                dtype=torch.long,
                            ),
                        }
                    )

                def batch_decode(
                    self,
                    tokens,
                    skip_special_tokens=True,
                ):  # noqa: ANN001
                    return [
                        "<result>\n"
                        "Request: harmful\n"
                        "Response: harmful\n"
                        "</result>",
                        "<result>\n"
                        "Request: unharmful\n"
                        "Response: unharmful\n"
                        "</result>",
                    ]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(
                    self, **kwargs
                ):  # noqa: ANN001
                    bs = kwargs["input_ids"].shape[0]
                    return torch.ones(
                        (bs, 7), dtype=torch.long
                    )

            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=_FakeProcessor(),
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                preds = adapter.predict_batch(
                    [short, long], threshold=0.5
                )

        self.assertEqual(len(preds), 2)
        self.assertTrue(preds[0].unsafe_label)
        self.assertFalse(preds[1].unsafe_label)

    def test_prompt_splits_user_and_assistant(
        self,
    ) -> None:
        """Regression: assistant turns must appear under
        'AI assistant:', not under 'Human user:'."""
        try:
            from PIL import Image  # type: ignore[import-untyped]
            import torch
        except ImportError:
            self.skipTest("Pillow/torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (32, 32), "red").save(
                img_path
            )
            sample = NormalizedSample(
                id="multi-turn",
                dataset="test",
                split="test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "How to hack?"},
                            {
                                "type": "media",
                                "media": {
                                    "modality": "image",
                                    "uri": img_path.as_posix(),
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "Sure, here is how.",
                    },
                ],
                label={"unsafe": True},
            )
            config = ResolvedModelConfig(
                adapter="hf_vlm_guard",
                model_name=(
                    "yueliu1999/GuardReasoner-VL-7B"
                ),
            )
            adapter = (
                HuggingFaceVLMGuardAdapter.from_config(
                    config
                )
            )

            class _CaptureProcessor:
                tokenizer = SimpleNamespace(
                    padding_side="right",
                    pad_token_id=0,
                    eos_token_id=0,
                    eos_token="</s>",
                )

                def __init__(self) -> None:
                    self.captured: (
                        list[dict] | None
                    ) = None

                def apply_chat_template(
                    self, conversations, **kwargs
                ):  # noqa: ANN001
                    self.captured = conversations
                    return ["prompt"]

                def __call__(
                    self, **kwargs
                ):  # noqa: ANN001
                    return _FakeBatch(
                        {
                            "input_ids": torch.ones(
                                (1, 3),
                                dtype=torch.long,
                            ),
                            "attention_mask": torch.ones(
                                (1, 3),
                                dtype=torch.long,
                            ),
                        }
                    )

                def batch_decode(
                    self,
                    tokens,
                    skip_special_tokens=True,
                ):  # noqa: ANN001
                    return [
                        "<result>\n"
                        "Request: harmful\n"
                        "Response: harmful\n"
                        "</result>"
                    ]

            class _FakeModel:
                device = torch.device("cpu")

                def eval(self) -> None:
                    return None

                def generate(
                    self, **kwargs
                ):  # noqa: ANN001
                    return torch.ones(
                        (1, 5), dtype=torch.long
                    )

            fake_proc = _CaptureProcessor()
            with (
                patch.object(
                    adapter,
                    "_get_processor",
                    return_value=fake_proc,
                ),
                patch.object(
                    adapter,
                    "_get_vlm_model",
                    return_value=(_FakeModel(), None),
                ),
            ):
                adapter.predict_batch(
                    [sample], threshold=0.5
                )

            # The user message text should contain
            # "AI assistant:\nSure, here is how."
            conv = fake_proc.captured[0]
            user_msg = conv[1]
            text_parts = [
                p["text"]
                for p in user_msg["content"]
                if p.get("type") == "text"
            ]
            full_text = " ".join(text_parts)
            self.assertIn(
                "AI assistant:\nSure, here is how.",
                full_text,
            )
            self.assertIn(
                "Human user:\nHow to hack?",
                full_text,
            )

    def test_model_class_resolution(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf_vlm_guard",
            model_name=(
                "yueliu1999/GuardReasoner-VL-7B"
            ),
        )
        adapter = HuggingFaceVLMGuardAdapter.from_config(
            config
        )
        self.assertEqual(
            adapter._model_class_name(),
            "Qwen2_5_VLForConditionalGeneration",
        )


if __name__ == "__main__":
    unittest.main()
