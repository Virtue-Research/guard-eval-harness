"""Tests for the vLLM adapter."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.schemas import NormalizedSample


def _make_output(text: str) -> SimpleNamespace:
    """Build a mock vLLM RequestOutput."""
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _sample(
    sample_id: str = "s1",
    content: str = "Check this",
    unsafe: bool = True,
) -> NormalizedSample:
    return NormalizedSample(
        id=sample_id,
        dataset="demo",
        split="test",
        messages=[{"role": "user", "content": content}],
        label={"unsafe": unsafe},
    )


class VLLMAdapterTest(unittest.TestCase):
    """Validate prompt rendering, score extraction, and batching."""

    def _make_adapter(self, **extra_args):
        from guard_eval_harness.models.vllm_adapter import (
            VLLMAdapter,
        )

        args = dict(extra_args)
        config = ResolvedModelConfig(
            adapter="vllm",
            model_name="test-model",
            args=args,
        )
        with patch(
            "guard_eval_harness.models.vllm_adapter.find_spec",
            return_value=True,
        ):
            return VLLMAdapter.from_config(config)

    # ------------------------------------------------------------------
    # Import guard
    # ------------------------------------------------------------------

    def test_vllm_not_installed_raises(self) -> None:
        from guard_eval_harness.models.vllm_adapter import (
            VLLMAdapter,
        )

        config = ResolvedModelConfig(adapter="vllm", model_name="test-model")
        with patch(
            "guard_eval_harness.models.vllm_adapter.find_spec",
            return_value=None,
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                VLLMAdapter(config)
            self.assertIn("guard-eval-harness[vllm]", str(ctx.exception))

    def test_quiet_mode_sets_vllm_logging_to_error(self) -> None:
        from guard_eval_harness.models.vllm_adapter import (
            _configure_vllm_quiet_mode,
        )

        with patch.dict(os.environ, {"GEH_DEBUG": ""}, clear=False):
            os.environ.pop("VLLM_LOGGING_LEVEL", None)
            _configure_vllm_quiet_mode()

        self.assertEqual("ERROR", os.environ["VLLM_LOGGING_LEVEL"])
        self.assertEqual("ERROR", os.environ["FLASHINFER_LOGGING_LEVEL"])
        self.assertEqual(logging.ERROR, logging.getLogger("vllm").level)
        self.assertTrue(
            any(
                entry[0] == "ignore"
                and getattr(entry[3], "pattern", "") == "nvidia_cutlass_dsl(\\..*)?"
                for entry in warnings.filters
            )
        )

    def test_debug_mode_preserves_vllm_logging(self) -> None:
        from guard_eval_harness.models.vllm_adapter import (
            _configure_vllm_quiet_mode,
        )

        with patch.dict(
            os.environ,
            {"GEH_DEBUG": "1", "VLLM_LOGGING_LEVEL": "INFO"},
            clear=False,
        ):
            _configure_vllm_quiet_mode()
            self.assertEqual("INFO", os.environ["VLLM_LOGGING_LEVEL"])

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def test_capabilities(self) -> None:
        adapter = self._make_adapter()
        caps = adapter.capabilities
        self.assertTrue(caps.batching)
        self.assertFalse(caps.probability_scores)
        self.assertFalse(caps.concurrency)
        self.assertEqual(caps.adapter_name, "vllm")
        self.assertTrue(caps.supports_category_outputs)

    def test_generation_capabilities_include_image(self) -> None:
        adapter = self._make_adapter(task="text-generation")
        self.assertIn("image", adapter.capabilities.supported_input_modalities)

    def test_sampling_params_auto_constrains_to_text_mapping_choices(
        self,
    ) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        adapter = self._make_adapter(
            text_score_mapping={"safe": 0.0, "unsafe": 1.0},
        )

        params = adapter._sampling_params()

        self.assertIsNotNone(params.structured_outputs)
        self.assertEqual(
            params.structured_outputs.choice,
            ["safe", "unsafe"],
        )

    def test_sampling_params_accepts_guided_choice_alias(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        adapter = self._make_adapter(guided_choice=["yes", "no"])

        params = adapter._sampling_params()

        self.assertIsNotNone(params.structured_outputs)
        self.assertEqual(
            params.structured_outputs.choice,
            ["yes", "no"],
        )

    def test_text_score_mapping_accepts_unique_truncated_prefix(self) -> None:
        adapter = self._make_adapter(
            text_score_mapping={"safe": 0.0, "unsafe": 1.0},
        )

        self.assertEqual(adapter._unsafe_score_from_generated_text("saf"), 0.0)
        self.assertEqual(
            adapter._unsafe_score_from_generated_text("unsa"),
            1.0,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def test_prompt_plain_text(self) -> None:
        adapter = self._make_adapter()
        sample = _sample()
        prompt = adapter._prompt_for_sample(sample)
        self.assertEqual(prompt, "user: Check this")

    def test_prompt_template(self) -> None:
        adapter = self._make_adapter(
            prompt_template="{dataset}::{messages_text}"
        )
        sample = _sample()
        prompt = adapter._prompt_for_sample(sample)
        self.assertEqual(prompt, "demo::user: Check this")

    def test_prompt_chat_template(self) -> None:
        adapter = self._make_adapter(apply_chat_template=True)
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = (
            "<|user|>Check this<|end|>"
        )
        with patch.object(
            adapter, "_get_tokenizer", return_value=fake_tokenizer
        ):
            prompt = adapter._prompt_for_sample(_sample())
        fake_tokenizer.apply_chat_template.assert_called_once()
        self.assertEqual(prompt, "<|user|>Check this<|end|>")

    def test_prompt_llama_guard(self) -> None:
        adapter = self._make_adapter(apply_chat_template=True)
        adapter.config = ResolvedModelConfig(
            adapter="vllm",
            model_name="meta-llama/Llama-Guard-3-8B",
            args={"apply_chat_template": True},
        )
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = (
            "<llama_guard>prompt</llama_guard>"
        )
        with (
            patch.object(
                adapter, "_get_tokenizer", return_value=fake_tokenizer
            ),
            patch(
                "guard_eval_harness.models.vllm_adapter"
                ".uses_llama_guard_chat_template",
                return_value=True,
            ),
            patch(
                "guard_eval_harness.models.vllm_adapter"
                ".prepare_llama_guard_chat_messages",
                return_value=[{"role": "user", "content": "shaped"}],
            ) as mock_prepare,
        ):
            prompt = adapter._prompt_for_sample(_sample())
        mock_prepare.assert_called_once()
        self.assertEqual(prompt, "<llama_guard>prompt</llama_guard>")

    def test_llavaguard_multimodal_message_shape(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            adapter = self._make_adapter(flow="llavaguard")
            sample = NormalizedSample(
                id="image-1",
                dataset="demo",
                split="test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check this image"},
                            {
                                "type": "media",
                                "media": {
                                    "modality": "image",
                                    "uri": image_path.as_posix(),
                                },
                            },
                        ],
                    }
                ],
                label={"unsafe": True},
            )

            messages = adapter._message_content_for_vllm(sample)

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image_url")
        self.assertIn(
            "vision safety classifier",
            messages[0]["content"][1]["text"],
        )

    def test_predict_batch_routes_multimodal_samples_to_chat(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            adapter = self._make_adapter(flow="internvl_chat")
            sample = NormalizedSample(
                id="image-1",
                dataset="demo",
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
            fake_llm = MagicMock()
            fake_llm.chat.return_value = [_make_output("safe")]
            with patch.object(adapter, "_get_llm", return_value=fake_llm):
                preds = adapter.predict_batch([sample], threshold=0.5)

        fake_llm.chat.assert_called_once()
        self.assertEqual(len(preds), 1)
        self.assertFalse(preds[0].unsafe_label)
        self.assertTrue(preds[0].metadata["multimodal"])

    def test_predict_batch_skips_text_prompt_builder_for_multimodal_samples(
        self,
    ) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            adapter = self._make_adapter(flow="internvl_chat")
            sample = NormalizedSample(
                id="image-1",
                dataset="demo",
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
            with (
                patch.object(
                    adapter,
                    "_prompt_for_sample",
                    side_effect=AssertionError(
                        "text prompt path should not run"
                    ),
                ),
                patch.object(
                    adapter,
                    "_predict_multimodal_generation",
                    return_value=[],
                ) as mock_predict,
            ):
                adapter.predict_batch([sample], threshold=0.5)

        mock_predict.assert_called_once()

    def test_multimodal_prompt_build_failures_drop_bad_samples(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color="red").save(image_path)
            adapter = self._make_adapter(flow="internvl_chat")
            good = NormalizedSample(
                id="good",
                dataset="demo",
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
            bad = good.model_copy(update={"id": "bad"})
            fake_llm = MagicMock()
            fake_llm.chat.return_value = [_make_output("unsafe")]
            with (
                patch.object(adapter, "_get_llm", return_value=fake_llm),
                patch.object(
                    adapter,
                    "_message_content_for_vllm",
                    side_effect=[
                        ValueError("bad image"),
                        [{"role": "user", "content": "ok"}],
                    ],
                ),
            ):
                preds = adapter.predict_batch([bad, good], threshold=0.5)

        fake_llm.chat.assert_called_once()
        self.assertEqual([pred.sample_id for pred in preds], ["good"])

    # ------------------------------------------------------------------
    # Sampling params
    # ------------------------------------------------------------------

    def test_sampling_params_defaults(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        adapter = self._make_adapter()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"vllm": MagicMock(SamplingParams=mock_cls)},
        ):
            adapter._sampling_params()
        mock_cls.assert_called_once_with(max_tokens=16, temperature=0.0)

    def test_sampling_params_custom(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed")
        adapter = self._make_adapter(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>"],
        )
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"vllm": MagicMock(SamplingParams=mock_cls)},
        ):
            adapter._sampling_params()
        mock_cls.assert_called_once_with(
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>"],
        )

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    def test_score_unsafe_text(self) -> None:
        adapter = self._make_adapter()
        score = adapter._unsafe_score_from_generated_text("unsafe")
        self.assertEqual(score, 1.0)

    def test_score_safe_text(self) -> None:
        adapter = self._make_adapter()
        score = adapter._unsafe_score_from_generated_text("safe")
        self.assertEqual(score, 0.0)

    def test_text_score_mapping(self) -> None:
        adapter = self._make_adapter(text_score_mapping={"yes": 1.0, "no": 0.0})
        self.assertEqual(adapter._unsafe_score_from_generated_text("no"), 0.0)
        self.assertEqual(adapter._unsafe_score_from_generated_text("yes"), 1.0)

    def test_text_score_mapping_prefix_match(self) -> None:
        adapter = self._make_adapter(
            text_score_mapping={"unsafe": 1.0, "safe": 0.0}
        )
        score = adapter._unsafe_score_from_generated_text("unsafe: hate speech")
        self.assertEqual(score, 1.0)

    def test_generated_text_line_index(self) -> None:
        adapter = self._make_adapter(generated_text_line_index=0)
        score = adapter._unsafe_score_from_generated_text("unsafe\nS1: Hate")
        self.assertEqual(score, 1.0)

    # ------------------------------------------------------------------
    # predict_batch
    # ------------------------------------------------------------------

    def test_predict_batch_unsafe(self) -> None:
        adapter = self._make_adapter()
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [_make_output("unsafe")]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            preds = adapter.predict_batch([_sample()], threshold=0.5)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].unsafe_score, 1.0)
        self.assertTrue(preds[0].unsafe_label)
        self.assertEqual(preds[0].metadata["adapter"], "vllm")
        self.assertEqual(preds[0].metadata["generated_text"], "unsafe")

    def test_predict_batch_safe(self) -> None:
        adapter = self._make_adapter()
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [_make_output("safe")]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            preds = adapter.predict_batch([_sample()], threshold=0.5)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].unsafe_score, 0.0)
        self.assertFalse(preds[0].unsafe_label)

    def test_predict_batch_empty(self) -> None:
        adapter = self._make_adapter()
        preds = adapter.predict_batch([], threshold=0.5)
        self.assertEqual(preds, [])

    def test_predict_batch_multiple_samples(self) -> None:
        """Batch with 4 samples of varying lengths."""
        adapter = self._make_adapter()
        samples = [
            _sample("s1", "short"),
            _sample("s2", "a slightly longer prompt here"),
            _sample("s3", "x"),
            _sample("s4", "another medium length prompt"),
        ]
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [
            _make_output("unsafe"),
            _make_output("safe"),
            _make_output("unsafe"),
            _make_output("safe"),
        ]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            preds = adapter.predict_batch(samples, threshold=0.5)
        self.assertEqual(len(preds), 4)
        self.assertEqual([p.unsafe_score for p in preds], [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(
            [p.sample_id for p in preds],
            ["s1", "s2", "s3", "s4"],
        )

    def test_predict_batch_with_text_score_mapping(self) -> None:
        adapter = self._make_adapter(
            text_score_mapping={"safe": 0.0, "unsafe": 1.0}
        )
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [
            _make_output("safe"),
            _make_output("unsafe"),
        ]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            preds = adapter.predict_batch(
                [_sample("s1"), _sample("s2")], threshold=0.5
            )
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].unsafe_score, 0.0)
        self.assertEqual(preds[1].unsafe_score, 1.0)

    def test_drop_failed_predictions(self) -> None:
        adapter = self._make_adapter(drop_failed_predictions=True)
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [
            _make_output("safe"),
            _make_output("xyzzy gibberish 42 99"),
        ]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            preds = adapter.predict_batch(
                [_sample("s1"), _sample("s2")], threshold=0.5
            )
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].sample_id, "s1")

    def test_output_count_mismatch_raises(self) -> None:
        adapter = self._make_adapter()
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [_make_output("safe")]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(adapter, "_sampling_params", return_value="params"),
        ):
            with self.assertRaises(ValueError) as ctx:
                adapter.predict_batch(
                    [_sample("s1"), _sample("s2")],
                    threshold=0.5,
                )
            self.assertIn("1 outputs", str(ctx.exception))
            self.assertIn("2 prompts", str(ctx.exception))

    # ------------------------------------------------------------------
    # Tokenizer fallback
    # ------------------------------------------------------------------

    def test_tokenizer_fallback_on_oserror(self) -> None:
        """_get_tokenizer falls back to PreTrainedTokenizerFast
        when AutoTokenizer raises OSError."""
        adapter = self._make_adapter()
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            OSError("tokenizer config missing")
        )
        fast_tok = MagicMock()
        mock_transformers.PreTrainedTokenizerFast.from_pretrained = (
            MagicMock(return_value=fast_tok)
        )
        with patch(
            "guard_eval_harness.models.vllm_adapter"
            ".importlib.import_module",
            return_value=mock_transformers,
        ):
            result = adapter._get_tokenizer()
        self.assertIs(result, fast_tok)

    def test_tokenizer_fallback_on_valueerror(self) -> None:
        """_get_tokenizer falls back to PreTrainedTokenizerFast
        when AutoTokenizer raises ValueError."""
        adapter = self._make_adapter()
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            ValueError("invalid tokenizer_class")
        )
        fast_tok = MagicMock()
        mock_transformers.PreTrainedTokenizerFast.from_pretrained = (
            MagicMock(return_value=fast_tok)
        )
        with patch(
            "guard_eval_harness.models.vllm_adapter"
            ".importlib.import_module",
            return_value=mock_transformers,
        ):
            result = adapter._get_tokenizer()
        self.assertIs(result, fast_tok)

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def test_get_llm_passes_config_args(self) -> None:
        adapter = self._make_adapter(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            dtype="float16",
            max_num_batched_tokens=8192,
        )
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertFalse(call_kwargs["trust_remote_code"])
        self.assertEqual(call_kwargs["tensor_parallel_size"], 2)
        self.assertEqual(call_kwargs["gpu_memory_utilization"], 0.8)
        self.assertEqual(call_kwargs["dtype"], "float16")
        self.assertEqual(call_kwargs["max_num_batched_tokens"], 8192)
        self.assertTrue(call_kwargs["disable_log_stats"])
        self.assertFalse(call_kwargs["use_tqdm_on_load"])

    def test_get_llm_honors_explicit_tqdm_on_load(self) -> None:
        adapter = self._make_adapter(use_tqdm_on_load=True)
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertTrue(call_kwargs["use_tqdm_on_load"])

    def test_get_llm_caches(self) -> None:
        adapter = self._make_adapter()
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            first = adapter._get_llm()
            second = adapter._get_llm()
        self.assertIs(first, second)
        self.assertEqual(mock_llm_cls.call_count, 1)

    def _mock_import(self, mock_vllm):
        """Return a side_effect for importlib.import_module."""
        mock_pooler_mod = MagicMock()

        def _import(name):
            if name == "vllm":
                return mock_vllm
            if "pooler" in name:
                return mock_pooler_mod
            return MagicMock()

        return _import

    def test_get_llm_passes_convert_classify_for_classification(
        self,
    ) -> None:
        """Without registry_patches, convert='classify' is set."""
        adapter = self._make_adapter(task="text-classification")
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertEqual(call_kwargs["convert"], "classify")

    def test_get_llm_skips_convert_with_registry_patches(
        self,
    ) -> None:
        """With registry_patches, convert is NOT set."""
        adapter = self._make_adapter(
            task="text-classification",
            registry_patches=["some.patch"],
        )
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertNotIn("convert", call_kwargs)

    def test_get_llm_classification_does_not_default_to_eager(self) -> None:
        adapter = self._make_adapter(task="text-classification")
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertNotIn("enforce_eager", call_kwargs)

    def test_get_llm_classification_allows_eager_override(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            enforce_eager=False,
        )
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertIs(call_kwargs["enforce_eager"], False)

    def test_get_llm_classification_does_not_default_max_num_seqs(self) -> None:
        adapter = self._make_adapter(task="text-classification")
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertNotIn("max_num_seqs", call_kwargs)

    def test_get_llm_classification_allows_max_num_seqs_override(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            max_num_seqs=128,
        )
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch(
            "guard_eval_harness.models.vllm_adapter.importlib.import_module",
            side_effect=self._mock_import(mock_vllm),
        ):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertEqual(call_kwargs["max_num_seqs"], 128)

    def test_get_llm_no_convert_for_generation(self) -> None:
        adapter = self._make_adapter(task="text-generation")
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            adapter._get_llm()
        call_kwargs = mock_llm_cls.call_args[1]
        self.assertNotIn("convert", call_kwargs)

    def test_resolved_model_path_with_subfolder(self) -> None:
        adapter = self._make_adapter(model_subfolder="some-subfolder")
        self.assertEqual(
            adapter._resolved_model_path(),
            "test-model/some-subfolder",
        )

    def test_resolved_model_path_without_subfolder(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter._resolved_model_path(), "test-model")

    # ------------------------------------------------------------------
    # Classification: capabilities
    # ------------------------------------------------------------------

    def test_capabilities_classification(self) -> None:
        adapter = self._make_adapter(task="text-classification")
        caps = adapter.capabilities
        self.assertTrue(caps.probability_scores)
        self.assertTrue(caps.batching)

    def test_classification_pooler_defaults_to_raw_logits(self) -> None:
        adapter = self._make_adapter(task="text-classification")

        self.assertEqual(
            adapter._classification_pooler_config(),
            {"pooling_type": "LAST", "use_activation": False},
        )

    # ------------------------------------------------------------------
    # Classification: score from logits
    # ------------------------------------------------------------------

    def test_unsafe_score_from_logits_defaults_to_softmax_for_two_labels(
        self,
    ) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            label_names=["safe", "unsafe"],
        )
        # Match HF raw-sequence-classification: absent an explicit activation,
        # two-label classifiers use softmax, not independent sigmoid scores.
        score = adapter._unsafe_score_from_logits([2.0, -2.0])
        self.assertAlmostEqual(score, 0.0180, places=3)

    def test_unsafe_score_from_logits_defaults_to_sigmoid_for_one_label(
        self,
    ) -> None:
        adapter = self._make_adapter(task="text-classification")
        score = adapter._unsafe_score_from_logits([2.0])
        self.assertAlmostEqual(score, 0.8807, places=3)

    def test_unsafe_score_from_logits_defaults_to_sigmoid_for_multilabel(
        self,
    ) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            label_names=["hate", "violence", "safe"],
            label_score_aggregation="max",
        )
        model_config = SimpleNamespace(
            problem_type="multi_label_classification",
            num_labels=3,
        )

        with patch.object(adapter, "_get_hf_config", return_value=model_config):
            score = adapter._unsafe_score_from_logits([1.0, 2.0, -1.0])

        self.assertAlmostEqual(score, 0.8808, places=3)

    def test_unsafe_score_from_logits_aggregated_labels_use_sigmoid(
        self,
    ) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            label_names=["LABEL_0", "LABEL_1", "LABEL_2"],
            label_score_aggregation="max",
        )
        model_config = SimpleNamespace(problem_type=None, num_labels=3)

        with patch.object(adapter, "_get_hf_config", return_value=model_config):
            score = adapter._unsafe_score_from_logits([1.0, 2.0, -1.0])

        self.assertAlmostEqual(score, 0.8808, places=3)

    def test_unsafe_score_from_logits_sigmoid(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            activation="sigmoid",
            label_score_aggregation="max",
        )
        # Two unlabeled logits: sigmoid(2.0) ≈ 0.88, sigmoid(-2.0) ≈ 0.12
        score = adapter._unsafe_score_from_logits([2.0, -2.0])
        self.assertAlmostEqual(score, 0.8807, places=3)

    def test_unsafe_score_from_logits_softmax_safe_inversion(
        self,
    ) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            activation="softmax",
            label_names=["safe", "unsafe"],
        )
        # softmax([3.0, -1.0]) → safe ≈ 0.982, unsafe ≈ 0.018
        score = adapter._unsafe_score_from_logits([3.0, -1.0])
        self.assertAlmostEqual(score, 0.0180, places=3)

    def test_unsafe_score_aggregation_max(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            activation="sigmoid",
            label_names=["hate", "violence", "safe"],
            label_score_aggregation="max",
        )
        # sigmoid([1.0, 2.0, -1.0]) ≈ [0.731, 0.881, 0.269]
        # hate and violence are unsafe → max(0.731, 0.881) = 0.881
        score = adapter._unsafe_score_from_logits([1.0, 2.0, -1.0])
        self.assertAlmostEqual(score, 0.8808, places=3)

    # ------------------------------------------------------------------
    # Classification: predict_batch
    # ------------------------------------------------------------------

    def test_predict_batch_classification(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            activation="sigmoid",
        )
        fake_llm = MagicMock()
        # The default classification pooler returns raw logits, so the
        # adapter applies the configured sigmoid activation.
        fake_llm.classify.return_value = [
            SimpleNamespace(outputs=SimpleNamespace(probs=[3.0, -2.0])),
            SimpleNamespace(outputs=SimpleNamespace(probs=[-3.0, -2.0])),
        ]
        with patch.object(adapter, "_get_llm", return_value=fake_llm):
            preds = adapter.predict_batch(
                [_sample("s1"), _sample("s2")], threshold=0.5
            )
        self.assertEqual(len(preds), 2)
        self.assertTrue(preds[0].unsafe_label)
        self.assertFalse(preds[1].unsafe_label)
        self.assertEqual(preds[0].metadata["task"], "text-classification")

    def test_predict_batch_classification_calls_classify(
        self,
    ) -> None:
        """Verify classify() is called, not generate()."""
        adapter = self._make_adapter(task="text-classification")
        fake_llm = MagicMock()
        fake_llm.classify.return_value = [
            SimpleNamespace(outputs=SimpleNamespace(probs=[0.0])),
        ]
        with patch.object(adapter, "_get_llm", return_value=fake_llm):
            adapter.predict_batch([_sample()], threshold=0.5)
        fake_llm.classify.assert_called_once()
        fake_llm.generate.assert_not_called()

    def test_pooler_config_zero_disables_activation(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            pooler_config={"use_activation": 0},
        )
        fake_llm = MagicMock()
        fake_llm.classify.return_value = [
            SimpleNamespace(outputs=SimpleNamespace(probs=[3.0, -1.0])),
        ]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(
                adapter,
                "_unsafe_score_from_logits",
                return_value=0.1,
            ) as mock_score,
        ):
            adapter.predict_batch([_sample()], threshold=0.5)
        self.assertFalse(mock_score.call_args.kwargs["already_activated"])

    def test_pooler_config_false_string_disables_activation(self) -> None:
        adapter = self._make_adapter(
            task="text-classification",
            pooler_config={"normalize": "false"},
        )
        fake_llm = MagicMock()
        fake_llm.classify.return_value = [
            SimpleNamespace(outputs=SimpleNamespace(probs=[3.0, -1.0])),
        ]
        with (
            patch.object(adapter, "_get_llm", return_value=fake_llm),
            patch.object(
                adapter,
                "_unsafe_score_from_logits",
                return_value=0.1,
            ) as mock_score,
        ):
            adapter.predict_batch([_sample()], threshold=0.5)
        self.assertFalse(mock_score.call_args.kwargs["already_activated"])

    # ------------------------------------------------------------------
    # Registry patches
    # ------------------------------------------------------------------

    def test_registry_patches_applied(self) -> None:
        adapter = self._make_adapter(
            registry_patches=[
                "some.fake.registry_patch",
            ]
        )
        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock(LLM=mock_llm_cls)
        with (
            patch.dict("sys.modules", {"vllm": mock_vllm}),
            patch(
                "guard_eval_harness.models.vllm_adapter.importlib.import_module"
            ) as mock_import,
        ):
            mock_import.return_value = MagicMock()
            adapter._get_llm()
        # Should have been called with the patch module
        mock_import.assert_any_call("some.fake.registry_patch")


if __name__ == "__main__":
    unittest.main()
