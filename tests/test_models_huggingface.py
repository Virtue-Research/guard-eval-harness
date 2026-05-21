"""Tests for the Hugging Face adapter."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.models.huggingface import HuggingFaceAdapter
from guard_eval_harness.models.templates import score_from_text
from guard_eval_harness.schemas import NormalizedSample
from guard_eval_harness.models.templates import MD_JUDGE_PROMPT


class HuggingFaceAdapterTest(unittest.TestCase):
    """Validate prompt rendering and score normalization."""

    def test_predict_batch_renders_prompt_and_scores_labels(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"prompt_template": "{dataset}::{messages_text}"},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        captured_prompts: list[list[str]] = []

        def fake_backend(prompts):
            captured_prompts.append(list(prompts))
            return [{"label": "SAFE", "score": 0.8}]

        with patch.object(adapter, "_run_backend", side_effect=fake_backend):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(captured_prompts, [["demo::user: Check this"]])
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0].unsafe_score, 0.2)
        self.assertFalse(predictions[0].unsafe_label)
        self.assertEqual(predictions[0].metadata["adapter"], "hf")

    def test_capabilities_report_batching_support(self) -> None:
        config = ResolvedModelConfig(adapter="hf", model_name="demo-hf")
        adapter = HuggingFaceAdapter.from_config(config)

        self.assertTrue(adapter.capabilities.batching)
        self.assertTrue(adapter.capabilities.probability_scores)
        self.assertTrue(adapter.capabilities.supports_category_outputs)

    def test_generated_text_output_maps_to_binary_score(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "text-generation"},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"generated_text": "Safety: Unsafe\nCategories: Hate"}
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].unsafe_score, 1.0)
        self.assertTrue(predictions[0].unsafe_label)

    def test_generated_text_respects_explicit_text_score_mapping(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text2text-generation",
                "text_score_mapping": {"positive": 1.0, "negative": 0.0},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[{"generated_text": "negative"}],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].unsafe_score, 0.0)
        self.assertFalse(predictions[0].unsafe_label)

    def test_text_score_mapping_accepts_yes_no_with_trailing_reasoning(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-generation",
                "text_score_mapping": {"yes": 1.0, "no": 0.0},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"generated_text": "Yes, it violates policy because ..."}
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].unsafe_score, 1.0)
        self.assertTrue(predictions[0].unsafe_label)

    def test_text_score_mapping_accepts_unique_truncated_prefix(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-generation",
                "text_score_mapping": {"safe": 0.0, "unsafe": 1.0},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        self.assertEqual(adapter._unsafe_score_from_output("saf"), 0.0)
        self.assertEqual(adapter._unsafe_score_from_output("unsa"), 1.0)

    def test_label_score_mapping_supports_score_and_inverse(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "label_score_mapping": {
                    "safe": "1-score",
                    "unsafe": "score",
                }
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"label": "SAFE", "score": 0.8},
                {"label": "UNSAFE", "score": 0.9},
            ],
        ):
            predictions = adapter.predict_batch(
                [sample, sample.model_copy(update={"id": "sample-2"})],
                threshold=0.5,
            )

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.2)
        self.assertAlmostEqual(predictions[1].unsafe_score, 0.9)
        self.assertTrue(predictions[1].unsafe_label)

    def test_label_score_mapping_is_case_insensitive(self) -> None:
        """Capitalized YAML keys like 'Safe'/'Unsafe' must match model output."""
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "label_score_mapping": {
                    "Safe": "1-score",
                    "Unsafe": "score",
                }
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[{"label": "SAFE", "score": 0.8}],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.2)
        self.assertFalse(predictions[0].unsafe_label)

    def test_label_score_mapping_can_select_unsafe_score(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "label_score_mapping": {
                    "unsafe": "unsafe_score",
                }
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"label": "UNSAFE", "score": 0.1, "unsafe_score": 0.8}
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.8)
        self.assertTrue(predictions[0].unsafe_label)

    def test_text2text_generation_does_not_set_return_full_text(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "text2text-generation"},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        def fake_backend(prompts, **kwargs):
            self.assertNotIn("return_full_text", kwargs)
            return [{"generated_text": "positive"}]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            outputs = adapter._run_backend(["hello"])

        self.assertEqual(outputs, [{"generated_text": "positive"}])

    def test_get_tokenizer_passes_use_fast_when_configured(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"use_fast": False},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        auto_tokenizer = SimpleNamespace(
            from_pretrained=unittest.mock.Mock(return_value=object())
        )
        transformers = SimpleNamespace(AutoTokenizer=auto_tokenizer)

        with patch(
            "importlib.import_module",
            return_value=transformers,
        ):
            adapter._get_tokenizer()

        auto_tokenizer.from_pretrained.assert_called_once()
        _, kwargs = auto_tokenizer.from_pretrained.call_args
        self.assertFalse(kwargs["use_fast"])

    def test_backend_does_not_force_pipeline_batch_size_by_default(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "text-generation"},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        observed_kwargs: dict[str, object] = {}

        def fake_backend(prompts, **kwargs):
            observed_kwargs.update(kwargs)
            return [{"generated_text": "safe"} for _ in prompts]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            adapter._run_backend(["a", "b", "c", "d"])

        self.assertNotIn("batch_size", observed_kwargs)

    def test_backend_respects_explicit_pipeline_batch_size(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-generation",
                "pipeline_batch_size": 7,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        observed_kwargs: dict[str, object] = {}

        def fake_backend(prompts, **kwargs):
            observed_kwargs.update(kwargs)
            return [{"generated_text": "safe"} for _ in prompts]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            adapter._run_backend(["a", "b", "c"])

        self.assertEqual(observed_kwargs.get("batch_size"), 7)

    def test_backend_skips_explicit_generation_batching_without_pad_token(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "text-generation", "pipeline_batch_size": 7},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        observed_kwargs: dict[str, object] = {}
        fake_backend = unittest.mock.Mock(
            return_value=[
                {"generated_text": "safe"},
                {"generated_text": "safe"},
            ]
        )
        fake_backend.tokenizer = SimpleNamespace(pad_token_id=None)

        def capture_backend(prompts, **kwargs):
            observed_kwargs.update(kwargs)
            return fake_backend(prompts, **kwargs)

        capture_backend.tokenizer = fake_backend.tokenizer

        with patch.object(
            adapter, "_get_backend", return_value=capture_backend
        ):
            adapter._run_backend(["a", "b"])

        self.assertNotIn("batch_size", observed_kwargs)

    def test_predict_batch_allows_text_classification_top_k_label_lists(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "label_score_mapping": {
                    "safe": "1-score",
                    "unsafe": "score",
                },
                "generation_kwargs": {"top_k": 2},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        def fake_backend(prompts, **kwargs):
            self.assertEqual(kwargs.get("top_k"), 2)
            return [
                [
                    {"label": "SAFE", "score": 0.8},
                    {"label": "UNSAFE", "score": 0.2},
                ]
            ]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.2)
        self.assertFalse(predictions[0].unsafe_label)
        self.assertEqual(predictions[0].metadata["label"], "SAFE")
        self.assertAlmostEqual(predictions[0].metadata["score"], 0.8)

    def test_predict_batch_rejects_nested_multi_candidate_backend_outputs(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-generation",
                "generation_kwargs": {"num_return_sequences": 2},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        def fake_backend(prompts, **kwargs):
            self.assertEqual(kwargs.get("num_return_sequences"), 2)
            return [
                [
                    {"generated_text": "safe"},
                    {"generated_text": "unsafe"},
                ]
            ]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            with self.assertRaisesRegex(
                ValueError,
                "returned 2 candidates for input sample 1",
            ):
                adapter.predict_batch([sample], threshold=0.5)

    def test_run_backend_unwraps_single_nested_candidate(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "text-generation"},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        def fake_backend(prompts, **kwargs):
            self.assertFalse(kwargs.get("do_sample"))
            return [[{"generated_text": "positive"}]]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            outputs = adapter._run_backend(["hello"])

        self.assertEqual(outputs, [{"generated_text": "positive"}])

    def test_generated_text_can_extract_a_specific_line(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-generation",
                "generated_text_line_index": 1,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"generated_text": (" Safe\nSafety classifier: unsafe\nSafety")}
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertEqual(predictions[0].unsafe_score, 1.0)
        self.assertTrue(predictions[0].unsafe_label)

    def test_predict_batch_rejects_surplus_backend_outputs(self) -> None:
        config = ResolvedModelConfig(adapter="hf", model_name="demo-hf")
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                {"label": "SAFE", "score": 0.9},
                {"label": "UNSAFE", "score": 0.9},
            ],
        ):
            with self.assertRaisesRegex(
                ValueError,
                "returned 2 outputs for 1 input samples",
            ):
                adapter.predict_batch([sample], threshold=0.5)

    def test_predict_batch_rejects_missing_backend_outputs(self) -> None:
        config = ResolvedModelConfig(adapter="hf", model_name="demo-hf")
        adapter = HuggingFaceAdapter.from_config(config)
        samples = [
            NormalizedSample(
                id="sample-1",
                dataset="demo",
                split="test",
                messages=[{"role": "user", "content": "Check this"}],
                label={"unsafe": True},
            ),
            NormalizedSample(
                id="sample-2",
                dataset="demo",
                split="test",
                messages=[{"role": "user", "content": "Check this too"}],
                label={"unsafe": True},
            ),
        ]

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[{"label": "SAFE", "score": 0.9}],
        ):
            with self.assertRaisesRegex(
                ValueError,
                "returned 1 outputs for 2 input samples",
            ):
                adapter.predict_batch(samples, threshold=0.5)

    def test_label_score_aggregation_can_filter_to_specific_labels(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-classification",
                "label_score_aggregation": "max",
                "score_labels": ["toxicity", "insult"],
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                [
                    {"label": "male", "score": 0.99},
                    {"label": "toxicity", "score": 0.2},
                    {"label": "insult", "score": 0.4},
                ]
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.4)
        self.assertFalse(predictions[0].unsafe_label)

    def test_prompt_template_uses_default_user_target_role(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"prompt_template": MD_JUDGE_PROMPT},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        prompt = adapter._prompt_for_sample(sample)

        self.assertIn("unsafe content in 'User' messages", prompt)
        self.assertIn("assessment for User", prompt)
        self.assertNotIn("{metadata_target_role}", prompt)

    def test_prompt_template_respects_assistant_target_role_metadata(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"prompt_template": MD_JUDGE_PROMPT},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "user", "content": "Prompt"},
                {"role": "assistant", "content": "Response"},
            ],
            label={"unsafe": True},
            metadata={"target_role": "assistant"},
        )

        prompt = adapter._prompt_for_sample(sample)

        self.assertIn("unsafe content in 'Assistant' messages", prompt)
        self.assertIn("assessment for Assistant", prompt)

    def test_prompt_for_sample_uses_llama_guard_chat_shaping(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="meta-llama/Llama-Guard-3-1B",
            args={
                "apply_chat_template": True,
                "chat_template_profile": "llama_guard",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )
        tokenizer = SimpleNamespace(
            chat_template="selectattr('type', 'equalto', 'text')",
            apply_chat_template=unittest.mock.Mock(return_value="prompt"),
        )

        with patch.object(adapter, "_get_tokenizer", return_value=tokenizer):
            prompt = adapter._prompt_for_sample(sample)

        self.assertEqual(prompt, "prompt")
        tokenizer.apply_chat_template.assert_called_once()
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Check this"}],
                }
            ],
        )

    def test_prompt_for_sample_uses_adjacent_granite_guardian_response_pair(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="ibm-granite/granite-guardian-3.1-8b",
            args={
                "apply_chat_template": True,
                "chat_template_profile": "granite_guardian",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "user", "content": "Original prompt"},
                {"role": "assistant", "content": "Original response"},
                {"role": "user", "content": "Unanswered follow-up"},
            ],
            label={"unsafe": True},
            metadata={"target_role": "assistant"},
        )
        tokenizer = SimpleNamespace(
            apply_chat_template=unittest.mock.Mock(return_value="prompt"),
        )

        with patch.object(adapter, "_get_tokenizer", return_value=tokenizer):
            prompt = adapter._prompt_for_sample(sample)

        self.assertEqual(prompt, "prompt")
        tokenizer.apply_chat_template.assert_called_once()
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Original prompt"},
                {"role": "assistant", "content": "Original response"},
            ],
        )

    def test_prompt_for_sample_falls_back_when_granite_response_is_unpaired(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="ibm-granite/granite-guardian-3.1-8b",
            args={
                "apply_chat_template": True,
                "chat_template_profile": "granite_guardian",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[
                {"role": "user", "content": "Prompt"},
                {"role": "assistant", "content": "First response"},
                {"role": "assistant", "content": "Latest response"},
            ],
            label={"unsafe": True},
            metadata={"target_role": "assistant"},
        )
        tokenizer = SimpleNamespace(
            apply_chat_template=unittest.mock.Mock(return_value="prompt"),
        )

        with patch.object(adapter, "_get_tokenizer", return_value=tokenizer):
            prompt = adapter._prompt_for_sample(sample)

        self.assertEqual(prompt, "prompt")
        tokenizer.apply_chat_template.assert_called_once()
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Latest response"},
            ],
        )

    def test_backend_uses_subfolder_model_and_tokenizer(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-classification",
                "model_subfolder": "classifier-subdir",
                "tokenizer_subfolder": "tokenizer-subdir",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        auto_tokenizer = SimpleNamespace()
        auto_tokenizer.from_pretrained = unittest.mock.Mock(return_value="tok")
        auto_model = SimpleNamespace()
        auto_model.from_pretrained = unittest.mock.Mock(return_value="model")
        pipeline = unittest.mock.Mock(return_value="backend")
        transformers = SimpleNamespace(
            AutoTokenizer=auto_tokenizer,
            AutoModelForSequenceClassification=auto_model,
            AutoModelForSeq2SeqLM=SimpleNamespace(),
            AutoModelForCausalLM=SimpleNamespace(),
            pipeline=pipeline,
        )

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            return_value=transformers,
        ):
            backend = adapter._get_backend()

        self.assertEqual(backend, "backend")
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "demo-hf",
            trust_remote_code=False,
            revision="main",
            subfolder="tokenizer-subdir",
        )
        auto_model.from_pretrained.assert_called_once_with(
            "demo-hf",
            trust_remote_code=False,
            revision="main",
            subfolder="classifier-subdir",
        )
        call_kwargs = pipeline.call_args
        self.assertEqual(call_kwargs[1]["task"], "text-classification")
        self.assertEqual(call_kwargs[1]["model"], "model")
        self.assertEqual(call_kwargs[1]["tokenizer"], "tok")
        self.assertIn("device", call_kwargs[1])

    def test_backend_omits_pipeline_device_with_device_map(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-classification",
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        pipeline = unittest.mock.Mock(return_value="backend")
        transformers = SimpleNamespace(pipeline=pipeline)
        torch = SimpleNamespace(float16="float16")

        def fake_import(name: str):
            if name == "transformers":
                return transformers
            if name == "torch":
                return torch
            raise AssertionError(name)

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            side_effect=fake_import,
        ):
            backend = adapter._get_backend()

        self.assertEqual(backend, "backend")
        pipeline.assert_called_once()
        self.assertNotIn("device", pipeline.call_args.kwargs)
        self.assertEqual(
            pipeline.call_args.kwargs["model_kwargs"],
            {"device_map": "auto", "torch_dtype": "float16"},
        )

    def test_backend_omits_pipeline_device_with_model_kwargs_device_map(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-classification",
                "model_kwargs": {"device_map": "auto"},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        pipeline = unittest.mock.Mock(return_value="backend")
        transformers = SimpleNamespace(pipeline=pipeline)

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            return_value=transformers,
        ):
            backend = adapter._get_backend()

        self.assertEqual(backend, "backend")
        pipeline.assert_called_once()
        self.assertNotIn("device", pipeline.call_args.kwargs)
        self.assertEqual(
            pipeline.call_args.kwargs["model_kwargs"],
            {"device_map": "auto"},
        )

    def test_subfolder_backend_omits_pipeline_device_with_device_map(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "text-classification",
                "model_subfolder": "classifier-subdir",
                "tokenizer_subfolder": "tokenizer-subdir",
                "device_map": "auto",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        auto_tokenizer = SimpleNamespace()
        auto_tokenizer.from_pretrained = unittest.mock.Mock(return_value="tok")
        auto_model = SimpleNamespace()
        auto_model.from_pretrained = unittest.mock.Mock(return_value="model")
        pipeline = unittest.mock.Mock(return_value="backend")
        transformers = SimpleNamespace(
            AutoTokenizer=auto_tokenizer,
            AutoModelForSequenceClassification=auto_model,
            AutoModelForSeq2SeqLM=SimpleNamespace(),
            AutoModelForCausalLM=SimpleNamespace(),
            pipeline=pipeline,
        )

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            return_value=transformers,
        ):
            backend = adapter._get_backend()

        self.assertEqual(backend, "backend")
        auto_model.from_pretrained.assert_called_once_with(
            "demo-hf",
            trust_remote_code=False,
            revision="main",
            subfolder="classifier-subdir",
            device_map="auto",
        )
        pipeline.assert_called_once()
        self.assertNotIn("device", pipeline.call_args.kwargs)

    def test_raw_sequence_classification_task_uses_raw_backend(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"task": "raw-sequence-classification"},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with (
            patch.object(adapter, "_run_backend") as run_backend,
            patch.object(
                adapter,
                "_run_raw_sequence_classification",
                return_value=[
                    {
                        "unsafe_score": 0.91,
                        "top_label": "LABEL_0",
                        "top_label_score": 0.91,
                    }
                ],
            ),
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        run_backend.assert_not_called()
        self.assertEqual(predictions[0].unsafe_score, 0.91)
        self.assertTrue(predictions[0].unsafe_label)
        self.assertEqual(predictions[0].metadata["top_label"], "LABEL_0")

    def test_raw_sequence_classification_normalizes_scores(self) -> None:
        class FakeTensor:
            def __init__(self, values) -> None:
                self._values = values

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._values

        class FakeEncoded:
            def __init__(self) -> None:
                self.device = None

            def to(self, device):
                self.device = device
                return self

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_softmax(logits, dim=-1):
            self.assertEqual(dim, -1)
            normalized = []
            for row in logits.tolist():
                exponents = [math.exp(value) for value in row]
                denominator = sum(exponents)
                normalized.append([value / denominator for value in exponents])
            return FakeTensor(normalized)

        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "activation": "softmax",
                "label_names": ["safe", "unsafe"],
                "include_label_scores": True,
                "max_length": 32,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        encoded = FakeEncoded()
        tokenizer = unittest.mock.Mock(return_value={"input_ids": encoded})
        raw_logits = [0.0, math.log(9.0)]
        model = unittest.mock.Mock(
            return_value=SimpleNamespace(logits=FakeTensor([raw_logits]))
        )
        model.config = SimpleNamespace(
            problem_type="single_label_classification",
            num_labels=2,
        )
        torch = SimpleNamespace(
            no_grad=lambda: FakeNoGrad(),
            softmax=fake_softmax,
        )

        with (
            patch.object(adapter, "_get_tokenizer", return_value=tokenizer),
            patch.object(
                adapter, "_get_raw_model", return_value=(model, "cpu")
            ),
            patch(
                "guard_eval_harness.models.huggingface.importlib.import_module",
                return_value=torch,
            ),
        ):
            outputs = adapter._run_raw_sequence_classification(["hello"])

        tokenizer.assert_called_once_with(
            ["hello"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32,
        )
        self.assertEqual(encoded.device, "cpu")
        self.assertEqual(outputs[0]["top_label"], "unsafe")
        self.assertAlmostEqual(outputs[0]["unsafe_score"], 0.9)
        self.assertAlmostEqual(outputs[0]["top_label_score"], 0.9)
        self.assertAlmostEqual(outputs[0]["label_scores"]["safe"], 0.1)
        self.assertAlmostEqual(outputs[0]["label_scores"]["unsafe"], 0.9)

    def test_raw_sequence_classification_defaults_to_softmax(self) -> None:
        class FakeTensor:
            def __init__(self, values) -> None:
                self._values = values

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._values

        class FakeEncoded:
            def to(self, device):
                return self

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_softmax(logits, dim=-1):
            self.assertEqual(dim, -1)
            normalized = []
            for row in logits.tolist():
                exponents = [math.exp(value) for value in row]
                denominator = sum(exponents)
                normalized.append([value / denominator for value in exponents])
            return FakeTensor(normalized)

        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "label_names": ["safe", "unsafe"],
                "include_label_scores": True,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        tokenizer = unittest.mock.Mock(
            return_value={"input_ids": FakeEncoded()}
        )
        model = unittest.mock.Mock(
            return_value=SimpleNamespace(logits=FakeTensor([[2.0, 1.0]]))
        )
        model.config = SimpleNamespace(
            problem_type="single_label_classification",
            num_labels=2,
        )
        torch = SimpleNamespace(
            no_grad=lambda: FakeNoGrad(),
            softmax=fake_softmax,
        )

        with (
            patch.object(adapter, "_get_tokenizer", return_value=tokenizer),
            patch.object(
                adapter, "_get_raw_model", return_value=(model, "cpu")
            ),
            patch(
                "guard_eval_harness.models.huggingface.importlib.import_module",
                return_value=torch,
            ),
        ):
            outputs = adapter._run_raw_sequence_classification(["hello"])

        self.assertAlmostEqual(outputs[0]["unsafe_score"], 0.2689414213699951)
        self.assertEqual(outputs[0]["top_label"], "safe")
        self.assertAlmostEqual(
            outputs[0]["top_label_score"],
            0.7310585786300049,
        )

    def test_raw_sequence_classification_inverts_safe_only_sigmoid_head(
        self,
    ) -> None:
        class FakeTensor:
            def __init__(self, values) -> None:
                self._values = values

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._values

        class FakeEncoded:
            def to(self, device):
                return self

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_sigmoid(logits):
            normalized = []
            for row in logits.tolist():
                normalized.append(
                    [1.0 / (1.0 + math.exp(-value)) for value in row]
                )
            return FakeTensor(normalized)

        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "label_names": ["safe"],
                "include_label_scores": True,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        tokenizer = unittest.mock.Mock(
            return_value={"input_ids": FakeEncoded()}
        )
        model = unittest.mock.Mock(
            return_value=SimpleNamespace(logits=FakeTensor([[math.log(9.0)]]))
        )
        model.config = SimpleNamespace(num_labels=1)
        torch = SimpleNamespace(
            no_grad=lambda: FakeNoGrad(),
            sigmoid=fake_sigmoid,
        )

        with (
            patch.object(adapter, "_get_tokenizer", return_value=tokenizer),
            patch.object(
                adapter, "_get_raw_model", return_value=(model, "cpu")
            ),
            patch(
                "guard_eval_harness.models.huggingface.importlib.import_module",
                return_value=torch,
            ),
        ):
            outputs = adapter._run_raw_sequence_classification(["hello"])

        self.assertAlmostEqual(outputs[0]["unsafe_score"], 0.1)
        self.assertEqual(outputs[0]["top_label"], "safe")
        self.assertAlmostEqual(outputs[0]["top_label_score"], 0.9)
        self.assertAlmostEqual(outputs[0]["label_scores"]["safe"], 0.9)

    def test_raw_sequence_classification_rejects_identity_activation(
        self,
    ) -> None:
        class FakeTensor:
            def __init__(self, values) -> None:
                self._values = values

            def tolist(self):
                return self._values

        class FakeEncoded:
            def to(self, device):
                return self

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "activation": "none",
                "label_names": ["safe", "unsafe"],
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        tokenizer = unittest.mock.Mock(
            return_value={"input_ids": FakeEncoded()}
        )
        model = unittest.mock.Mock(
            return_value=SimpleNamespace(logits=FakeTensor([[2.0, 1.0]]))
        )
        model.config = SimpleNamespace(
            problem_type="single_label_classification",
            num_labels=2,
        )
        torch = SimpleNamespace(no_grad=lambda: FakeNoGrad())

        with (
            patch.object(adapter, "_get_tokenizer", return_value=tokenizer),
            patch.object(
                adapter, "_get_raw_model", return_value=(model, "cpu")
            ),
            patch(
                "guard_eval_harness.models.huggingface.importlib.import_module",
                return_value=torch,
            ),
            self.assertRaisesRegex(
                ValueError,
                "requires normalized probabilities",
            ),
        ):
            adapter._run_raw_sequence_classification(["hello"])

    def test_raw_sequence_classification_combines_partial_label_mappings(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "activation": "softmax",
                "label_score_mapping": {"safe": 0.0},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        self.assertEqual(
            adapter._unsafe_score_from_label_scores(
                {"safe": 0.4, "harmful": 0.95}
            ),
            0.95,
        )

    def test_raw_sequence_classification_uses_subfolder_and_device_map(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "model_subfolder": "classifier-subdir",
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        raw_model = unittest.mock.Mock()
        raw_model.config = SimpleNamespace(id2label={0: "SAFE", 1: "UNSAFE"})
        auto_model = SimpleNamespace()
        auto_model.from_pretrained = unittest.mock.Mock(return_value=raw_model)
        transformers = SimpleNamespace(
            AutoModelForSequenceClassification=auto_model,
        )
        torch = SimpleNamespace(device=unittest.mock.Mock(), float16="float16")

        def fake_import(name: str):
            if name == "transformers":
                return transformers
            if name == "torch":
                return torch
            raise AssertionError(name)

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            side_effect=fake_import,
        ):
            model, device = adapter._get_raw_model()

        self.assertEqual(model, raw_model)
        self.assertIsNone(device)
        auto_model.from_pretrained.assert_called_once_with(
            "demo-hf",
            trust_remote_code=False,
            revision="main",
            subfolder="classifier-subdir",
            device_map="auto",
            torch_dtype="float16",
        )
        raw_model.to.assert_not_called()
        raw_model.eval.assert_called_once_with()

    def test_raw_sequence_classification_honors_model_kwargs_device_map(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "task": "raw-sequence-classification",
                "model_subfolder": "classifier-subdir",
                "model_kwargs": {"device_map": "auto"},
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)

        raw_model = unittest.mock.Mock()
        raw_model.config = SimpleNamespace(id2label={0: "SAFE"})
        auto_model = SimpleNamespace()
        auto_model.from_pretrained = unittest.mock.Mock(return_value=raw_model)
        transformers = SimpleNamespace(
            AutoModelForSequenceClassification=auto_model,
        )
        torch = SimpleNamespace(device=unittest.mock.Mock())

        def fake_import(name: str):
            if name == "transformers":
                return transformers
            if name == "torch":
                return torch
            raise AssertionError(name)

        with patch(
            "guard_eval_harness.models.huggingface.importlib.import_module",
            side_effect=fake_import,
        ):
            model, device = adapter._get_raw_model()

        self.assertEqual(model, raw_model)
        self.assertIsNone(device)
        auto_model.from_pretrained.assert_called_once_with(
            "demo-hf",
            trust_remote_code=False,
            revision="main",
            subfolder="classifier-subdir",
            device_map="auto",
        )
        raw_model.to.assert_not_called()
        raw_model.eval.assert_called_once_with()

    def test_multilabel_label_scores_aggregate_by_max(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "label_score_aggregation": "max",
                "label_score_mapping": {
                    "label_11": 0.0,
                    "label_7": "score",
                    "label_12": "score",
                },
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                [
                    {"label": "LABEL_11", "score": 0.80},
                    {"label": "LABEL_7", "score": 0.65},
                    {"label": "LABEL_12", "score": 0.15},
                ]
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.65)
        self.assertTrue(predictions[0].unsafe_label)

    def test_multilabel_scores_use_all_labels_when_no_safe_mapping_exists(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"label_score_aggregation": "max"},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Check this"}],
            label={"unsafe": True},
        )

        with patch.object(
            adapter,
            "_run_backend",
            return_value=[
                [
                    {"label": "LABEL_11", "score": 0.80},
                    {"label": "LABEL_7", "score": 0.65},
                    {"label": "LABEL_12", "score": 0.15},
                ]
            ],
        ):
            predictions = adapter.predict_batch([sample], threshold=0.5)

        self.assertAlmostEqual(predictions[0].unsafe_score, 0.80)
        self.assertTrue(predictions[0].unsafe_label)

    def test_multilabel_text_classification_forces_sigmoid(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"label_score_aggregation": "max"},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        def fake_backend(prompts, **kwargs):
            self.assertEqual(kwargs.get("function_to_apply"), "sigmoid")
            self.assertIsNone(kwargs.get("top_k"))
            return [
                [
                    {"label": "LABEL_0", "score": 0.2},
                    {"label": "LABEL_1", "score": 0.7},
                ]
            ]

        with patch.object(adapter, "_get_backend", return_value=fake_backend):
            outputs = adapter._run_backend(["hello"])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 2)

    def test_raw_sequence_classification_uses_max_aggregation(self) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"score_aggregation": "max"},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        self.assertEqual(
            adapter._aggregate_label_scores([0.1, 0.9, 0.2]),
            0.9,
        )

    def test_score_parser_handles_safe_and_unsafe_words(self) -> None:
        self.assertEqual(score_from_text("Safety: Safe"), 0.0)
        self.assertEqual(score_from_text("Safety: Controversial"), 0.5)
        self.assertEqual(score_from_text("Safety: Unsafe"), 1.0)

    def test_score_parser_prefers_numeric_values_over_keywords(self) -> None:
        self.assertEqual(score_from_text("unsafe score: 0.2"), 0.2)
        self.assertEqual(score_from_text("safe 80%"), 0.8)
        self.assertEqual(score_from_text("unsafe score: 0.2 (20%)"), 0.2)
        self.assertEqual(score_from_text("result 0.7"), 0.7)
        self.assertEqual(score_from_text("value=0.2"), 0.2)
        self.assertEqual(score_from_text("result 0.7."), 0.7)
        self.assertEqual(score_from_text("0.2."), 0.2)

    def test_score_parser_ignores_ordinal_numbers_near_labels(self) -> None:
        self.assertEqual(score_from_text("safe #1"), 0.0)
        self.assertEqual(score_from_text("1) safe"), 0.0)

    def test_predict_batch_drops_unparseable_samples_by_default(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"prompt_template": "{messages_text}"},
        )
        adapter = HuggingFaceAdapter.from_config(config)
        samples = [
            NormalizedSample(
                id="s1",
                dataset="d",
                split="test",
                messages=[{"role": "user", "content": "ok"}],
                label={"unsafe": False},
            ),
            NormalizedSample(
                id="s2",
                dataset="d",
                split="test",
                messages=[{"role": "user", "content": "bad"}],
                label={"unsafe": True},
            ),
        ]

        def fake_backend(prompts):
            return [
                {"label": "SAFE", "score": 0.9},
                {"generated_text": "UNPARSEABLE GIBBERISH"},
            ]

        with patch.object(adapter, "_run_backend", side_effect=fake_backend):
            predictions = adapter.predict_batch(samples, threshold=0.5)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].sample_id, "s1")
        self.assertTrue(adapter.allow_partial_predictions)

    def test_predict_batch_raises_when_drop_failed_disabled(
        self,
    ) -> None:
        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={
                "prompt_template": "{messages_text}",
                "drop_failed_predictions": False,
            },
        )
        adapter = HuggingFaceAdapter.from_config(config)
        sample = NormalizedSample(
            id="s1",
            dataset="d",
            split="test",
            messages=[{"role": "user", "content": "ok"}],
            label={"unsafe": False},
        )

        def fake_backend(prompts):
            return [{"generated_text": "UNPARSEABLE GIBBERISH"}]

        with patch.object(adapter, "_run_backend", side_effect=fake_backend):
            with self.assertRaises(ValueError):
                adapter.predict_batch([sample], threshold=0.5)

        self.assertFalse(adapter.allow_partial_predictions)


    def test_empty_score_rows_raises_descriptive_error(self) -> None:
        """Empty model output raises ValueError, not IndexError."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        config = ResolvedModelConfig(
            adapter="hf",
            model_name="demo-hf",
            args={"label_score_mapping": {"safe": 0.0, "unsafe": 1.0}},
        )
        adapter = HuggingFaceAdapter.from_config(config)

        empty_logits = torch.empty((0, 2), dtype=torch.float32)

        class _FakeModel:
            config = SimpleNamespace(
                id2label={0: "safe", 1: "unsafe"},
                is_encoder_decoder=True,
            )

            def eval(self):
                return None

            def __call__(self, **kwargs):
                return SimpleNamespace(logits=empty_logits)

        fake_model = _FakeModel()
        class _FakeTokenizer:
            padding_side = "right"
            pad_token_id = 0
            cls_token = "[CLS]"

            def __call__(self, *a, **kw):
                return {
                    "input_ids": torch.zeros(
                        (1, 1), dtype=torch.long
                    ),
                    "attention_mask": torch.ones(
                        (1, 1), dtype=torch.long
                    ),
                }

        fake_tokenizer = _FakeTokenizer()

        with (
            patch.object(
                adapter,
                "_get_tokenizer",
                return_value=fake_tokenizer,
            ),
            patch.object(
                adapter,
                "_get_raw_model",
                return_value=(fake_model, None),
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                adapter._run_raw_sequence_classification(
                    ["hello"],
                )
            self.assertIn("empty logits", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
