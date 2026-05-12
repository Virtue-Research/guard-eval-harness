"""Contract tests for shared schemas."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from guard_eval_harness.config.models import (
    ResolvedExecutionConfig,
    ResolvedModelConfig,
    ResolvedOutputConfig,
)
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    DatasetMetadata,
    MediaPart,
    MediaRef,
    Message,
    NormalizedPrediction,
    NormalizedSample,
    RunEnvironment,
    RunManifest,
    TextPart,
)


class SchemaContractsTest(unittest.TestCase):
    """Validate stable schema contracts."""

    def test_message_rejects_empty_content(self) -> None:
        with self.assertRaises(ValidationError):
            Message(role="user", content="   ")

    def test_message_allows_empty_assistant_content(self) -> None:
        message = Message(role="assistant", content="")

        self.assertEqual(message.content, "")

    def test_normalized_sample_uses_messages_contract(self) -> None:
        sample = NormalizedSample(
            id="sample-1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": False},
        )
        self.assertEqual(sample.messages[0].role, "user")
        self.assertFalse(sample.label.unsafe)

    def test_prediction_requires_canonical_threshold_alignment(self) -> None:
        with self.assertRaises(ValidationError):
            NormalizedPrediction(
                sample_id="sample-1",
                unsafe_score=0.9,
                unsafe_label=False,
                threshold=0.5,
            )

    def test_dataset_metadata_validates_counts(self) -> None:
        with self.assertRaises(ValidationError):
            DatasetMetadata(
                name="demo",
                display_name="Demo",
                sample_count=1,
                unsafe_count=2,
            )

    def test_run_manifest_locks_shared_nested_types(self) -> None:
        manifest = RunManifest(
            tool_version="0.1.0",
            run_name="demo",
            run_dir="/tmp/demo",
            status="completed",
            started_at="2026-03-17T00:00:00+00:00",
            finished_at="2026-03-17T00:01:00+00:00",
            resolved_config_sha256="abc123",
            model=ResolvedModelConfig(adapter="mock"),
            execution=ResolvedExecutionConfig(batch_size=1),
            output=ResolvedOutputConfig(run_dir="/tmp/demo"),
            threshold=0.5,
            datasets=[
                DatasetMetadata(
                    name="demo",
                    display_name="Demo",
                    sample_count=1,
                    unsafe_count=0,
                )
            ],
            adapter_capabilities=AdapterCapabilities(
                adapter_name="mock",
                probability_scores=True,
                batching=True,
                concurrency=False,
                cost_estimation=False,
                token_accounting=False,
            ),
            environment=RunEnvironment(
                python_version="3.10.12",
                platform="linux",
                hostname="localhost",
            ),
        )
        self.assertEqual(manifest.datasets[0].name, "demo")


class MultimodalSchemaTest(unittest.TestCase):
    """Validate multimodal schema extensions."""

    def test_message_str_content_still_works(self) -> None:
        msg = Message(role="user", content="Hello")
        self.assertEqual(msg.content, "Hello")
        self.assertEqual(msg.text_content, "Hello")
        self.assertEqual(msg.image_refs, [])
        self.assertEqual(msg.audio_refs, [])
        self.assertEqual(msg.media_refs, [])

    def test_message_list_content_with_text_only(self) -> None:
        msg = Message(
            role="user",
            content=[TextPart(text="Hello")],
        )
        self.assertIsInstance(msg.content, list)
        self.assertEqual(msg.text_content, "Hello")

    def test_message_list_content_with_image(self) -> None:
        ref = MediaRef(modality="image", uri="/tmp/test.jpg")
        msg = Message(
            role="user",
            content=[
                TextPart(text="Describe this"),
                MediaPart(media=ref),
            ],
        )
        self.assertEqual(msg.text_content, "Describe this")
        self.assertEqual(len(msg.image_refs), 1)
        self.assertEqual(msg.image_refs[0].uri, "/tmp/test.jpg")
        self.assertEqual(len(msg.media_refs), 1)

    def test_message_list_content_with_audio(self) -> None:
        ref = MediaRef(modality="audio", uri="/tmp/test.wav")
        msg = Message(
            role="user",
            content=[
                TextPart(text="Listen"),
                MediaPart(media=ref),
            ],
        )
        self.assertEqual(msg.text_content, "Listen")
        self.assertEqual(len(msg.audio_refs), 1)
        self.assertEqual(msg.audio_refs[0].uri, "/tmp/test.wav")
        self.assertEqual(len(msg.media_refs), 1)

    def test_message_image_only_is_valid(self) -> None:
        ref = MediaRef(modality="image", uri="/tmp/test.jpg")
        msg = Message(
            role="user",
            content=[MediaPart(media=ref)],
        )
        self.assertEqual(msg.text_content, "")
        self.assertEqual(len(msg.image_refs), 1)

    def test_message_empty_assistant_list_content_allowed(self) -> None:
        msg = Message(
            role="assistant",
            content=[TextPart(text="")],
        )
        self.assertEqual(msg.text_content, "")

    def test_message_empty_list_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Message(
                role="user",
                content=[TextPart(text="   ")],
            )

    def test_media_ref_rejects_empty_uri(self) -> None:
        with self.assertRaises(ValidationError):
            MediaRef(modality="image", uri="")

    def test_malformed_image_url_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": None,
                    },
                ],
            )

    def test_openai_text_content_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[{"type": "text", "text": "Hello"}],
        )
        self.assertIsInstance(msg.content, list)
        self.assertEqual(msg.text_content, "Hello")

    def test_openai_image_url_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/img.png",
                    },
                },
            ],
        )
        self.assertEqual(msg.text_content, "What is this?")
        self.assertEqual(len(msg.image_refs), 1)
        self.assertEqual(
            msg.image_refs[0].uri,
            "https://example.com/img.png",
        )

    def test_openai_image_url_string_form_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Check"},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/img.png",
                },
            ],
        )
        self.assertEqual(len(msg.image_refs), 1)
        self.assertEqual(
            msg.image_refs[0].uri,
            "https://example.com/img.png",
        )

    def test_openai_image_type_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Check"},
                {
                    "type": "image",
                    "url": "https://example.com/photo.jpg",
                },
            ],
        )
        self.assertEqual(len(msg.image_refs), 1)
        self.assertEqual(
            msg.image_refs[0].uri,
            "https://example.com/photo.jpg",
        )

    def test_openai_audio_type_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Check"},
                {
                    "type": "audio",
                    "audio_url": "https://example.com/audio.wav",
                },
            ],
        )
        self.assertEqual(len(msg.audio_refs), 1)
        self.assertEqual(
            msg.audio_refs[0].uri,
            "https://example.com/audio.wav",
        )

    def test_openai_audio_object_form_coercion(self) -> None:
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Check"},
                {
                    "type": "audio",
                    "audio_url": {
                        "url": "https://example.com/audio.wav",
                    },
                },
            ],
        )
        self.assertEqual(len(msg.audio_refs), 1)
        self.assertEqual(
            msg.audio_refs[0].uri,
            "https://example.com/audio.wav",
        )

    def test_media_ref_fields(self) -> None:
        ref = MediaRef(
            modality="image",
            uri="/data/img.png",
            sha256="abc123",
            width=640,
            height=480,
        )
        self.assertEqual(ref.modality, "image")
        self.assertEqual(ref.sha256, "abc123")
        self.assertEqual(ref.width, 640)

    def test_audio_media_ref_fields(self) -> None:
        ref = MediaRef(
            modality="audio",
            uri="/data/audio.wav",
            sha256="abc123",
            mime_type="audio/wav",
            duration_seconds=1.5,
            sample_rate_hz=16000,
            channels=1,
        )
        self.assertEqual(ref.modality, "audio")
        self.assertEqual(ref.mime_type, "audio/wav")
        self.assertEqual(ref.duration_seconds, 1.5)
        self.assertEqual(ref.sample_rate_hz, 16000)
        self.assertEqual(ref.channels, 1)

    def test_normalized_sample_category_labels(self) -> None:
        sample = NormalizedSample(
            id="s1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": True},
            category_labels=("violence", "hate"),
        )
        self.assertEqual(
            sample.category_labels, ("violence", "hate")
        )

    def test_normalized_sample_default_no_categories(self) -> None:
        sample = NormalizedSample(
            id="s1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": False},
        )
        self.assertEqual(sample.category_labels, ())

    def test_prediction_category_sidecars(self) -> None:
        pred = NormalizedPrediction(
            sample_id="s1",
            unsafe_score=0.9,
            unsafe_label=True,
            threshold=0.5,
            predicted_categories=("S12",),
            category_scores={"S12": 0.9, "S1": 0.1},
        )
        self.assertEqual(pred.predicted_categories, ("S12",))
        self.assertAlmostEqual(
            pred.category_scores["S12"], 0.9
        )

    def test_prediction_default_no_categories(self) -> None:
        pred = NormalizedPrediction(
            sample_id="s1",
            unsafe_score=0.1,
            unsafe_label=False,
            threshold=0.5,
        )
        self.assertEqual(pred.predicted_categories, ())
        self.assertEqual(pred.category_scores, {})

    def test_dataset_metadata_input_modalities(self) -> None:
        meta = DatasetMetadata(
            name="img",
            display_name="Images",
            input_modalities=("text", "image"),
        )
        self.assertEqual(
            meta.input_modalities, ("text", "image")
        )

    def test_dataset_metadata_default_modality(self) -> None:
        meta = DatasetMetadata(
            name="txt", display_name="Text"
        )
        self.assertEqual(meta.input_modalities, ("text",))

    def test_adapter_capabilities_modalities(self) -> None:
        caps = AdapterCapabilities(
            adapter_name="vision",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text", "image"),
            supports_category_outputs=True,
        )
        self.assertIn(
            "image", caps.supported_input_modalities
        )
        self.assertTrue(caps.supports_category_outputs)

    def test_adapter_capabilities_defaults(self) -> None:
        caps = AdapterCapabilities(
            adapter_name="text",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
        )
        self.assertEqual(
            caps.supported_input_modalities, ("text",)
        )
        self.assertFalse(caps.supports_category_outputs)


# Keys that would render ground truth into prompts if a dataset loader
# stashed them on ``NormalizedSample.metadata`` (which flows unchanged
# into ``PredictSample.metadata`` and is exposed to prompt templates as
# ``{metadata_<key>}``). Add to this list as new leak shapes appear.
_LEAKY_METADATA_KEYS = (
    # generic
    "raw_label",
    "label",
    "label_name",
    "binary_label",
    "majority_label",
    "safety_label",
    "image_safety_label",
    "image_safe",
    "source_role",
    "safety_reason",
    # wildguardmix
    "prompt_harm_label",
    "response_harm_label",
    "response_refusal_label",
    # jigsaw / civil comments per-category toxicity columns
    "toxic",
    "severe_toxic",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "identity_attack",
    "sexual_explicit",
    "prompt_toxicity",
    "continuation_toxicity",
    "class",
    "prompt_verdict",
    "default_answer_verdict",
    "hate_speech_score",
    "offensiveYN",
    "violated_categories",
    "active_categories",
    "data_type",
    "degree_of_harm",
    "type",
    "hate_speech_count",
    "offensive_language_count",
    "neither_count",
    # openai moderation eval column names
    "S",
    "H",
    "V",
    "HR",
    "SH",
    "S3",
    "H2",
    "V2",
)


class LabelLeakageRegressionTest(unittest.TestCase):
    """Guard against ground-truth re-entering the predict-path sample."""

    def test_no_dataset_loader_stores_label_shaped_metadata(self) -> None:
        import ast
        from pathlib import Path

        loaders_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "guard_eval_harness"
            / "datasets"
        )
        leaky = set(_LEAKY_METADATA_KEYS)

        def _flag_dict_keys(node: ast.Dict, path: Path) -> list[str]:
            hits: list[str] = []
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in leaky
                ):
                    hits.append(f"{path.name}:{key.lineno}: {key.value}")
            return hits

        offenders: list[str] = []
        for path in sorted(loaders_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                # Case 1: metadata={...} keyword argument anywhere
                # (covers NormalizedSample(... metadata={...} ...) calls).
                if isinstance(node, ast.keyword) and node.arg == "metadata":
                    if isinstance(node.value, ast.Dict):
                        offenders.extend(_flag_dict_keys(node.value, path))
                # Case 2: metadata["key"] = ... explicit subscript assign.
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "metadata"
                            and isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str)
                            and target.slice.value in leaky
                        ):
                            offenders.append(
                                f"{path.name}:{target.lineno}: "
                                f"{target.slice.value}"
                            )
                # Case 3: metadata_fields_to_preserve = (...) class attrs.
                elif isinstance(node, ast.Assign):
                    pass  # handled above
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "metadata_fields_to_preserve"
                ):
                    # Resolve direct tuple / list literals.
                    candidates: list[ast.AST] = []
                    if isinstance(node.value, (ast.Tuple, ast.List)):
                        candidates = list(node.value.elts)
                    # Resolve `metadata_fields_to_preserve = SOME_NAME`
                    # by looking up SOME_NAME at module scope.
                    elif isinstance(node.value, ast.Name):
                        target = node.value.id
                        for top in ast.iter_child_nodes(tree):
                            if (
                                isinstance(top, ast.Assign)
                                and len(top.targets) == 1
                                and isinstance(top.targets[0], ast.Name)
                                and top.targets[0].id == target
                                and isinstance(top.value, (ast.Tuple, ast.List))
                            ):
                                candidates = list(top.value.elts)
                                break
                    for elt in candidates:
                        if (
                            isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                            and elt.value in leaky
                        ):
                            offenders.append(
                                f"{path.name}:{elt.lineno}: "
                                f"metadata_fields_to_preserve {elt.value}"
                            )
        self.assertEqual(
            offenders,
            [],
            msg=(
                "Dataset loaders must not store ground-truth-shaped keys "
                "in NormalizedSample.metadata; they propagate to "
                "PredictSample.metadata and into prompt templates as "
                "{metadata_<key>}. Found:\n" + "\n".join(offenders)
            ),
        )

    def test_normalized_sample_to_predict_sample_drops_label_fields(self) -> None:
        from guard_eval_harness.schemas import NormalizedSample

        sample = NormalizedSample(
            id="s",
            dataset="d",
            split="test",
            messages=[Message(role="user", content="hi")],
            label={"unsafe": True},
            category_labels=("violence",),
            metadata={"safe_field": "ok"},
        )
        predict = sample.to_predict_sample()
        self.assertFalse(hasattr(predict, "label"))
        self.assertFalse(hasattr(predict, "category_labels"))
        self.assertEqual(predict.metadata, {"safe_field": "ok"})


if __name__ == "__main__":
    unittest.main()
