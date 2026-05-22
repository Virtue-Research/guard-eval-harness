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
    PredictSample,
    RunEnvironment,
    RunManifest,
    SampleGroundTruth,
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

    def test_normalized_sample_to_predict_sample_default_denies_metadata(
        self,
    ) -> None:
        sample = NormalizedSample(
            id="s1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": True},
            category_labels=("violence",),
            metadata={
                "category": "policy",
                "raw_label": "unsafe",
            },
        )

        predict = sample.to_predict_sample()

        self.assertIsInstance(predict, PredictSample)
        self.assertFalse(hasattr(predict, "label"))
        self.assertFalse(hasattr(predict, "category_labels"))
        self.assertEqual(predict.metadata, {})

    def test_normalized_sample_to_predict_sample_uses_allowlist(self) -> None:
        sample = NormalizedSample(
            id="s1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": False},
            metadata={
                "category": "benign",
                "source": "local",
                "raw_label": "safe",
            },
        )

        predict = sample.to_predict_sample(
            predict_metadata_fields=("source", "missing"),
        )

        self.assertEqual(predict.metadata, {"source": "local"})

    def test_predict_sample_rejects_label_like_metadata(self) -> None:
        with self.assertRaises(ValidationError):
            PredictSample(
                id="s1",
                dataset="demo",
                split="test",
                messages=[{"role": "user", "content": "Hello"}],
                metadata={"raw_label": "unsafe"},
            )

    def test_normalized_sample_to_ground_truth(self) -> None:
        sample = NormalizedSample(
            id="s1",
            dataset="demo",
            split="test",
            messages=[{"role": "user", "content": "Hello"}],
            label={"unsafe": True},
            category_labels=("violence",),
        )

        truth = sample.to_ground_truth()

        self.assertIsInstance(truth, SampleGroundTruth)
        self.assertTrue(truth.label.unsafe)
        self.assertEqual(truth.sample_id, "s1")
        self.assertEqual(truth.category_labels, ("violence",))

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
        self.assertFalse(caps.requires_ground_truth)


if __name__ == "__main__":
    unittest.main()
