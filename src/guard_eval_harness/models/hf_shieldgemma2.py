"""Hugging Face ShieldGemma 2 adapter."""

from __future__ import annotations

import importlib
import time
from typing import Any, Mapping, Sequence

from guard_eval_harness.models.huggingface import HuggingFaceAdapter
from guard_eval_harness.models.multimodal import (
    load_sample_images,
    model_device,
    move_batch_to_device,
)
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    PredictSample,
)


_POLICY_ALIASES = {
    "sexually_explicit": "sexual",
    "sexual_explicit": "sexual",
}


@model_registry.register("hf_shieldgemma2")
class HuggingFaceShieldGemma2Adapter(HuggingFaceAdapter):
    """Policy-aware ShieldGemma 2 image moderation adapter."""

    adapter_name = "hf_shieldgemma2"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_device: Any | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("image",),
            supports_category_outputs=True,
            notes=("transformers-shieldgemma2",),
        )

    def _get_processor(self) -> Any:
        """Load the ShieldGemma 2 processor once."""
        if self._processor is None:
            transformers = importlib.import_module("transformers")
            processor_kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get("trust_remote_code", False)
                ),
                "revision": self.config.args.get("revision", "main"),
            }
            self._processor = transformers.AutoProcessor.from_pretrained(
                self.config.args.get("processor") or self._model_name(),
                **processor_kwargs,
            )
        return self._processor

    def _get_model(self) -> tuple[Any, Any | None]:
        """Load the ShieldGemma 2 model once."""
        if self._model is None:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get("trust_remote_code", False)
                ),
                "revision": self.config.args.get("revision", "main"),
            }
            device_map = self._configured_device_map()
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            torch_dtype = self._resolve_torch_dtype()
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs.update(dict(self.config.args.get("model_kwargs", {})))
            model_cls_name = str(
                self.config.args.get(
                    "model_class",
                    "ShieldGemma2ForImageClassification",
                )
            )
            model_cls = getattr(transformers, model_cls_name)
            self._model = model_cls.from_pretrained(
                self._model_name(),
                **model_kwargs,
            )
            if device_map is None:
                self._model_device = self._resolve_device(torch)
                self._model.to(self._model_device)
            else:
                self._model_device = None
            self._model.eval()
        return self._model, self._model_device

    def _resolved_policies(self, processor: Any) -> list[str]:
        """Resolve the ordered policy list for one batch."""
        configured = self.config.args.get("policies")
        if isinstance(configured, Sequence) and not isinstance(
            configured,
            str,
        ):
            return [
                _POLICY_ALIASES.get(str(policy), str(policy))
                for policy in configured
            ]

        policy_definitions = getattr(processor, "policy_definitions", None)
        if isinstance(policy_definitions, Mapping):
            merged = {
                **dict(policy_definitions),
                **self._custom_policies(),
            }
            return [str(policy) for policy in merged]
        raise ValueError(
            "hf_shieldgemma2 requires explicit model.args.policies when "
            "the processor does not expose policy_definitions"
        )

    def _custom_policies(self) -> dict[str, str]:
        """Return configured custom policies."""
        configured = self.config.args.get("custom_policies", {})
        if not isinstance(configured, Mapping):
            return {}
        return {str(key): str(value) for key, value in configured.items()}

    def _policy_aggregation(self, scores: Sequence[float]) -> float:
        """Aggregate per-policy violation scores into one unsafe score."""
        if not scores:
            raise ValueError("ShieldGemma 2 returned no policy scores")
        aggregation = str(
            self.config.args.get("policy_aggregation", "max")
        ).lower()
        if aggregation in {"any", "max"}:
            return max(scores)
        if aggregation == "mean":
            return sum(scores) / len(scores)
        if aggregation == "first":
            return scores[0]
        if aggregation == "last":
            return scores[-1]
        raise ValueError(f"unsupported policy_aggregation: {aggregation}")

    def _category_threshold(self, threshold: float) -> float:
        """Return the threshold used for predicted policy categories."""
        configured = self.config.args.get("category_threshold")
        if configured is None:
            return threshold
        return float(configured)

    def _emits_categories(self) -> bool:
        """Return whether policy categories should be populated."""
        return bool(self.config.args.get("emit_categories", True))

    def _include_policy_scores(self) -> bool:
        """Return whether policy scores should be preserved."""
        return self._emits_categories() or bool(
            self.config.args.get("include_policy_scores", False)
        )

    def _image_for_sample(self, sample: PredictSample) -> Any:
        """Load exactly one image for one sample."""
        images = load_sample_images(sample)
        if len(images) != 1:
            raise ValueError(
                f"{self.adapter_name} expects exactly one image, got "
                f"{len(images)} for sample {sample.id}"
            )
        if not bool(self.config.args.get("ignore_text_content", False)) and any(
            message.text_content.strip() for message in sample.messages
        ):
            raise ValueError(
                "hf_shieldgemma2 does not support text input; set "
                "model.args.ignore_text_content=true to ignore message text"
            )
        return images[0]

    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        """Run ShieldGemma 2 policy scoring for one batch."""
        if not samples:
            return []

        torch = importlib.import_module("torch")
        processor = self._get_processor()
        model, configured_device = self._get_model()
        policies = self._resolved_policies(processor)
        custom_policies = self._custom_policies()

        prepared_samples: list[PredictSample] = []
        images: list[Any] = []
        drop_failed = self.allow_partial_predictions
        for sample in samples:
            try:
                images.append(self._image_for_sample(sample))
                prepared_samples.append(sample)
            except Exception:
                if not drop_failed:
                    raise
                continue

        if not prepared_samples:
            return []

        processor_kwargs: dict[str, Any] = {
            "images": images,
            "return_tensors": "pt",
        }
        if policies:
            processor_kwargs["policies"] = policies
        if custom_policies:
            processor_kwargs["custom_policies"] = custom_policies

        started_at = time.perf_counter()
        inputs = processor(**processor_kwargs)
        inputs = move_batch_to_device(
            inputs,
            configured_device or model_device(model),
        )
        with torch.no_grad():
            outputs = model(**inputs)
        latency_ms = (
            (time.perf_counter() - started_at) * 1000.0 / len(prepared_samples)
        )

        probabilities = outputs.probabilities.detach().float().cpu()
        expected_rows = len(prepared_samples) * len(policies)
        if int(probabilities.shape[0]) != expected_rows:
            raise ValueError(
                "ShieldGemma 2 returned an unexpected probability shape: "
                f"{tuple(probabilities.shape)} for {len(prepared_samples)} "
                f"samples and {len(policies)} policies"
            )

        predictions: list[NormalizedPrediction] = []
        category_threshold = self._category_threshold(threshold)
        include_policy_scores = self._include_policy_scores()
        emit_categories = self._emits_categories()

        for sample_index, sample in enumerate(prepared_samples):
            start = sample_index * len(policies)
            end = start + len(policies)
            row_probabilities = probabilities[start:end, 0].tolist()
            policy_scores = {
                policies[index]: float(score)
                for index, score in enumerate(row_probabilities)
            }
            unsafe_score = self._policy_aggregation(row_probabilities)
            top_policy = max(
                policy_scores,
                key=policy_scores.__getitem__,
            )
            predicted_categories = ()
            if emit_categories:
                predicted_categories = tuple(
                    policy
                    for policy, score in policy_scores.items()
                    if score >= category_threshold
                )
            prediction = NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=unsafe_score,
                unsafe_label=unsafe_score >= threshold,
                threshold=threshold,
                latency_ms=latency_ms,
                predicted_categories=predicted_categories,
                category_scores=(
                    dict(policy_scores) if include_policy_scores else {}
                ),
                metadata={
                    "adapter": self.adapter_name,
                    "model_name": self._model_name(),
                    "top_policy": top_policy,
                    "top_policy_score": policy_scores[top_policy],
                },
            )
            predictions.append(prediction)
        return predictions
