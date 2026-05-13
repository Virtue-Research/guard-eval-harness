"""Hugging Face image-classification adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from guard_eval_harness.models.huggingface import HuggingFaceAdapter
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    PredictSample,
)


@model_registry.register("hf_image_classifier")
class HuggingFaceImageClassifierAdapter(HuggingFaceAdapter):
    """Transformers-backed image safety classifier."""

    adapter_name = "hf_image_classifier"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._image_model: Any | None = None
        self._image_device: Any | None = None
        self._image_processor: Any | None = None

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
            notes=("transformers-image-classification",),
        )

    def _use_siglip_loader(self) -> bool:
        """Return whether the model requires the SigLIP loader."""
        explicit = self.config.args.get("loader")
        if explicit is not None:
            return str(explicit).strip().lower() == "siglip"
        return "siglip" in self._model_name().lower()

    def _get_image_processor(self) -> Any:
        """Load the configured image processor once."""
        if self._image_processor is None:
            transformers = importlib.import_module("transformers")
            processor_name = (
                self.config.args.get("image_processor")
                or self.config.args.get("processor")
                or self._model_name()
            )
            kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get("trust_remote_code", False)
                ),
                "revision": self.config.args.get("revision", "main"),
            }
            self._image_processor = transformers.AutoImageProcessor.from_pretrained(
                processor_name,
                **kwargs,
            )
        return self._image_processor

    def _get_image_model(self) -> tuple[Any, Any | None]:
        """Load the underlying image classifier model."""
        if self._image_model is None:
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
            model_kwargs.update(
                dict(self.config.args.get("model_kwargs", {}))
            )
            if self._use_siglip_loader():
                model_cls = transformers.SiglipForImageClassification
            else:
                model_cls = transformers.AutoModelForImageClassification
            self._image_model = model_cls.from_pretrained(
                self._model_name(),
                **model_kwargs,
            )
            if self._configured_device_map() is None:
                self._image_device = self._resolve_device(torch)
                self._image_model.to(self._image_device)
            else:
                self._image_device = None
            self._image_model.eval()
        return self._image_model, self._image_device

    def _label_names(self, label_count: int) -> list[str]:
        """Resolve label names for image-classification outputs."""
        configured = self.config.args.get("label_names")
        if isinstance(configured, Sequence) and not isinstance(configured, str):
            names = [str(name) for name in configured]
            if len(names) == label_count:
                return names

        model, _device = self._get_image_model()
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, Mapping):
            return [
                str(
                    id2label.get(
                        index,
                        id2label.get(str(index), f"LABEL_{index}"),
                    )
                )
                for index in range(label_count)
            ]
        return [f"LABEL_{index}" for index in range(label_count)]

    def _label_is_safe(self, label: str) -> bool:
        """Return whether a label name indicates a safe class."""
        normalized = label.lower()
        safe_labels = tuple(
            str(candidate).lower()
            for candidate in self.config.args.get(
                "safe_labels",
                (
                    "safe",
                    "benign",
                    "clean",
                    "allow",
                    "non-toxic",
                    "normal",
                    "sfw",
                ),
            )
        )
        if any(token in normalized for token in safe_labels):
            return True
        return "safe" in normalized and "unsafe" not in normalized

    def _label_is_unsafe(self, label: str) -> bool:
        """Return whether a label name indicates an unsafe class."""
        normalized = label.lower()
        unsafe_labels = tuple(
            str(candidate).lower()
            for candidate in self.config.args.get(
                "unsafe_labels",
                (
                    "unsafe",
                    "harmful",
                    "toxic",
                    "violation",
                    "reject",
                    "nsfw",
                ),
            )
        )
        return any(token in normalized for token in unsafe_labels)

    def _image_for_sample(self, sample: PredictSample) -> Any:
        """Load the first image for one sample as a PIL image."""
        image_refs = [
            image_ref
            for message in sample.messages
            for image_ref in message.image_refs
        ]
        if not image_refs:
            raise ValueError(
                f"sample {sample.id} does not contain any images"
            )
        if len(image_refs) > 1:
            raise ValueError(
                f"sample {sample.id} contains multiple images; "
                "hf_image_classifier expects exactly one"
            )

        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for image classification support"
            ) from exc

        path = Path(image_refs[0].uri)
        with Image.open(path) as image:
            return image.convert("RGB")

    def _activation_name(
        self,
        label_count: int,
        *,
        model: Any | None = None,
    ) -> str:
        """Resolve the configured activation for logits."""
        activation = self.config.args.get("activation")
        if activation is not None:
            return str(activation).lower()
        model_config = getattr(model, "config", None)
        if getattr(model_config, "problem_type", None) == (
            "multi_label_classification"
        ):
            return "sigmoid"
        return "sigmoid" if label_count == 1 else "softmax"

    def _probabilities_from_logits(
        self,
        logits: Any,
        *,
        torch: Any,
        model: Any | None = None,
    ) -> list[list[float]]:
        """Normalize logits into per-label probabilities."""
        label_count = int(logits.shape[-1])
        activation = self._activation_name(label_count, model=model)
        if activation == "sigmoid":
            probabilities = torch.sigmoid(logits)
        elif activation == "softmax":
            probabilities = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"unsupported activation: {activation}")
        return probabilities.detach().float().cpu().tolist()

    def _unsafe_score_from_image_label_scores(
        self,
        label_scores: Mapping[str, float],
    ) -> float:
        """Resolve one unsafe score from image-classification probabilities.

        For image classifiers, numeric ``label_score_mapping`` values are
        treated as weights over model probabilities rather than constants.
        This preserves the classifier's probability mass for binary mappings
        like ``normal -> 0.0`` and ``nsfw -> 1.0``.
        """
        raw_mapping = self.config.args.get("label_score_mapping", {})
        if isinstance(raw_mapping, Mapping):
            explicit_mapping = {
                str(key).strip().lower(): value
                for key, value in raw_mapping.items()
            }
            weighted_sum = 0.0
            saw_numeric_mapping = False
            for label, score in label_scores.items():
                normalized_label = str(label).strip().lower()
                mapped = explicit_mapping.get(normalized_label)
                if isinstance(mapped, (int, float)):
                    weighted_sum += float(mapped) * float(score)
                    saw_numeric_mapping = True
            if saw_numeric_mapping:
                return max(0.0, min(1.0, weighted_sum))

        return self._unsafe_score_from_label_scores(label_scores)

    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        """Run image classification and emit canonical predictions."""
        if not samples:
            return []

        torch = importlib.import_module("torch")
        processor = self._get_image_processor()
        model, device = self._get_image_model()
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

        started = time.perf_counter()
        inputs = processor(images=images, return_tensors="pt")
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        score_rows = self._probabilities_from_logits(
            outputs.logits,
            torch=torch,
            model=model,
        )
        if not score_rows:
            raise ValueError(
                "model returned empty logits for "
                f"{len(prepared_samples)} input images; "
                "expected one score row per image"
            )
        label_names = self._label_names(len(score_rows[0]))
        latency_ms = elapsed_ms / len(prepared_samples)

        predictions: list[NormalizedPrediction] = []
        for sample, score_row in zip(prepared_samples, score_rows):
            label_score_map = {
                label_names[index]: float(score)
                for index, score in enumerate(score_row)
            }
            top_index = max(range(len(score_row)), key=score_row.__getitem__)
            top_label = label_names[top_index]
            score = self._unsafe_score_from_image_label_scores(label_score_map)
            prediction = NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=score,
                unsafe_label=score >= threshold,
                threshold=threshold,
                latency_ms=latency_ms,
                metadata={
                    "adapter": self.adapter_name,
                    "model_name": self._model_name(),
                    "top_label": top_label,
                    "top_label_score": score_row[top_index],
                },
            )
            if bool(self.config.args.get("include_label_scores", False)):
                prediction = prediction.model_copy(
                    update={"category_scores": label_score_map}
                )
            predictions.append(prediction)
        return predictions
