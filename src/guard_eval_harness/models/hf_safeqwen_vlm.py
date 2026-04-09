"""SafeQwen2.5-VL multimodal safety classifier adapter."""

from __future__ import annotations

import importlib
import time
from typing import Any, Sequence

from guard_eval_harness.models.huggingface import HuggingFaceAdapter
from guard_eval_harness.models.multimodal import (
    load_sample_images,
    model_device,
    move_batch_to_device,
)
from guard_eval_harness.models.templates import sample_to_text
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    NormalizedSample,
)


_DEFAULT_SAFETY_CATEGORIES: tuple[str, ...] = (
    "safe",
    "gender_discrimination",
    "race_discrimination",
    "religion_discrimination",
    "harassment",
    "disability_discrimination",
    "drug_related_hazards",
    "property_crime",
    "facial_data_exposure",
    "identity_data_exposure",
    "physical_self_injury",
    "suicide",
    "animal_abuse",
    "obscene_gestures",
    "physical_altercation",
    "terrorism",
    "weapon_related_violence",
    "sexual_content",
    "financial_advice",
    "medical_advice",
)


@model_registry.register("hf_safeqwen_vlm")
class SafeQwenVLMAdapter(HuggingFaceAdapter):
    """Adapter for SafeQwen2.5-VL image safety classification.

    Unlike standard VLM guards that use ``model.generate()`` and
    parse textual verdicts, SafeQwen exposes a dedicated safety
    head activated via ``do_safety=True`` during the forward pass.
    The head produces per-category probabilities over 20 safety
    classes (index 0 = safe, 1..19 = unsafe categories).
    """

    adapter_name = "hf_safeqwen_vlm"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._processor: Any | None = None
        self._vlm_model: Any | None = None
        self._vlm_device: Any | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text", "image"),
            supports_category_outputs=True,
            notes=("safeqwen-vlm-safety-head",),
        )

    def _processor_name(self) -> str:
        return str(
            self.config.args.get("processor")
            or self.config.args.get("tokenizer")
            or "Qwen/Qwen2.5-VL-7B-Instruct"
        )

    def _get_processor(self) -> Any:
        if self._processor is None:
            transformers = importlib.import_module("transformers")
            kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get(
                        "trust_remote_code", True
                    )
                ),
                "revision": self.config.args.get(
                    "revision", "main"
                ),
            }
            self._processor = (
                transformers.AutoProcessor.from_pretrained(
                    self._processor_name(),
                    **kwargs,
                )
            )
            tokenizer = getattr(
                self._processor, "tokenizer", None
            )
            if tokenizer is not None:
                tokenizer.padding_side = str(
                    self.config.args.get(
                        "padding_side", "left"
                    )
                )
                if (
                    getattr(tokenizer, "pad_token_id", None)
                    is None
                    and getattr(tokenizer, "eos_token", None)
                    is not None
                ):
                    tokenizer.pad_token = tokenizer.eos_token
        return self._processor

    def _get_vlm_model(self) -> tuple[Any, Any | None]:
        if self._vlm_model is None:
            transformers = importlib.import_module(
                "transformers"
            )
            torch = importlib.import_module("torch")
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get(
                        "trust_remote_code", True
                    )
                ),
                "revision": self.config.args.get(
                    "revision", "main"
                ),
            }
            device_map = self._configured_device_map()
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            torch_dtype = self._resolve_torch_dtype()
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs.update(
                dict(
                    self.config.args.get("model_kwargs", {})
                )
            )
            # Resolve the Auto class for loading.
            # AutoModelForVision2Seq was removed in transformers
            # >=5.4; AutoModelForImageTextToText is the replacement.
            # Some models (e.g. SafeQwen2.5-VL) only register in
            # auto_map under the old name, so we pre-load the config
            # and patch auto_map to include the new name.
            for auto_name in (
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
            ):
                model_cls = getattr(
                    transformers, auto_name, None
                )
                if model_cls is not None:
                    break
            if model_cls is None:
                raise ImportError(
                    "No suitable vision-language Auto class"
                )
            config = transformers.AutoConfig.from_pretrained(
                self._model_name(),
                trust_remote_code=model_kwargs.get(
                    "trust_remote_code", True
                ),
                revision=model_kwargs.get("revision", "main"),
            )
            auto_map = getattr(config, "auto_map", None) or {}
            old_entry = auto_map.get("AutoModelForVision2Seq")
            if (
                old_entry
                and auto_name != "AutoModelForVision2Seq"
                and auto_name not in auto_map
            ):
                auto_map[auto_name] = old_entry
                config.auto_map = auto_map

            # SafeQwen2.5-VL's custom model code reads
            # config.hidden_size and config.vocab_size at the top
            # level, but VL configs store these under text_config.
            # Promote them so the custom __init__ doesn't crash.
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                for attr in ("hidden_size", "vocab_size"):
                    if (
                        getattr(config, attr, None) is None
                        and getattr(text_cfg, attr, None)
                        is not None
                    ):
                        setattr(config, attr, getattr(text_cfg, attr))

            model_kwargs["config"] = config
            self._vlm_model = model_cls.from_pretrained(
                self._model_name(),
                **model_kwargs,
            )
            if device_map is None:
                self._vlm_device = self._resolve_device(
                    torch
                )
                self._vlm_model.to(self._vlm_device)
            else:
                self._vlm_device = None
            self._vlm_model.eval()
        return self._vlm_model, self._vlm_device

    def _safety_categories(self) -> tuple[str, ...]:
        configured = self.config.args.get(
            "safety_categories"
        )
        if configured is not None:
            return tuple(str(c) for c in configured)
        return _DEFAULT_SAFETY_CATEGORIES

    def _safe_category_index(self) -> int:
        return int(
            self.config.args.get("safe_category_index", 0)
        )

    def predict_batch(
        self,
        samples: Sequence[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        """Run SafeQwen safety head for a batch."""
        if not samples:
            return []

        torch = importlib.import_module("torch")
        processor = self._get_processor()
        model, configured_device = self._get_vlm_model()

        prepared_samples: list[NormalizedSample] = []
        prompts: list[str] = []
        all_images: list[list[Any]] = []
        drop_failed = self.allow_partial_predictions

        for sample in samples:
            try:
                loaded_images = load_sample_images(sample)
                if len(loaded_images) != 1:
                    raise ValueError(
                        f"hf_safeqwen_vlm expects exactly "
                        f"one image, got "
                        f"{len(loaded_images)} for sample "
                        f"{sample.id}"
                    )
                conversation_text = (
                    sample_to_text(sample).strip()
                )
                if not conversation_text:
                    conversation_text = (
                        "Is this image safe?"
                    )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": conversation_text,
                            },
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
                all_images.append(loaded_images)
                prepared_samples.append(sample)
            except Exception:
                if not drop_failed:
                    raise
                continue

        if not prepared_samples:
            return []

        flat_images = [imgs[0] for imgs in all_images]
        inputs = processor(
            text=prompts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )
        device = (
            configured_device or model_device(model)
        )
        inputs = move_batch_to_device(inputs, device)

        started_at = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                **inputs, do_safety=True
            )
        latency_ms = (
            (time.perf_counter() - started_at)
            * 1000.0
            / len(prepared_samples)
        )

        safety_probs = outputs.img_safety_probs
        categories = self._safety_categories()
        safe_index = self._safe_category_index()
        include_scores = bool(
            self.config.args.get(
                "include_label_scores", False
            )
        )

        predictions: list[NormalizedPrediction] = []
        for i, sample in enumerate(prepared_samples):
            probs = safety_probs[i]
            prob_list = probs.detach().float().cpu().tolist()
            safe_prob = (
                prob_list[safe_index]
                if safe_index < len(prob_list)
                else 0.0
            )
            unsafe_score = 1.0 - safe_prob

            top_index = max(
                range(len(prob_list)),
                key=prob_list.__getitem__,
            )
            top_category = (
                categories[top_index]
                if top_index < len(categories)
                else f"LABEL_{top_index}"
            )

            predicted_cats: tuple[str, ...] = ()
            if unsafe_score >= threshold and (
                top_index != safe_index
            ):
                predicted_cats = (top_category,)

            metadata: dict[str, Any] = {
                "adapter": self.adapter_name,
                "model_name": self._model_name(),
                "top_category": top_category,
                "top_category_score": prob_list[
                    top_index
                ],
                "safe_prob": safe_prob,
            }

            category_scores: dict[str, float] = {}
            if include_scores:
                for j, cat in enumerate(categories):
                    if j < len(prob_list):
                        category_scores[cat] = prob_list[
                            j
                        ]

            predictions.append(
                NormalizedPrediction(
                    sample_id=sample.id,
                    unsafe_score=unsafe_score,
                    unsafe_label=unsafe_score >= threshold,
                    threshold=threshold,
                    latency_ms=latency_ms,
                    predicted_categories=predicted_cats,
                    category_scores=category_scores,
                    metadata=metadata,
                )
            )
        return predictions
