"""Hugging Face image-classification backend.

Wraps ``AutoImageProcessor`` + ``AutoModelForImageClassification`` for
discriminative image safety models (NSFW detectors, ImageGuard, etc.).
Returns per-label softmax probabilities.

Use it together with ``HFImageClassifierGuard`` (which maps the
returned label distribution into a single ``unsafe_score``).
"""

import logging
from pathlib import Path
from typing import Any, Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    ClassifierBackend,
    backend_registry,
)
from guard_eval_harness.schemas import MediaPart, Message

_log = logging.getLogger(__name__)


@backend_registry.register("hf_image_classifier")
class HFImageClassifierBackend(ClassifierBackend):
    """Local image-classifier backend.

    Configurable via ``args``:
      - ``device`` (str): default ``"auto"``.
      - ``dtype`` (str): default ``"auto"``.
      - ``trust_remote_code`` (bool): default ``False``.
      - ``revision`` (str): optional HF revision pin.
      - ``image_processor`` (str): override processor repo
        (defaults to the same as the model).
      - ``loader`` (str): ``"auto"`` (default) or ``"siglip"`` to force
        ``SiglipForImageClassification``.
      - ``batch_size`` (int): default 1 (sequential).
    """

    kind = "hf_image_classifier"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if config.model is None:
            raise ValueError(
                "HFImageClassifierBackend requires backend.name"
            )
        args = config.args
        self.model_name: str = config.model
        self.device: str = str(args.get("device", "auto"))
        self.dtype: str = str(args.get("dtype", "auto"))
        self.trust_remote_code: bool = bool(
            args.get("trust_remote_code", False)
        )
        self.revision: str | None = args.get("revision")
        self.processor_name: str = str(
            args.get("image_processor", config.model)
        )
        self.loader: str = str(args.get("loader", "auto")).lower()
        self.batch_size: int = int(args.get("batch_size", 1))

        self._model = None
        self._processor = None
        self._device = None
        self._id2label: dict[int, str] = {}

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFImageClassifierBackend requires torch"
            ) from exc
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self) -> Any:
        if self.dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFImageClassifierBackend requires torch"
            ) from exc
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype in mapping:
            return mapping[self.dtype]
        raise ValueError(f"unknown dtype: {self.dtype!r}")

    def _load(self) -> None:
        """Lazy-load processor + model."""
        if self._model is not None:
            return
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HFImageClassifierBackend requires transformers: "
                "pip install transformers"
            ) from exc

        device = self._resolve_device()
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        dtype = self._resolve_dtype()
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = "auto"

        self._processor = transformers.AutoImageProcessor.from_pretrained(
            self.processor_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

        use_siglip = self.loader == "siglip" or (
            self.loader == "auto" and "siglip" in self.model_name.lower()
        )
        if use_siglip:
            model_cls = transformers.SiglipForImageClassification
        else:
            model_cls = transformers.AutoModelForImageClassification
        self._model = model_cls.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        # Build id → label mapping
        config_id2label = getattr(self._model.config, "id2label", None)
        if isinstance(config_id2label, dict):
            self._id2label = {int(k): str(v) for k, v in config_id2label.items()}
        else:
            num_labels = getattr(self._model.config, "num_labels", 0)
            self._id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

    @staticmethod
    def _extract_image(messages: Sequence[Message]) -> Any:
        """Pull the first image from the message list and return a PIL.Image."""
        for message in messages:
            if isinstance(message.content, str):
                continue
            for part in message.content:
                if isinstance(part, MediaPart) and part.media.modality == "image":
                    try:
                        from PIL import Image
                    except ImportError as exc:
                        raise ImportError(
                            "HFImageClassifierBackend requires Pillow"
                        ) from exc
                    return Image.open(Path(part.media.uri)).convert("RGB")
        raise ValueError(
            "HFImageClassifierBackend: no image part found in messages"
        )

    def classify(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[dict[str, float]]:
        """Return per-label softmax probabilities for each conversation."""
        self._load()
        import torch

        results: list[dict[str, float]] = []
        for messages in batch:
            image = self._extract_image(messages)
            inputs = self._processor(images=image, return_tensors="pt").to(
                self._device
            )
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            results.append(
                {
                    self._id2label.get(idx, f"LABEL_{idx}"): float(probs[idx])
                    for idx in range(probs.shape[0])
                }
            )
        return results
