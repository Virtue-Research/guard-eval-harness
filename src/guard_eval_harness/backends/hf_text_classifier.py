"""Hugging Face text-classification backend.

Wraps ``AutoTokenizer`` + ``AutoModelForSequenceClassification`` for
discriminative text safety models (prompt-injection classifiers,
toxicity heads, etc.). Returns per-label softmax probabilities.

Pair with ``PromptGuardGuard`` (or any classifier-style ``Guard``)
that maps the returned label distribution into ``unsafe_score``.
"""

import logging
from typing import Any, Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    ClassifierBackend,
    backend_registry,
)
from guard_eval_harness.schemas import MediaPart, Message, TextPart

_log = logging.getLogger(__name__)


@backend_registry.register("hf_text_classifier")
class HFTextClassifierBackend(ClassifierBackend):
    """Local sequence-classification backend.

    Configurable via ``args``:
      - ``device`` (str): default ``"auto"``.
      - ``dtype`` (str): default ``"auto"``.
      - ``trust_remote_code`` (bool): default ``False``.
      - ``revision`` (str): optional HF revision pin.
      - ``max_length`` (int): tokenizer truncation length (default 512).
      - ``role_filter`` (list[str] | None): if set, only concatenate
        messages whose role appears in this list. Default ``["user"]``
        — matches prompt-injection classifiers which score the human
        request only.
      - ``join_separator`` (str): glue between included message texts
        (default ``"\\n\\n"``).
    """

    kind = "hf_text_classifier"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if config.model is None:
            raise ValueError(
                "HFTextClassifierBackend requires backend.name "
                "(HF repo ID or local model path)."
            )
        args = config.args
        self.model_name: str = config.model
        self.device: str = str(args.get("device", "auto"))
        self.dtype: str = str(args.get("dtype", "auto"))
        self.trust_remote_code: bool = bool(
            args.get("trust_remote_code", False)
        )
        self.revision: str | None = args.get("revision")
        self.max_length: int = int(args.get("max_length", 512))
        role_filter = args.get("role_filter", ["user"])
        self.role_filter: tuple[str, ...] | None = (
            tuple(role_filter) if role_filter else None
        )
        self.join_separator: str = str(args.get("join_separator", "\n\n"))

        self._model = None
        self._tokenizer = None
        self._device = None
        self._id2label: dict[int, str] = {}

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFTextClassifierBackend requires torch"
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
                "HFTextClassifierBackend requires torch"
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
        """Lazy-load tokenizer + classifier head."""
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "HFTextClassifierBackend requires transformers: "
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

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        config_id2label = getattr(self._model.config, "id2label", None)
        if isinstance(config_id2label, dict):
            self._id2label = {
                int(k): str(v) for k, v in config_id2label.items()
            }
        else:
            num_labels = getattr(self._model.config, "num_labels", 0)
            self._id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

    def _extract_text(self, messages: Sequence[Message]) -> str:
        """Concatenate text from the filtered roles."""
        chunks: list[str] = []
        for message in messages:
            if self.role_filter is not None and message.role not in self.role_filter:
                continue
            if isinstance(message.content, str):
                if message.content.strip():
                    chunks.append(message.content)
                continue
            for part in message.content:
                if isinstance(part, TextPart):
                    if part.text.strip():
                        chunks.append(part.text)
                elif isinstance(part, MediaPart):
                    # Text classifiers can't consume media; skip silently.
                    continue
        if not chunks:
            raise ValueError(
                "HFTextClassifierBackend: no text content found in "
                f"messages matching role_filter={self.role_filter!r}"
            )
        return self.join_separator.join(chunks)

    def classify(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[dict[str, float]]:
        """Return per-label softmax probabilities for each conversation."""
        self._load()
        import torch

        results: list[dict[str, float]] = []
        for messages in batch:
            text = self._extract_text(messages)
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
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
