"""Local Hugging Face VLM (vision-language) backend.

Wraps ``AutoProcessor`` + ``AutoModelForVision2Seq``. Handles image
content parts in messages by attaching PIL images to the processor's
chat-template call (HF's standard multimodal chat template path).
"""

import logging
from pathlib import Path
from typing import Any, Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    GenerationBackend,
    backend_registry,
)
from guard_eval_harness.schemas import MediaPart, Message, TextPart

_log = logging.getLogger(__name__)


@backend_registry.register("hf_vlm")
class HFVLMBackend(GenerationBackend):
    """VLM generate backend (image + text → text).

    Configurable via ``args``:
      - ``device`` (str): default ``"auto"``.
      - ``dtype`` (str): default ``"auto"``.
      - ``trust_remote_code`` (bool): default ``False``.
      - ``revision`` (str): optional HF revision pin.
      - ``model_loader`` (str): ``"vision2seq"`` (default) or
        ``"causal_lm"`` for VLMs that ship as causal LMs.
    """

    kind = "hf_vlm"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if config.model is None:
            raise ValueError("HFVLMBackend requires backend.name")
        args = config.args
        self.model_name: str = config.model
        self.device: str = str(args.get("device", "auto"))
        self.dtype: str = str(args.get("dtype", "auto"))
        self.trust_remote_code: bool = bool(
            args.get("trust_remote_code", False)
        )
        self.revision: str | None = args.get("revision")
        self.model_loader: str = str(
            args.get("model_loader", "vision2seq")
        )

        self._model = None
        self._processor = None
        self._device = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:
            raise ImportError("HFVLMBackend requires torch") from exc
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
            raise ImportError("HFVLMBackend requires torch") from exc
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
        if self._model is not None:
            return
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HFVLMBackend requires transformers: pip install transformers"
            ) from exc

        device = self._resolve_device()
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        dtype = self._resolve_dtype()
        load_kwargs["torch_dtype"] = (
            dtype if dtype != "auto" else "auto"
        )

        self._processor = transformers.AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        if self.model_loader == "causal_lm":
            model_cls = transformers.AutoModelForCausalLM
        else:
            model_cls = transformers.AutoModelForVision2Seq
        self._model = model_cls.from_pretrained(
            self.model_name, **load_kwargs
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device

    @staticmethod
    def _open_pil(uri: str) -> Any:
        """Open an image URI as a PIL.Image (RGB)."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "HFVLMBackend requires Pillow: pip install Pillow"
            ) from exc
        return Image.open(Path(uri)).convert("RGB")

    @classmethod
    def _to_chat_messages(
        cls,
        messages: Sequence[Message],
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Convert harness Messages → HF chat template format + PIL images."""
        chat: list[dict[str, Any]] = []
        images: list[Any] = []
        for message in messages:
            if isinstance(message.content, str):
                chat.append(
                    {
                        "role": message.role,
                        "content": [
                            {"type": "text", "text": message.content},
                        ],
                    }
                )
                continue
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, MediaPart):
                    if part.media.modality != "image":
                        raise ValueError(
                            "HFVLMBackend only supports image media; "
                            f"got {part.media.modality!r}"
                        )
                    parts.append({"type": "image"})
                    images.append(cls._open_pil(part.media.uri))
            chat.append({"role": message.role, "content": parts})
        return chat, images

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate text for each (image + text) conversation."""
        self._load()
        import torch

        outputs: list[str] = []
        do_sample = temperature > 0.0
        for messages in batch:
            chat, images = self._to_chat_messages(messages)
            inputs = self._processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images=images if images else None,
            ).to(self._device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs, **gen_kwargs
                )
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_length:]
            text = self._processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            outputs.append(text)
        return outputs
