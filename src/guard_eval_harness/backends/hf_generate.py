"""Hugging Face transformers generate backend (text + multimodal)."""

import logging
from typing import Any, Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    GenerationBackend,
    backend_registry,
)
from guard_eval_harness.schemas import MediaPart, Message, TextPart

_log = logging.getLogger(__name__)


@backend_registry.register("hf_generate")
class HFGenerateBackend(GenerationBackend):
    """Local transformers backend for chat-LLM guards.

    Configurable via ``args``:
      - ``device`` (str): default ``"auto"`` (picks cuda → mps → cpu).
      - ``dtype`` (str): default ``"auto"`` (uses ``torch_dtype="auto"``).
      - ``trust_remote_code`` (bool): default ``False``.
      - ``revision`` (str): optional HF model revision pin.
      - ``apply_chat_template`` (bool): default ``True``. Uses the
        tokenizer's chat template to render messages.
      - ``add_generation_prompt`` (bool): default ``True``.
      - ``raw_text_input`` (bool): default ``False``. When ``True``,
        forwards the (single) user message's text content verbatim and
        tokenizes with ``add_special_tokens=False``. Intended for
        guards like WildGuard whose template already embeds ``<s>``
        and other special tokens — applying a chat template would
        double-wrap them.
      - ``batch_size`` (int): generation batch size (default 1). HF
        generate doesn't pad-and-batch well for chat templates with
        varying lengths, so default sequential.

    The configured ``BackendConfig.model`` is the HF repo ID or local
    path passed to ``AutoModelForCausalLM.from_pretrained``.
    """

    kind = "hf_generate"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if config.model is None:
            raise ValueError(
                "HFGenerateBackend requires backend.name "
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
        self.apply_chat_template: bool = bool(
            args.get("apply_chat_template", True)
        )
        self.add_generation_prompt: bool = bool(
            args.get("add_generation_prompt", True)
        )
        self.raw_text_input: bool = bool(
            args.get("raw_text_input", False)
        )
        self.batch_size: int = int(args.get("batch_size", 1))

        self._model = None
        self._tokenizer = None

    # ---------------------------------------------------------------
    # Lazy load
    # ---------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Pick the best device when ``args.device == 'auto'``."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFGenerateBackend requires torch: pip install torch"
            ) from exc
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self) -> Any:
        """Pick the dtype to load weights at."""
        if self.dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFGenerateBackend requires torch: pip install torch"
            ) from exc
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype in mapping:
            return mapping[self.dtype]
        raise ValueError(f"unknown dtype: {self.dtype!r}")

    def _load(self) -> None:
        """Lazy-load tokenizer + model into memory."""
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "HFGenerateBackend requires transformers: "
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

        _log.info(
            "HFGenerateBackend: loading %s on device=%s",
            self.model_name,
            device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device

    # ---------------------------------------------------------------
    # Message → prompt string
    # ---------------------------------------------------------------

    @staticmethod
    def _message_to_chat_dict(message: Message) -> dict[str, Any]:
        """Convert a Message to the dict format chat templates expect."""
        if isinstance(message.content, str):
            return {"role": message.role, "content": message.content}
        # For multimodal, most chat templates only understand text;
        # we concatenate text parts and warn on dropped media. Image
        # guards should use a multimodal-aware backend.
        text_chunks: list[str] = []
        dropped_media = False
        for part in message.content:
            if isinstance(part, TextPart):
                text_chunks.append(part.text)
            elif isinstance(part, MediaPart):
                dropped_media = True
        if dropped_media:
            _log.warning(
                "HFGenerateBackend: dropping non-text content parts in "
                "message; use a multimodal backend for image guards"
            )
        return {
            "role": message.role,
            "content": " ".join(text_chunks),
        }

    def _render_prompt(self, messages: Sequence[Message]) -> str:
        """Apply the tokenizer's chat template (or fall back to plain concat).

        When ``raw_text_input`` is set, the (single) user message is
        emitted verbatim — the caller's template owns BOS / role markers.
        """
        chat_messages = [self._message_to_chat_dict(m) for m in messages]
        if self.raw_text_input:
            return "\n\n".join(m["content"] for m in chat_messages)
        if self.apply_chat_template and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            return self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
        # Fallback: simple role-prefixed concatenation.
        return "\n\n".join(
            f"{m['role']}: {m['content']}" for m in chat_messages
        )

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[str]:
        """Generate raw text for each conversation."""
        self._load()
        outputs: list[str] = []
        do_sample = self.temperature > 0.0
        for messages in batch:
            prompt = self._render_prompt(messages)
            tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            if self.raw_text_input:
                tokenizer_kwargs["add_special_tokens"] = False
            inputs = self._tokenizer(
                prompt,
                **tokenizer_kwargs,
            ).to(self._device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.temperature
            import torch

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_length:]
            text = self._tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=True,
            )
            outputs.append(text)
        return outputs
