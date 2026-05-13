"""Hugging Face backend adapter."""

from __future__ import annotations

import importlib
import logging
import math
import time
from typing import Any, Mapping, Sequence

from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.models.openai_compatible import _bool_arg
from guard_eval_harness.models.granite_guardian import (
    prepare_granite_guardian_chat_messages,
    uses_granite_guardian_chat_template,
)
from guard_eval_harness.models.llama_guard import (
    prepare_llama_guard_chat_messages,
    uses_llama_guard_chat_template,
)
from guard_eval_harness.models.templates import (
    extract_judge_categories,
    render_template,
    resolve_score,
    sample_context,
    sample_messages,
    sample_to_text,
)
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    PredictSample,
)


_log = logging.getLogger(__name__)


class _Seq2SeqWrapper:
    """Thin callable wrapping a Seq2SeqLM model to mimic a pipeline.

    The ``text2text-generation`` pipeline was removed in transformers 5.x.
    This wrapper loads the model via ``AutoModelForSeq2SeqLM`` and exposes
    a ``__call__`` that matches the output format the HF adapter expects
    for text-generation pipelines.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any,
        default_max_length: int | None = None,
    ) -> None:
        # When ``device_map="auto"`` (or any accelerate dispatch) is used,
        # the model is already sharded across devices and ``.to(device)``
        # would raise. Respect the existing placement in that case and
        # let accelerate route inputs through its forward hooks.
        if getattr(model, "hf_device_map", None):
            self.model = model.eval()
        else:
            self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.default_max_length = default_max_length

    # Pipeline-only kwargs that the text-generation/text2text pipelines
    # accept but ``model.generate`` rejects. Silently dropped so configs
    # that set e.g. ``pipeline_batch_size`` (which surfaces as
    # ``batch_size`` in _run_backend) don't blow up the seq2seq path.
    _PIPELINE_ONLY_KWARGS = frozenset({
        "batch_size",
        "return_full_text",
        "return_tensors",
        "return_text",
        "clean_up_tokenization_spaces",
    })

    def __call__(
        self,
        inputs: str | list[str],
        *,
        truncation: bool = True,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, str]]]:
        import torch

        if isinstance(inputs, str):
            inputs = [inputs]
        effective_max_length = (
            max_length
            if max_length is not None
            else self.default_max_length
        )
        if effective_max_length is None:
            effective_max_length = getattr(
                self.tokenizer, "model_max_length", None
            )
            # Some tokenizers expose an unrealistic sentinel
            # (e.g. int(1e30)) to mean "unbounded"; fall back to 512
            # only when there's no meaningful limit anywhere.
            if (
                effective_max_length is None
                or effective_max_length > 10**6
            ):
                effective_max_length = 512
        enc = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=truncation,
            max_length=effective_max_length,
        ).to(self.device)
        gen_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in self._PIPELINE_ONLY_KWARGS
        }
        gen_kwargs.setdefault("max_new_tokens", 16)
        gen_kwargs.setdefault("do_sample", False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [[{"generated_text": t}] for t in decoded]


@model_registry.register("hf")
class HuggingFaceAdapter(ModelAdapter):
    """Transformers-backed safety scorer."""

    adapter_name = "hf"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._backend: Any | None = None
        self._raw_model: Any | None = None
        self._raw_device: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def allow_partial_predictions(self) -> bool:
        """Allow dropping samples that fail score resolution."""
        return _bool_arg(
            self.config.args.get("drop_failed_predictions", True),
            arg_name="drop_failed_predictions",
        )

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text", "code"),
            supports_category_outputs=True,
            notes=("transformers",),
        )

    def _model_name(self) -> str:
        name = self.config.model_name or self.config.args.get("pretrained")
        if not name:
            raise ValueError("hf adapter requires model_name or pretrained")
        return str(name)

    def _model_subfolder(self) -> str | None:
        """Return the configured model subfolder, if any."""
        subfolder = self.config.args.get(
            "model_subfolder"
        ) or self.config.args.get("subfolder")
        if subfolder is None:
            return None
        return str(subfolder)

    def _tokenizer_subfolder(self) -> str | None:
        """Return the configured tokenizer subfolder, if any."""
        subfolder = self.config.args.get("tokenizer_subfolder")
        if subfolder is None:
            return self._model_subfolder()
        return str(subfolder)

    def _prompt_for_sample(self, sample: PredictSample) -> str:
        if bool(self.config.args.get("apply_chat_template", False)):
            tokenizer = self._get_tokenizer()
            messages = sample_messages(sample)
            if uses_granite_guardian_chat_template(self.config):
                messages = prepare_granite_guardian_chat_messages(
                    sample,
                )
            elif uses_llama_guard_chat_template(self.config):
                messages = prepare_llama_guard_chat_messages(
                    sample,
                    tokenizer=tokenizer,
                )
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(
                    self.config.args.get("add_generation_prompt", True)
                ),
            )
        template = self.config.args.get("prompt_template")
        if template:
            return render_template(str(template), sample_context(sample))
        return sample_to_text(sample)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            transformers = importlib.import_module("transformers")
            tokenizer_name = (
                self.config.args.get("tokenizer") or self._model_name()
            )
            tokenizer_kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self.config.args.get("trust_remote_code", False)
                ),
                "revision": self.config.args.get("revision", "main"),
            }
            use_fast = self.config.args.get("use_fast")
            if isinstance(use_fast, bool):
                tokenizer_kwargs["use_fast"] = use_fast
            tokenizer_subfolder = self._tokenizer_subfolder()
            if tokenizer_subfolder is not None:
                tokenizer_kwargs["subfolder"] = tokenizer_subfolder
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_name,
                **tokenizer_kwargs,
            )
        return self._tokenizer

    def _resolve_torch_dtype(self) -> Any:
        """Resolve an optional torch dtype from config."""
        torch_dtype = self.config.args.get("torch_dtype")
        if torch_dtype is None or not isinstance(torch_dtype, str):
            return torch_dtype
        torch = importlib.import_module("torch")
        return getattr(torch, torch_dtype, torch_dtype)

    def _configured_device_map(self) -> Any:
        """Resolve the configured device_map from top-level args or model_kwargs."""
        device_map = self.config.args.get("device_map")
        if device_map is not None:
            return device_map
        model_kwargs = self.config.args.get("model_kwargs", {})
        if isinstance(model_kwargs, Mapping):
            return model_kwargs.get("device_map")
        return None

    def _load_model(self, transformers: Any, *, task: str) -> Any:
        """Load a model object when subfolder support is needed."""
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(
                self.config.args.get("trust_remote_code", False)
            ),
            "revision": self.config.args.get("revision", "main"),
        }
        model_subfolder = self._model_subfolder()
        if model_subfolder is not None:
            model_kwargs["subfolder"] = model_subfolder
        extra_model_kwargs = dict(self.config.args.get("model_kwargs", {}))
        device_map = self.config.args.get("device_map")
        if device_map is not None:
            extra_model_kwargs["device_map"] = device_map
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            extra_model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs.update(extra_model_kwargs)

        if task in {"text-classification", "raw-sequence-classification"}:
            model_cls = transformers.AutoModelForSequenceClassification
        elif task == "text2text-generation":
            model_cls = transformers.AutoModelForSeq2SeqLM
        else:
            model_cls = transformers.AutoModelForCausalLM
        return model_cls.from_pretrained(self._model_name(), **model_kwargs)

    def _resolve_device(self, torch: Any) -> Any:
        """Resolve the configured torch device."""
        configured = self.config.args.get("device", -1)
        if isinstance(configured, str):
            return torch.device(configured)
        if isinstance(configured, int):
            if configured < 0:
                return torch.device("cpu")
            return torch.device(f"cuda:{configured}")
        return torch.device("cpu")

    def _get_raw_model(self) -> tuple[Any, Any | None]:
        """Load a raw sequence classification model for direct logits access."""
        if self._raw_model is None:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            self._raw_model = self._load_model(
                transformers,
                task="raw-sequence-classification",
            )
            if self._configured_device_map() is None:
                self._raw_device = self._resolve_device(torch)
                self._raw_model.to(self._raw_device)
            else:
                self._raw_device = None
            self._raw_model.eval()
        return self._raw_model, self._raw_device

    def _raw_activation_name(
        self,
        *,
        model: Any | None = None,
        label_count: int | None = None,
    ) -> str:
        """Resolve the configured activation name for raw logits."""
        activation = self.config.args.get("activation")
        if activation is None:
            activation = self.config.args.get("function_to_apply")
        if activation is not None:
            return str(activation).lower()

        model_config = getattr(model, "config", None)
        if getattr(model_config, "problem_type", None) == (
            "multi_label_classification"
        ):
            return "sigmoid"

        if label_count is None:
            num_labels = getattr(model_config, "num_labels", None)
            if isinstance(num_labels, int) and num_labels > 0:
                label_count = num_labels
            else:
                id2label = getattr(model_config, "id2label", None)
                if isinstance(id2label, Mapping) and id2label:
                    label_count = len(id2label)

        if label_count == 1:
            return "sigmoid"
        return "softmax"

    @staticmethod
    def _default_device() -> int:
        """Pick the GPU with the most free memory, or CPU if none available.

        Respects ``CUDA_VISIBLE_DEVICES`` when set.  Queries
        ``nvidia-smi`` to avoid initializing CUDA contexts on every
        device.  Falls back to CPU (``-1``) when no GPU is usable.
        """
        import os

        # Respect CUDA_VISIBLE_DEVICES — if explicitly empty, use CPU.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip() == "":
            _log.info("No GPU detected — running on CPU")
            return -1

        # If CUDA_VISIBLE_DEVICES restricts to specific GPUs, use
        # device 0 (PyTorch re-indexes visible devices starting at 0).
        if visible is not None:
            _log.info(
                "CUDA_VISIBLE_DEVICES=%s — using device 0", visible
            )
            return 0

        # No env override — query nvidia-smi for the freest GPU.
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                best_gpu, best_free = 0, 0
                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    idx = int(parts[0].strip())
                    free_mb = int(parts[1].strip())
                    if free_mb > best_free:
                        best_free = free_mb
                        best_gpu = idx
                _log.info(
                    "Auto-selected GPU %d (%.1f GiB free)",
                    best_gpu,
                    best_free / 1024,
                )
                return best_gpu
        except FileNotFoundError:
            pass
        except Exception:
            _log.debug("nvidia-smi query failed", exc_info=True)

        # Last resort — check torch.
        try:
            import torch
            if torch.cuda.is_available():
                return 0
        except ImportError:
            pass
        _log.info("No GPU detected — running on CPU")
        return -1

    def _get_backend(self) -> Any:
        if self._backend is None:
            task = str(self.config.args.get("task", "text-classification"))
            transformers = importlib.import_module("transformers")
            _tf_logging = getattr(
                getattr(transformers, "utils", None), "logging", None
            )
            _prev_verbosity = None
            if _tf_logging is not None:
                _prev_verbosity = _tf_logging.get_verbosity()
                _tf_logging.set_verbosity_error()
            device_map = self._configured_device_map()
            # text2text-generation pipeline was removed in transformers 5.x;
            # wrap the model + tokenizer in a thin callable so the rest of
            # the adapter can call it like a pipeline. Handled before the
            # subfolder branch so subfolder seq2seq configs also route
            # through the wrapper instead of the removed pipeline.
            if task == "text2text-generation":
                torch = importlib.import_module("torch")
                if "device" in self.config.args:
                    seq2seq_device = self._resolve_device(torch)
                else:
                    idx = self._default_device()
                    seq2seq_device = (
                        torch.device("cpu") if idx < 0
                        else torch.device(f"cuda:{idx}")
                    )
                _cfg_max_length = self.config.args.get("max_length")
                self._backend = _Seq2SeqWrapper(
                    model=self._load_model(transformers, task=task),
                    tokenizer=self._get_tokenizer(),
                    device=seq2seq_device,
                    default_max_length=(
                        _cfg_max_length
                        if isinstance(_cfg_max_length, int)
                        and _cfg_max_length > 0
                        else None
                    ),
                )
                if _tf_logging is not None and _prev_verbosity is not None:
                    _tf_logging.set_verbosity(_prev_verbosity)
                return self._backend
            if (
                self._model_subfolder() is not None
                or self._tokenizer_subfolder() is not None
            ):
                pipeline_kwargs = {
                    "task": task,
                    "model": self._load_model(transformers, task=task),
                    "tokenizer": self._get_tokenizer(),
                }
                if device_map is None:
                    pipeline_kwargs["device"] = (
                        self.config.args["device"]
                        if "device" in self.config.args
                        else self._default_device()
                    )
                self._backend = transformers.pipeline(**pipeline_kwargs)
                if _tf_logging is not None and _prev_verbosity is not None:
                    _tf_logging.set_verbosity(_prev_verbosity)
                return self._backend
            pipeline_kwargs = {
                "task": task,
                "model": self._model_name(),
                "trust_remote_code": bool(
                    self.config.args.get("trust_remote_code", False)
                ),
                "revision": self.config.args.get("revision", "main"),
            }
            if self.config.args.get("use_fast") is not None:
                pipeline_kwargs["tokenizer"] = self._get_tokenizer()
            elif self.config.args.get("tokenizer") is not None:
                pipeline_kwargs["tokenizer"] = self.config.args["tokenizer"]
            else:
                pipeline_kwargs["tokenizer"] = self.config.args.get("tokenizer")
            model_kwargs = dict(self.config.args.get("model_kwargs", {}))
            top_level_device_map = self.config.args.get("device_map")
            if top_level_device_map is not None:
                model_kwargs["device_map"] = top_level_device_map
            else:
                if device_map is None:
                    pipeline_kwargs["device"] = (
                        self.config.args["device"]
                        if "device" in self.config.args
                        else self._default_device()
                    )
            torch_dtype = self._resolve_torch_dtype()
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if model_kwargs:
                pipeline_kwargs["model_kwargs"] = model_kwargs
            self._backend = transformers.pipeline(**pipeline_kwargs)
            if _tf_logging is not None and _prev_verbosity is not None:
                _tf_logging.set_verbosity(_prev_verbosity)
        return self._backend

    def _raw_activation(self, logits: Any, *, torch: Any, model: Any) -> Any:
        """Apply the configured activation to raw logits."""
        activation = self._raw_activation_name(model=model)
        if activation == "sigmoid":
            return torch.sigmoid(logits)
        if activation == "softmax":
            return torch.softmax(logits, dim=-1)
        if activation in {"identity", "none"}:
            raise ValueError(
                "raw-sequence-classification requires normalized probabilities; "
                "use sigmoid or softmax"
            )
        raise ValueError(f"unsupported raw activation: {activation}")

    def _aggregate_label_scores(self, scores: Sequence[float]) -> float:
        """Collapse per-label scores into one unsafe score."""
        if not scores:
            raise ValueError("raw sequence classification returned no scores")
        aggregation = str(
            self.config.args.get(
                "score_aggregation",
                self.config.args.get("label_score_aggregation", "max"),
            )
        ).lower()
        if aggregation in {"any", "max"}:
            return max(scores)
        if aggregation == "mean":
            return sum(scores) / len(scores)
        if aggregation == "first":
            return scores[0]
        if aggregation == "last":
            return scores[-1]
        raise ValueError(f"unsupported score_aggregation: {aggregation}")

    def _label_names(self, label_count: int) -> list[str]:
        """Resolve label names for raw sequence classification outputs."""
        configured = self.config.args.get("label_names")
        if isinstance(configured, Sequence) and not isinstance(configured, str):
            names = [str(name) for name in configured]
            if len(names) == label_count:
                return names

        model, _device = self._get_raw_model()
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, Mapping):
            return [
                str(
                    id2label.get(
                        index, id2label.get(str(index), f"LABEL_{index}")
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
                ("safe", "benign", "clean", "allow", "non-toxic"),
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
                ("unsafe", "harmful", "toxic", "violation", "reject"),
            )
        )
        return any(token in normalized for token in unsafe_labels)

    def _unsafe_score_from_label_scores(
        self,
        label_scores: Mapping[str, float],
        *,
        model: Any | None = None,
    ) -> float:
        """Resolve one unsafe score from raw per-label probabilities."""
        explicit_scores: list[float] = []
        unsafe_scores: list[float] = []
        safe_scores: list[float] = []
        has_non_safe_explicit = False

        for label, score in label_scores.items():
            numeric = float(score)
            explicit_score = self._score_from_mapping_value(
                {"label": label, "score": numeric},
                "label_score_mapping",
            )
            if explicit_score is not None:
                explicit_scores.append(explicit_score)
                if not self._label_is_safe(label):
                    has_non_safe_explicit = True
                continue
            if self._label_is_unsafe(label):
                unsafe_scores.append(numeric)
                continue
            if self._label_is_safe(label):
                safe_scores.append(numeric)

        candidate_scores = [*explicit_scores, *unsafe_scores]
        activation = self._raw_activation_name(
            model=model,
            label_count=len(label_scores),
        )
        if (
            not unsafe_scores
            and not has_non_safe_explicit
            and len(safe_scores) == 1
            and (activation == "softmax" or len(label_scores) == 1)
        ):
            candidate_scores.append(1.0 - safe_scores[0])

        if candidate_scores:
            return self._aggregate_label_scores(candidate_scores)
        return self._aggregate_label_scores(list(label_scores.values()))

    def _run_raw_sequence_classification(
        self,
        prompts: Sequence[str],
    ) -> list[Mapping[str, Any]]:
        """Run direct sequence classification and expose aggregated scores."""
        if not prompts:
            return []

        torch = importlib.import_module("torch")
        tokenizer = self._get_tokenizer()
        model, device = self._get_raw_model()
        is_decoder_only = (
            not getattr(model.config, "is_encoder_decoder", False)
            and getattr(tokenizer, "cls_token", None) is None
        )
        if is_decoder_only:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        tokenization_kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        max_length = self.config.args.get("max_length")
        if isinstance(max_length, int) and max_length > 0:
            tokenization_kwargs["max_length"] = max_length

        encoded = tokenizer(list(prompts), **tokenization_kwargs)
        if device is not None:
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = self._raw_activation(
                outputs.logits,
                torch=torch,
                model=model,
            )

        score_rows = probabilities.detach().float().cpu().tolist()
        if not score_rows:
            raise ValueError(
                "model returned empty logits for "
                f"{len(prompts)} input prompts; "
                "expected one score row per prompt"
            )
        label_names = self._label_names(len(score_rows[0]))
        normalized_outputs: list[Mapping[str, Any]] = []
        include_label_scores = bool(
            self.config.args.get("include_label_scores", False)
        )
        for score_row in score_rows:
            label_score_map = {
                label_names[index]: float(score)
                for index, score in enumerate(score_row)
            }
            top_index = max(
                range(len(score_row)),
                key=score_row.__getitem__,
            )
            output: dict[str, Any] = {
                "unsafe_score": self._unsafe_score_from_label_scores(
                    label_score_map,
                    model=model,
                ),
                "top_label": label_names[top_index],
                "top_label_score": score_row[top_index],
            }
            if include_label_scores:
                output["label_scores"] = label_score_map
            normalized_outputs.append(output)
        return normalized_outputs

    def _run_backend(self, prompts: Sequence[str]) -> list[Any]:
        task = str(self.config.args.get("task", "text-classification"))
        backend = self._get_backend()
        kwargs = dict(self.config.args.get("generation_kwargs", {}))
        tokenizer = getattr(backend, "tokenizer", None)
        generation_task = task in {"text-generation", "text2text-generation"}
        can_batch_generation = True
        if generation_task and tokenizer is not None:
            can_batch_generation = (
                getattr(tokenizer, "pad_token_id", None) is not None
            )
        pipeline_batch_size = self.config.args.get("pipeline_batch_size")
        if isinstance(pipeline_batch_size, int) and pipeline_batch_size > 0:
            if not (generation_task and not can_batch_generation):
                kwargs.setdefault("batch_size", pipeline_batch_size)
        aggregation = self.config.args.get("label_score_aggregation")
        if task == "text-classification":
            kwargs.setdefault("truncation", True)
            max_length = self.config.args.get("max_length")
            if isinstance(max_length, int) and max_length > 0:
                kwargs.setdefault("max_length", max_length)
            if aggregation is not None:
                kwargs.setdefault("top_k", None)
                kwargs.setdefault("function_to_apply", "sigmoid")
        if task == "text-generation":
            kwargs.setdefault("return_full_text", False)
        if task in {"text-generation", "text2text-generation"}:
            kwargs.setdefault("do_sample", False)
            kwargs.setdefault("max_new_tokens", 16)
            kwargs.setdefault("truncation", True)
            if tokenizer is not None:
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None:
                    kwargs.setdefault("pad_token_id", eos)
            max_length = self.config.args.get("max_length")
            if isinstance(max_length, int) and max_length > 0:
                # For text2text pipelines (e.g. T5), max_length conflicts
                # with max_new_tokens and truncation is unreliable. Pre-
                # truncate inputs via the tokenizer instead.
                if task == "text2text-generation":
                    _tok = getattr(backend, "tokenizer", None)
                    if _tok is not None:
                        prompts = [
                            _tok.decode(
                                _tok.encode(
                                    p,
                                    max_length=max_length,
                                    truncation=True,
                                ),
                                skip_special_tokens=True,
                            )
                            for p in prompts
                        ]
                else:
                    kwargs["max_length"] = max_length
            elif "max_new_tokens" in kwargs:
                # No user-supplied max_length; disable it so HF doesn't
                # warn about both max_length and max_new_tokens being
                # set with conflicting defaults.
                kwargs.setdefault("max_length", None)
        try:
            import transformers as _tf
            _prev = _tf.utils.logging.get_verbosity()
            _tf.utils.logging.set_verbosity_error()
        except Exception:
            _prev = None
        try:
            outputs = backend(list(prompts), **kwargs)
        finally:
            if _prev is not None:
                _tf.utils.logging.set_verbosity(_prev)
        if isinstance(outputs, Mapping):
            return [outputs]
        normalized_outputs = []
        for index, item in enumerate(list(outputs), start=1):
            if isinstance(item, list):
                if task == "text-classification":
                    normalized_outputs.append(item)
                else:
                    if len(item) != 1:
                        raise ValueError(
                            "hf backend returned "
                            f"{len(item)} candidates for input sample {index}; "
                            "expected exactly one output per sample"
                        )
                    normalized_outputs.append(item[0])
                continue
            normalized_outputs.append(item)
        return normalized_outputs

    def _score_from_mapping_value(
        self,
        mapping: Mapping[str, Any],
        key: str,
        fallback: float | None = None,
    ) -> float | None:
        """Resolve a configured explicit score mapping."""
        raw_mapping = self.config.args.get(key, {})
        if not isinstance(raw_mapping, Mapping):
            return None
        explicit_mapping = {
            str(k).strip().lower(): v for k, v in raw_mapping.items()
        }

        raw_label = mapping.get("label")
        if raw_label is None:
            return None
        normalized_label = str(raw_label).strip().lower()
        if normalized_label not in explicit_mapping:
            return None

        mapping_value = explicit_mapping[normalized_label]
        if isinstance(mapping_value, (int, float)):
            return float(mapping_value)
        if not isinstance(mapping_value, str):
            return None

        normalized_value = mapping_value.strip().lower()
        if normalized_value in {"score", "unsafe_score"}:
            score = mapping.get(normalized_value)
            if score is None:
                return fallback
            return float(score)
        if normalized_value in {"1-score", "invert_score"}:
            score = mapping.get("score")
            if score is None:
                return fallback
            return 1.0 - float(score)
        if normalized_value in {"safe", "false"}:
            return 0.0
        if normalized_value in {"unsafe", "true"}:
            return 1.0
        return fallback

    def _score_from_text_mapping(self, value: Any) -> float | None:
        """Resolve explicit text-to-score mappings for generated outputs."""
        if value is None:
            return None
        text_mapping = self.config.args.get("text_score_mapping", {})
        if not isinstance(text_mapping, Mapping):
            return None

        normalized_text = str(value).strip().lower()
        mapped = text_mapping.get(normalized_text)
        if mapped is None:
            for key, candidate in text_mapping.items():
                if not isinstance(key, str):
                    continue
                normalized_key = key.strip().lower()
                if not normalized_text.startswith(normalized_key):
                    continue
                if len(normalized_text) == len(normalized_key):
                    mapped = candidate
                    break
                suffix = normalized_text[len(normalized_key)]
                if suffix.isspace() or suffix in {":", ",", ".", ";", "!", "?"}:
                    mapped = candidate
                    break
        if mapped is None and normalized_text:
            prefix_matches: list[Any] = []
            for key, candidate in text_mapping.items():
                if not isinstance(key, str):
                    continue
                normalized_key = key.strip().lower()
                if normalized_key.startswith(normalized_text):
                    prefix_matches.append(candidate)
            if len(prefix_matches) == 1:
                mapped = prefix_matches[0]
        if mapped is None:
            return None

        if isinstance(mapped, (int, float)):
            return float(mapped)
        if not isinstance(mapped, str):
            return None
        normalized_value = mapped.strip().lower()
        if normalized_value in {"safe", "false"}:
            return 0.0
        if normalized_value in {"unsafe", "true"}:
            return 1.0
        try:
            return float(normalized_value)
        except ValueError:
            return None

    def _configured_generated_text(self, value: Any) -> Any:
        """Optionally narrow generated text to one configured line."""
        line_index = self.config.args.get("generated_text_line_index")
        if line_index is None or not isinstance(value, str):
            return value
        if not isinstance(line_index, int):
            return value
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if 0 <= line_index < len(lines):
            return lines[line_index]
        return value

    def _unsafe_score_from_output(self, output: Any) -> float:
        unsafe_labels = tuple(
            str(label).lower()
            for label in self.config.args.get(
                "unsafe_labels",
                ("unsafe", "harmful", "toxic", "violation", "reject"),
            )
        )
        safe_labels = tuple(
            str(label).lower()
            for label in self.config.args.get(
                "safe_labels",
                ("safe", "benign", "clean", "allow", "non-toxic"),
            )
        )

        if isinstance(output, str):
            explicit_score = self._score_from_text_mapping(output)
            if explicit_score is not None:
                return explicit_score

        if isinstance(output, list) and output:
            if isinstance(output[0], Mapping):
                output = output[0]

        if isinstance(output, Mapping):
            explicit_score = self._score_from_mapping_value(
                output,
                "label_score_mapping",
            )
            if explicit_score is not None:
                return explicit_score
            if "unsafe_score" in output:
                return float(output["unsafe_score"])
            if "generated_text" in output:
                generated_text = output["generated_text"]
                if isinstance(generated_text, list) and generated_text:
                    last_item = generated_text[-1]
                    if (
                        isinstance(last_item, Mapping)
                        and "content" in last_item
                    ):
                        content = self._configured_generated_text(
                            last_item["content"]
                        )
                        explicit_score = self._score_from_text_mapping(content)
                        if explicit_score is not None:
                            return explicit_score
                        return resolve_score(content)
                generated_text = self._configured_generated_text(generated_text)
                explicit_score = self._score_from_text_mapping(generated_text)
                if explicit_score is not None:
                    return explicit_score
                return resolve_score(generated_text)
            label = str(output.get("label", "")).lower()
            score = output.get("score")
            if score is not None:
                numeric = float(score)
                if any(token in label for token in unsafe_labels):
                    return numeric
                if any(token in label for token in safe_labels):
                    return 1.0 - numeric
                if "safe" in label and "unsafe" not in label:
                    return 1.0 - numeric
                return numeric
            return resolve_score(output)

        return resolve_score(output)

    def _unsafe_score_from_output_list(self, output: list[Any]) -> float:
        """Aggregate a multi-label text-classification output into one score."""
        aggregation = self.config.args.get("label_score_aggregation")
        if aggregation != "max":
            if output and isinstance(output[0], Mapping):
                return self._unsafe_score_from_output(output[0])
            return resolve_score(output)

        score_labels = self._configured_score_labels()
        scores: list[float] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if score_labels:
                label = item.get("label")
                if label is None:
                    continue
                normalized_label = str(label).strip().lower()
                if normalized_label not in score_labels:
                    continue
            explicit_score = self._score_from_mapping_value(
                item,
                "label_score_mapping",
            )
            if explicit_score is not None:
                scores.append(explicit_score)
                continue
            score = item.get("score")
            if score is not None:
                scores.append(float(score))

        if not scores:
            raise ValueError("could not resolve unsafe score from label list")
        return max(scores)

    def _configured_score_labels(self) -> tuple[str, ...]:
        """Return the subset of labels to include in multi-label scoring."""
        configured = self.config.args.get("score_labels")
        if not isinstance(configured, Sequence) or isinstance(configured, str):
            return ()

        labels: list[str] = []
        for label in configured:
            normalized = str(label).strip().lower()
            if normalized:
                labels.append(normalized)
        return tuple(labels)

    def _validate_output_count(
        self,
        outputs: Sequence[Any],
        *,
        expected_count: int,
    ) -> None:
        """Ensure the backend produced exactly one output per input sample."""
        actual_count = len(outputs)
        if actual_count != expected_count:
            raise ValueError(
                "hf backend returned "
                f"{actual_count} outputs for {expected_count} input samples; "
                "expected exactly one output per sample"
            )

    def _metadata_output(self, output: Any) -> Mapping[str, Any] | None:
        """Return the output mapping that should populate prediction metadata."""
        if isinstance(output, Mapping):
            return output
        if (
            self.config.args.get("label_score_aggregation") is None
            and isinstance(output, list)
            and output
            and isinstance(output[0], Mapping)
        ):
            return output[0]
        return None

    def _extract_categories(self, output: Any) -> tuple[str, ...]:
        """Extract predicted categories from raw output.

        Looks for ``#type: CWE-XX`` in generated text to
        support code vulnerability detection responses.
        Applies ``generated_text_line_index`` filtering so
        categories are extracted from the same text region
        used for scoring.
        """
        text = ""
        if isinstance(output, str):
            text = output
        elif isinstance(output, Mapping):
            gen = output.get("generated_text", "")
            if isinstance(gen, list) and gen:
                last = gen[-1]
                if isinstance(last, Mapping):
                    text = str(last.get("content", ""))
                else:
                    text = str(last)
            elif isinstance(gen, str):
                text = gen
        if text:
            text = self._configured_generated_text(text)
            return extract_judge_categories(text)
        return ()

    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        if not samples:
            return []

        prompts = [self._prompt_for_sample(sample) for sample in samples]
        started = time.perf_counter()
        task = str(self.config.args.get("task", "text-classification"))
        if task == "raw-sequence-classification":
            outputs = self._run_raw_sequence_classification(prompts)
        else:
            outputs = self._run_backend(prompts)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._validate_output_count(outputs, expected_count=len(samples))
        latency_ms = elapsed_ms / len(samples)

        predictions: list[NormalizedPrediction] = []
        drop_failed = self.allow_partial_predictions
        for sample, output in zip(samples, outputs):
            try:
                if isinstance(output, list):
                    score = self._unsafe_score_from_output_list(
                        output,
                    )
                else:
                    score = self._unsafe_score_from_output(output)
            except Exception:
                if drop_failed:
                    _log.warning(
                        "Score resolution failed for sample %s; "
                        "dropping from predictions",
                        sample.id,
                    )
                    continue
                raise
            if math.isnan(score) or math.isinf(score):
                if drop_failed:
                    _log.warning(
                        "Score for sample %s is %s; dropping from predictions",
                        sample.id,
                        score,
                    )
                    continue
                raise ValueError(f"Score for sample {sample.id} is {score}")
            metadata = {
                "adapter": self.adapter_name,
                "model_name": self._model_name(),
                "task": task,
            }
            metadata_output = self._metadata_output(output)
            if metadata_output is not None:
                for key in (
                    "label",
                    "score",
                    "top_label",
                    "top_label_score",
                    "label_scores",
                ):
                    if key in metadata_output:
                        metadata[key] = metadata_output[key]
            # Extract predicted categories from text
            # output (e.g. #type: CWE-XX from code vuln).
            predicted_categories = self._extract_categories(output)
            predictions.append(
                NormalizedPrediction(
                    sample_id=sample.id,
                    unsafe_score=score,
                    unsafe_label=score >= threshold,
                    threshold=threshold,
                    latency_ms=latency_ms,
                    predicted_categories=(predicted_categories),
                    metadata=metadata,
                )
            )
        return predictions
