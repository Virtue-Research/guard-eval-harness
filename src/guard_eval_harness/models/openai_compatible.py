"""OpenAI-compatible backend adapter with concurrency and retry."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error

import httpx

from guard_eval_harness.models._async_dispatch import run_async_batch
from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.models.templates import (
    async_json_post_with_retry,
    env_value,
    extract_judge_categories,
    join_url,
    json_post_with_retry,
    render_value,
    resolve_score,
    sample_context,
    sample_messages_openai,
    sample_has_media,
    sample_to_text,
)
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    PredictSample,
)

_MAX_CONCURRENCY = 2000
_AUTH_FAILURE_CODES = {401, 403}
_log = logging.getLogger(__name__)


def _bool_arg(value: Any, *, arg_name: str) -> bool:
    """Parse a config arg as a strict boolean-like value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{arg_name} must be a boolean or boolean-like string")


@model_registry.register("openai_compatible")
class OpenAICompatibleAdapter(ModelAdapter):
    """OpenAI-style chat or completions adapter.

    Supports concurrent API requests within a batch, automatic retry
    with exponential backoff on transient failures, and token-usage
    tracking from the provider response.
    """

    adapter_name = "openai_compatible"

    @property
    def capabilities(self) -> AdapterCapabilities:
        concurrency = int(self.config.args.get("concurrency", 1))
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=True,
            concurrency=concurrency > 1,
            cost_estimation=False,
            token_accounting=True,
            supported_input_modalities=("text", "image", "code"),
            supports_category_outputs=True,
            notes=("openai-compatible",),
        )

    @property
    def allow_partial_predictions(self) -> bool:
        """Allow dropped samples after retries instead of fake predictions."""
        return _bool_arg(
            self.config.args.get("drop_failed_predictions", True),
            arg_name="drop_failed_predictions",
        )

    def _model_name(self) -> str:
        name = self.config.model_name or self.config.args.get("model")
        if not name:
            raise ValueError(
                "openai_compatible adapter requires model_name or model"
            )
        return str(name)

    def _endpoint(self) -> str:
        url = self.config.args.get("url") or self.config.args.get("base_url")
        if url:
            return str(url)
        mode = str(self.config.args.get("mode", "chat"))
        root = str(self.config.args.get("root_url", "https://api.openai.com"))
        if mode == "completions":
            return join_url(root, "/v1/completions")
        return join_url(root, "/v1/chat/completions")

    def _mode(self) -> str:
        return str(self.config.args.get("mode", "chat"))

    def _headers(self) -> dict[str, str]:
        headers = dict(self.config.args.get("headers", {}))
        token = env_value(
            self.config.args.get("api_key_env"),
            self.config.args.get("api_key"),
        )
        if token:
            prefix = str(self.config.args.get("auth_prefix", "Bearer"))
            header_name = str(
                self.config.args.get("auth_header", "Authorization")
            )
            headers.setdefault(header_name, f"{prefix} {token}".strip())
        return headers

    def _requires_api_key(self) -> bool:
        configured = self.config.args.get("require_api_key")
        if configured is not None:
            return _bool_arg(configured, arg_name="require_api_key")
        if (
            "api_key_env" in self.config.args
            or "api_key" in self.config.args
        ):
            return True
        return "api.openai.com" in self._endpoint().lower()

    def _api_key_env_name(self) -> str:
        return str(self.config.args.get("api_key_env", "OPENAI_API_KEY"))

    def _has_auth_header(self, headers: Mapping[str, str]) -> bool:
        header_name = str(
            self.config.args.get("auth_header", "Authorization")
        ).lower()
        return any(
            key.lower() == header_name and str(value).strip()
            for key, value in headers.items()
        )

    def _validate_api_key_headers(
        self,
        headers: Mapping[str, str],
    ) -> None:
        if self._requires_api_key() and not self._has_auth_header(headers):
            key_env = self._api_key_env_name()
            raise ValueError(
                "OpenAI-compatible API key is missing or empty. "
                f"Set {key_env} or pass api_key in model args."
            )

    def _raise_for_auth_error(self, exc: BaseException) -> None:
        if (
            isinstance(exc, urllib_error.HTTPError)
            and exc.code in _AUTH_FAILURE_CODES
        ):
            reason = exc.reason or "authentication failed"
            key_env = self._api_key_env_name()
            raise ValueError(
                "OpenAI-compatible API authentication failed "
                f"(HTTP {exc.code}: {reason}). "
                f"Check {key_env} or model args.api_key."
            ) from exc

    def _request_payload(self, sample: PredictSample) -> dict[str, Any]:
        context = sample_context(sample)
        payload_template = self.config.args.get("payload_template")
        if payload_template is not None:
            payload = render_value(payload_template, context)
            if isinstance(payload, Mapping):
                return dict(payload)
            return {"prompt": payload}

        mode = self._mode()
        max_completion_tokens = self.config.args.get(
            "max_completion_tokens",
        )
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "temperature": float(self.config.args.get("temperature", 0.0)),
        }
        if max_completion_tokens is not None and mode != "completions":
            payload["max_completion_tokens"] = int(max_completion_tokens)
        else:
            payload["max_tokens"] = int(
                self.config.args.get("max_tokens", 1)
            )
        has_images = sample_has_media(sample)
        if mode == "completions":
            if (
                has_images
                and self._multimodal_prompt_template_mode() != "text_only"
            ):
                raise ValueError(
                    "openai_compatible completions mode does not support image "
                    "content; use chat mode or set "
                    "prompt_template_multimodal_mode=text_only"
                )
            template = self.config.args.get("prompt_template")
            if template:
                payload["prompt"] = str(render_value(template, context))
            else:
                payload["prompt"] = sample_to_text(sample)
        else:
            payload["messages"] = sample_messages_openai(sample)
            system_prompt = self.config.args.get("system_prompt")
            rendered_system_prompt: str | None = None
            if system_prompt:
                rendered_system_prompt = str(
                    render_value(system_prompt, context)
                )
                payload["messages"].insert(
                    0,
                    {"role": "system", "content": rendered_system_prompt},
                )
            template = self.config.args.get("prompt_template")
            if template:
                if (
                    has_images
                    and self._multimodal_prompt_template_mode() != "text_only"
                ):
                    raise ValueError(
                        "prompt_template would drop image content; set "
                        "prompt_template_multimodal_mode=text_only to opt in"
                    )
                templated_messages = [
                    {
                        "role": "user",
                        "content": str(render_value(template, context)),
                    }
                ]
                if rendered_system_prompt is not None:
                    payload["messages"] = [
                        {
                            "role": "system",
                            "content": rendered_system_prompt,
                        },
                        *templated_messages,
                    ]
                else:
                    payload["messages"] = templated_messages

        extra_payload = self.config.args.get("extra_payload", {})
        if isinstance(extra_payload, Mapping):
            payload.update(render_value(extra_payload, context))
        return payload

    def _multimodal_prompt_template_mode(self) -> str:
        """Resolve how prompt templates should behave for image inputs."""
        mode = self.config.args.get(
            "prompt_template_multimodal_mode",
            "error",
        )
        normalized = str(mode).strip().lower()
        if normalized not in {"error", "text_only"}:
            raise ValueError(
                "prompt_template_multimodal_mode must be 'error' or 'text_only'"
            )
        return normalized

    def _response_payload(self, response: Any) -> Any:
        response_path = self.config.args.get("response_path")
        if response_path:
            from guard_eval_harness.models.templates import (
                extract_path,
            )

            return extract_path(response, str(response_path))
        return response

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage counters from the API response."""
        if not isinstance(response, Mapping):
            return {}
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return {}
        result: dict[str, int] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                result[key] = value
        return result

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract raw text from an OpenAI-style response."""
        if not isinstance(response, Mapping):
            return str(response) if response else ""
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                msg = first.get("message")
                if isinstance(msg, Mapping):
                    content = msg.get("content")
                    if content:
                        return str(content)
                text = first.get("text")
                if text:
                    return str(text)
        for key in ("text", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _build_prediction(
        self,
        sample: PredictSample,
        response: Any,
        *,
        endpoint: str,
        elapsed_ms: float,
        threshold: float,
    ) -> NormalizedPrediction:
        usage = self._extract_usage(response)
        response_payload = self._response_payload(response)
        score = resolve_score(
            response_payload,
            score_path=self.config.args.get("score_path"),
        )
        predicted_categories: tuple[str, ...] = ()
        score_path = self.config.args.get("score_path")
        response_text = ""
        if score_path:
            from guard_eval_harness.models.templates import (
                extract_path,
            )

            try:
                extracted = extract_path(response_payload, score_path)
                if isinstance(extracted, str):
                    response_text = extracted
            except (KeyError, IndexError, ValueError):
                pass
        if not response_text:
            response_text = self._extract_response_text(response_payload)
        if not response_text:
            response_text = self._extract_response_text(response)
        if response_text:
            predicted_categories = extract_judge_categories(response_text)
        metadata: dict[str, Any] = {
            "adapter": self.adapter_name,
            "endpoint": endpoint,
            "model_name": self._model_name(),
            "mode": self._mode(),
        }
        if usage:
            metadata["usage"] = usage
        return NormalizedPrediction(
            sample_id=sample.id,
            unsafe_score=score,
            unsafe_label=score >= threshold,
            threshold=threshold,
            latency_ms=elapsed_ms,
            predicted_categories=predicted_categories,
            metadata=metadata,
        )

    def _predict_one(
        self,
        sample: PredictSample,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
        retries: int,
        backoff: float,
        threshold: float,
    ) -> NormalizedPrediction:
        """Run inference for a single sample with retry."""
        payload = self._request_payload(sample)
        started = time.perf_counter()
        response = json_post_with_retry(
            endpoint,
            payload,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._build_prediction(
            sample,
            response,
            endpoint=endpoint,
            elapsed_ms=elapsed_ms,
            threshold=threshold,
        )

    async def _apredict_one(
        self,
        client: httpx.AsyncClient,
        sample: PredictSample,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
        retries: int,
        backoff: float,
        threshold: float,
    ) -> NormalizedPrediction:
        payload = self._request_payload(sample)
        started = time.perf_counter()
        response = await async_json_post_with_retry(
            client,
            endpoint,
            payload,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._build_prediction(
            sample,
            response,
            endpoint=endpoint,
            elapsed_ms=elapsed_ms,
            threshold=threshold,
        )

    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        if not samples:
            return []

        endpoint = self._endpoint()
        headers = self._headers()
        self._validate_api_key_headers(headers)
        timeout = float(self.config.args.get("timeout", 30.0))
        retries = int(self.config.args.get("retries", 0))
        backoff = float(self.config.args.get("retry_backoff", 1.0))
        drop_failed_predictions = self.allow_partial_predictions
        concurrency = min(
            int(self.config.args.get("concurrency", 1)),
            _MAX_CONCURRENCY,
        )
        concurrency = max(1, concurrency)

        if concurrency <= 1:
            predictions: list[NormalizedPrediction] = []
            failed_id: str | None = None
            last_exc: Exception | None = None
            for sample in samples:
                try:
                    predictions.append(
                        self._predict_one(
                            sample,
                            endpoint=endpoint,
                            headers=headers,
                            timeout=timeout,
                            retries=retries,
                            backoff=backoff,
                            threshold=threshold,
                        )
                    )
                except Exception as exc:
                    self._raise_for_auth_error(exc)
                    _log.error(
                        "prediction failed for sample %s: %s",
                        sample.id,
                        exc,
                    )
                    failed_id = sample.id
                    last_exc = exc
                    if not drop_failed_predictions:
                        break
            if last_exc is not None:
                if drop_failed_predictions:
                    failed_ids = [
                        sample.id
                        for sample in samples
                        if sample.id
                        not in {
                            prediction.sample_id for prediction in predictions
                        }
                    ]
                    _log.warning(
                        "%d/%d predictions dropped after retries: %s",
                        len(failed_ids),
                        len(samples),
                        ", ".join(failed_ids),
                    )
                    return predictions
                raise RuntimeError(
                    f"prediction failed for sample {failed_id}: {last_exc}"
                ) from last_exc
            return predictions

        index_map: dict[int, NormalizedPrediction] = {}
        failed: list[tuple[int, str, Exception]] = []
        remaining_samples: Sequence[PredictSample] = samples
        offset = 0
        if self._requires_api_key():
            try:
                index_map[0] = self._predict_one(
                    samples[0],
                    endpoint=endpoint,
                    headers=headers,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    threshold=threshold,
                )
            except Exception as exc:
                self._raise_for_auth_error(exc)
                _log.error(
                    "prediction failed for sample %s: %s",
                    samples[0].id,
                    exc,
                )
                if not drop_failed_predictions:
                    raise RuntimeError(
                        f"prediction failed for sample {samples[0].id}: {exc}"
                    ) from exc
                failed.append((0, samples[0].id, exc))
            remaining_samples = samples[1:]
            offset = 1
            if not remaining_samples:
                return [index_map[i] for i in sorted(index_map)]

        async def _factory(
            client: httpx.AsyncClient,
            sample: PredictSample,
        ) -> NormalizedPrediction:
            return await self._apredict_one(
                client,
                sample,
                endpoint=endpoint,
                headers=headers,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
                threshold=threshold,
            )

        raw_results = run_async_batch(
            _factory,
            remaining_samples,
            concurrency=concurrency,
            timeout=timeout,
        )

        for idx, result in enumerate(raw_results):
            original_idx = idx + offset
            if isinstance(result, BaseException):
                self._raise_for_auth_error(result)
                sample_id = samples[original_idx].id
                _log.error(
                    "prediction failed for sample %s: %s",
                    sample_id,
                    result,
                )
                failed.append(
                    (
                        original_idx,
                        sample_id,
                        result
                        if isinstance(result, Exception)
                        else Exception(str(result)),
                    ),
                )
            else:
                index_map[original_idx] = result

        if failed:
            ids = ", ".join(sid for _, sid, _ in failed)
            if not drop_failed_predictions:
                first_exc = failed[0][2]
                raise RuntimeError(
                    f"prediction failed for sample(s) {ids}: {first_exc}"
                ) from first_exc
            _log.warning(
                "%d/%d predictions dropped after retries: %s",
                len(failed),
                len(samples),
                ids,
            )

        return [index_map[i] for i in range(len(samples)) if i in index_map]
