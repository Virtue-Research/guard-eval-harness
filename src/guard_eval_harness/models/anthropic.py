"""Anthropic Messages API adapter with concurrency and retry."""

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
    json_post_with_retry,
    render_value,
    resolve_score,
    sample_context,
    sample_has_media,
    sample_messages_openai,
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


@model_registry.register("anthropic")
class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API adapter.

    Talks directly to ``https://api.anthropic.com/v1/messages``
    using the Anthropic-native request/response format.
    """

    adapter_name = "anthropic"

    @property
    def allow_partial_predictions(self) -> bool:
        """Drop failed samples instead of crashing the run."""
        return True

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
        )

    def _model_name(self) -> str:
        name = self.config.model_name or self.config.args.get("model")
        if not name:
            raise ValueError(
                "anthropic adapter requires model_name or args.model"
            )
        return str(name)

    def _endpoint(self) -> str:
        root = self.config.args.get(
            "root_url", "https://api.anthropic.com"
        )
        return f"{root.rstrip('/')}/v1/messages"

    def _headers(self) -> dict[str, str]:
        token = env_value(
            self.config.args.get("api_key_env", "ANTHROPIC_API_KEY"),
            self.config.args.get("api_key"),
        )
        headers: dict[str, str] = {
            "content-type": "application/json",
            "anthropic-version": self.config.args.get(
                "anthropic_version", "2023-06-01"
            ),
        }
        if token:
            headers["x-api-key"] = token
        extra = self.config.args.get("headers", {})
        if isinstance(extra, Mapping):
            headers.update(extra)
        return headers

    def _api_key_env_name(self) -> str:
        return str(self.config.args.get("api_key_env", "ANTHROPIC_API_KEY"))

    def _validate_api_key_headers(
        self,
        headers: Mapping[str, str],
    ) -> None:
        has_api_key = any(
            key.lower() == "x-api-key" and str(value).strip()
            for key, value in headers.items()
        )
        if not has_api_key:
            key_env = self._api_key_env_name()
            raise ValueError(
                "Anthropic API key is missing or empty. "
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
                "Anthropic API authentication failed "
                f"(HTTP {exc.code}: {reason}). "
                f"Check {key_env} or model args.api_key."
            ) from exc

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

    def _request_payload(
        self, sample: PredictSample
    ) -> dict[str, Any]:
        context = sample_context(sample)
        prompt_template = self.config.args.get("prompt_template")
        if prompt_template:
            # A rendered prompt template collapses the conversation to
            # a single text turn, which would silently drop image
            # blocks and other original turns from multimodal samples.
            # Require an explicit opt-in to that behavior.
            if (
                sample_has_media(sample)
                and self._multimodal_prompt_template_mode() != "text_only"
            ):
                raise ValueError(
                    "prompt_template would drop image content; set "
                    "prompt_template_multimodal_mode=text_only to opt in"
                )
            rendered = str(render_value(prompt_template, context))
            openai_messages = [
                {"role": "user", "content": rendered}
            ]
        else:
            openai_messages = sample_messages_openai(sample)

        # Anthropic separates system from messages.
        # Accumulate all system turns into one block.
        system_parts: list[str] = []
        messages = []
        for msg in openai_messages:
            content = msg["content"]
            # Convert structured content to Anthropic format.
            # Supports text and image_url (base64 data URIs).
            if isinstance(content, list):
                anthropic_parts: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "text")
                        if block_type == "text":
                            anthropic_parts.append(
                                {
                                    "type": "text",
                                    "text": block.get("text", ""),
                                }
                            )
                        elif block_type == "image_url":
                            url = block.get("image_url", {}).get(
                                "url", ""
                            )
                            if url.startswith("data:"):
                                header, _, b64data = url.partition(
                                    ";base64,"
                                )
                                media_type = header.replace(
                                    "data:", ""
                                )
                                anthropic_parts.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64data,
                                        },
                                    }
                                )
                            else:
                                anthropic_parts.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": url,
                                        },
                                    }
                                )
                        else:
                            raise ValueError(
                                f"Anthropic adapter does not support "
                                f"content block type "
                                f"{block_type!r}"
                            )
                    else:
                        anthropic_parts.append(
                            {"type": "text", "text": str(block)}
                        )
                content = anthropic_parts
            if msg["role"] == "system":
                if isinstance(content, list):
                    has_non_text = any(
                        isinstance(b, dict)
                        and b.get("type") != "text"
                        for b in content
                    )
                    if has_non_text:
                        raise ValueError(
                            "Anthropic adapter does not support "
                            "non-text content in system messages"
                        )
                    system_parts.append(
                        " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict)
                            and b.get("type") == "text"
                        )
                    )
                else:
                    system_parts.append(str(content))
            else:
                messages.append(
                    {"role": msg["role"], "content": content}
                )
        config_system = self.config.args.get("system_prompt")
        if config_system:
            rendered = str(
                render_value(config_system, context)
            ).strip()
            system_parts.insert(0, rendered)

        system_text = "\n\n".join(system_parts)

        payload: dict[str, Any] = {
            "model": self._model_name(),
            "max_tokens": int(
                self.config.args.get("max_tokens", 8192)
            ),
            "messages": messages,
        }
        if system_text:
            if self.config.args.get("cache_system_prompt"):
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system_text

        temperature = self.config.args.get("temperature")
        if temperature is not None:
            payload["temperature"] = float(temperature)

        return payload

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if not isinstance(response, Mapping):
            return str(response) if response else ""
        content = response.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, Mapping):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
            if parts:
                return "\n".join(parts)
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        if not isinstance(response, Mapping):
            return {}
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return {}
        result: dict[str, int] = {}
        for src, dst in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            value = usage.get(src)
            if isinstance(value, int):
                result[dst] = value
        if "prompt_tokens" in result and "completion_tokens" in result:
            result["total_tokens"] = (
                result["prompt_tokens"] + result["completion_tokens"]
            )
        return result

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
        response_text = self._extract_response_text(response)
        score = resolve_score(response_text)

        predicted_categories: tuple[str, ...] = ()
        if response_text:
            predicted_categories = extract_judge_categories(
                response_text
            )

        metadata: dict[str, Any] = {
            "adapter": self.adapter_name,
            "endpoint": endpoint,
            "model_name": self._model_name(),
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
        timeout = float(self.config.args.get("timeout", 120.0))
        retries = int(self.config.args.get("retries", 3))
        backoff = float(self.config.args.get("retry_backoff", 1.0))
        drop_failed = self.allow_partial_predictions
        concurrency = min(
            int(self.config.args.get("concurrency", 1)),
            _MAX_CONCURRENCY,
        )
        concurrency = max(1, concurrency)

        if concurrency <= 1:
            predictions: list[NormalizedPrediction] = []
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
                    if not drop_failed:
                        raise
            return predictions

        index_map: dict[int, NormalizedPrediction] = {}
        failed: list[tuple[int, str, Exception]] = []
        remaining_samples: Sequence[PredictSample] = samples
        offset = 0
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
            if not drop_failed:
                raise
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

        if failed and not drop_failed:
            _, sid, exc = failed[0]
            raise RuntimeError(
                f"prediction failed for sample {sid}: {exc}"
            ) from exc
        if failed:
            failed_ids = [sid for _, sid, _ in failed]
            _log.warning(
                "%d/%d predictions dropped: %s",
                len(failed_ids),
                len(samples),
                ", ".join(failed_ids),
            )
        return [
            index_map[i]
            for i in sorted(index_map)
        ]
