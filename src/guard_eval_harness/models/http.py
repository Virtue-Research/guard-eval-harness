"""Generic HTTP backend adapter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any, Mapping, Sequence

from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.models.openai_compatible import _bool_arg
from guard_eval_harness.models.templates import (
    env_value,
    extract_judge_categories,
    json_post_with_retry,
    render_value,
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

_MAX_CONCURRENCY = 1024
_log = logging.getLogger(__name__)


@model_registry.register("http")
class HttpAdapter(ModelAdapter):
    """Generic JSON-over-HTTP scoring adapter."""

    adapter_name = "http"

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from common structured HTTP responses."""
        if isinstance(response, str):
            return response
        if not isinstance(response, Mapping):
            return ""

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        return content
                text = first.get("text")
                if isinstance(text, str) and text:
                    return text

        message = response.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content

        for key in ("text", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                nested = HttpAdapter._extract_response_text(value)
                if nested:
                    return nested
        return ""

    @property
    def capabilities(self) -> AdapterCapabilities:
        concurrency = int(self.config.args.get("concurrency", 1))
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=True,
            batching=True,
            concurrency=concurrency > 1,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text", "code"),
            supports_category_outputs=True,
            notes=("generic-http",),
        )

    @property
    def allow_partial_predictions(self) -> bool:
        """Allow dropped samples after failures instead of raising."""
        return _bool_arg(
            self.config.args.get("drop_failed_predictions", True),
            arg_name="drop_failed_predictions",
        )

    def _endpoint(self) -> str:
        url = self.config.args.get("url") or self.config.args.get("endpoint")
        if not url:
            raise ValueError("http adapter requires url or endpoint")
        return str(url)

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

    def _request_payload(self, sample: PredictSample) -> Mapping[str, Any]:
        context = sample_context(sample)
        payload_template = self.config.args.get("payload_template")
        if payload_template is not None:
            payload = render_value(payload_template, context)
            if isinstance(payload, Mapping):
                return payload
            return {"prompt": payload}

        model_name = self.config.model_name
        input_mode = str(self.config.args.get("input_mode", "messages"))
        if input_mode == "prompt":
            prompt_template = self.config.args.get("prompt_template")
            prompt = (
                render_value(prompt_template, context)
                if prompt_template
                else sample_to_text(sample)
            )
            payload: dict[str, Any] = {"prompt": prompt}
            if model_name:
                payload["model"] = model_name
            return payload

        payload: dict[str, Any] = {"messages": sample_messages(sample)}
        if model_name:
            payload["model"] = model_name
        prompt_template = self.config.args.get("prompt_template")
        if prompt_template:
            payload["messages"] = [
                {
                    "role": "user",
                    "content": str(render_value(prompt_template, context)),
                }
            ]
        return payload

    def _response_payload(self, response: Any) -> Any:
        response_path = self.config.args.get("response_path")
        if response_path:
            from guard_eval_harness.models.templates import extract_path

            return extract_path(response, str(response_path))
        return response

    def _predict_one(
        self,
        sample: PredictSample,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        timeout: float,
        retries: int,
        backoff: float,
        threshold: float,
    ) -> NormalizedPrediction:
        """Execute one HTTP request and normalize its score."""
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
                extracted = extract_path(
                    response_payload, score_path
                )
                if isinstance(extracted, str):
                    response_text = extracted
            except (KeyError, IndexError, ValueError):
                pass
        if not response_text:
            response_text = self._extract_response_text(
                response_payload
            )
        if not response_text:
            response_text = self._extract_response_text(
                response
            )
        if response_text:
            predicted_categories = extract_judge_categories(
                response_text
            )
        return NormalizedPrediction(
            sample_id=sample.id,
            unsafe_score=score,
            unsafe_label=score >= threshold,
            threshold=threshold,
            latency_ms=elapsed_ms,
            predicted_categories=predicted_categories,
            metadata={
                "adapter": self.adapter_name,
                "endpoint": endpoint,
                "input_mode": self.config.args.get("input_mode", "messages"),
            },
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
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    self._predict_one,
                    sample,
                    endpoint=endpoint,
                    headers=headers,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    threshold=threshold,
                ): idx
                for idx, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    index_map[idx] = future.result()
                except Exception as exc:
                    sample_id = samples[idx].id
                    _log.error(
                        "prediction failed for sample %s: %s",
                        sample_id,
                        exc,
                    )
                    failed.append((idx, sample_id, exc))

        if failed:
            ids = ", ".join(sample_id for _, sample_id, _ in failed)
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
