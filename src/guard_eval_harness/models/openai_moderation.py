"""OpenAI moderation adapter with image support."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.models.openai_compatible import _bool_arg
from guard_eval_harness.models.templates import (
    env_value,
    join_url,
    json_post_with_retry,
    render_value,
    sample_context,
    sample_openai_moderation_input,
)
from guard_eval_harness.registry import model_registry
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    NormalizedSample,
)

_MAX_CONCURRENCY = 750
_log = logging.getLogger(__name__)


@model_registry.register("openai_moderation")
class OpenAIModerationAdapter(ModelAdapter):
    """OpenAI moderation endpoint adapter."""

    adapter_name = "openai_moderation"

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
            supported_input_modalities=("text", "image"),
            supports_category_outputs=True,
            notes=("openai-moderation",),
        )

    @property
    def allow_partial_predictions(self) -> bool:
        """Allow dropped samples after failures instead of raising."""
        return _bool_arg(
            self.config.args.get("drop_failed_predictions", True),
            arg_name="drop_failed_predictions",
        )

    def _model_name(self) -> str:
        """Return the moderation model name."""
        return str(
            self.config.model_name
            or self.config.args.get("model")
            or "omni-moderation-latest"
        )

    def _endpoint(self) -> str:
        """Return the moderation endpoint URL."""
        url = self.config.args.get("url") or self.config.args.get("base_url")
        if url:
            return str(url)
        root = str(self.config.args.get("root_url", "https://api.openai.com"))
        return join_url(root, "/v1/moderations")

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth if configured."""
        headers = dict(self.config.args.get("headers", {}))
        token = env_value(
            self.config.args.get("api_key_env", "OPENAI_API_KEY"),
            self.config.args.get("api_key"),
        )
        if token:
            prefix = str(self.config.args.get("auth_prefix", "Bearer"))
            header_name = str(
                self.config.args.get("auth_header", "Authorization")
            )
            headers.setdefault(header_name, f"{prefix} {token}".strip())
        return headers

    def _request_payload(self, sample: NormalizedSample) -> dict[str, Any]:
        """Build one moderation request payload."""
        context = sample_context(sample)
        payload_template = self.config.args.get("payload_template")
        if payload_template is not None:
            payload = render_value(payload_template, context)
            if isinstance(payload, Mapping):
                return dict(payload)
            return {"input": payload}

        payload: dict[str, Any] = {
            "model": self._model_name(),
            "input": sample_openai_moderation_input(
                sample,
                include_role_prefix=self._include_role_prefix(),
            ),
        }
        extra_payload = self.config.args.get("extra_payload", {})
        if isinstance(extra_payload, Mapping):
            payload.update(render_value(extra_payload, context))
        return payload

    def _include_role_prefix(self) -> bool:
        """Return whether text inputs should be prefixed with message roles."""
        return _bool_arg(
            self.config.args.get("include_role_prefix", True),
            arg_name="include_role_prefix",
        )

    def _selected_categories(
        self,
        *,
        category_scores: Mapping[str, Any],
        categories: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return the configured or discovered provider categories."""
        configured = self.config.args.get("categories")
        if isinstance(configured, Sequence) and not isinstance(configured, str):
            return tuple(str(category) for category in configured)
        return tuple(
            dict.fromkeys([*category_scores.keys(), *categories.keys()])
        )

    def _category_threshold(self) -> float:
        """Return the threshold used for predicted category extraction."""
        return float(self.config.args.get("category_threshold", 0.5))

    def _score_aggregation(self) -> str:
        """Return the selected score aggregation strategy."""
        strategy = (
            str(self.config.args.get("binary_aggregation", "max_score"))
            .strip()
            .lower()
        )
        if strategy not in {
            "max_score",
            "mean_score",
            "any_category",
            "flagged",
        }:
            raise ValueError(
                "binary_aggregation must be one of: max_score, mean_score, "
                "any_category, flagged"
            )
        return strategy

    def _parse_response(
        self,
        response: Mapping[str, Any],
    ) -> tuple[float, tuple[str, ...], dict[str, float], dict[str, Any]]:
        """Normalize one moderation response into canonical fields."""
        results = response.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("moderation response did not include results[0]")
        first_result = results[0]
        if not isinstance(first_result, Mapping):
            raise ValueError("moderation response result was not an object")

        raw_scores = first_result.get("category_scores", {})
        raw_categories = first_result.get("categories", {})
        if not isinstance(raw_scores, Mapping):
            raw_scores = {}
        if not isinstance(raw_categories, Mapping):
            raw_categories = {}

        category_scores = {
            str(key): float(value)
            for key, value in raw_scores.items()
            if isinstance(value, (int, float))
        }
        category_flags = {
            str(key): bool(value)
            for key, value in raw_categories.items()
            if isinstance(value, bool)
        }
        selected_categories = self._selected_categories(
            category_scores=category_scores,
            categories=category_flags,
        )
        selected_scores = [
            category_scores.get(category, 0.0)
            for category in selected_categories
        ]
        flagged = bool(first_result.get("flagged", False))
        aggregation = self._score_aggregation()

        if aggregation == "mean_score" and selected_scores:
            unsafe_score = sum(selected_scores) / len(selected_scores)
        elif aggregation in {"any_category", "flagged"}:
            unsafe_score = (
                1.0
                if any(
                    category_flags.get(category, False)
                    for category in selected_categories
                )
                else (1.0 if flagged and aggregation == "flagged" else 0.0)
            )
        elif selected_scores:
            unsafe_score = max(selected_scores)
        else:
            unsafe_score = 1.0 if flagged else 0.0

        category_threshold = self._category_threshold()
        predicted_categories = tuple(
            category
            for category in selected_categories
            if category_flags.get(category, False)
            or category_scores.get(category, 0.0) >= category_threshold
        )
        metadata = {
            "request_id": response.get("id"),
            "provider_model": response.get("model"),
            "flagged": flagged,
            "provider_categories": category_flags,
        }
        return unsafe_score, predicted_categories, category_scores, metadata

    def _predict_one(
        self,
        sample: NormalizedSample,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
        retries: int,
        backoff: float,
        threshold: float,
    ) -> NormalizedPrediction:
        """Run one moderation request and normalize its response."""
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
        if not isinstance(response, Mapping):
            raise ValueError("moderation response must be a JSON object")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        unsafe_score, predicted_categories, category_scores, metadata = (
            self._parse_response(response)
        )
        return NormalizedPrediction(
            sample_id=sample.id,
            unsafe_score=unsafe_score,
            unsafe_label=unsafe_score >= threshold,
            threshold=threshold,
            latency_ms=elapsed_ms,
            predicted_categories=predicted_categories,
            category_scores=category_scores,
            metadata={
                "adapter": self.adapter_name,
                "endpoint": endpoint,
                "model_name": self._model_name(),
                **metadata,
            },
        )

    def predict_batch(
        self,
        samples: Sequence[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        """Execute one moderation request per sample."""
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
