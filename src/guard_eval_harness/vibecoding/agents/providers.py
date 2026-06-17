"""Chat-completion providers for the VibeCoding agent layer.

Routing (per the user's standing rule):
- **Anthropic / Claude** -> direct Anthropic Messages API.
- **OpenAI / GPT / Codex** -> direct OpenAI Chat Completions API.
- **Everything else** (DeepSeek, Gemini, Qwen, ...) -> **OpenRouter**, which
  speaks the OpenAI Chat Completions shape, so it reuses
  :class:`OpenAIChatProvider` with a different base URL + key + namespaced
  model id (e.g. ``google/gemini-2.5-pro``, ``deepseek/deepseek-chat``).

Each provider exposes ``complete(messages, system, model) -> ChatResponse`` and
routes its one network call through a module-level seam (:func:`_call_anthropic`
/ :func:`_call_openai_chat`) so tests monkeypatch a single function. Both seams
post via :func:`_post_with_retry` (bounded retries for transient 429/5xx/
timeouts -- see its docstring for the full policy). Response text +
token-usage extraction is the only other provider-specific logic.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import httpx

from guard_eval_harness.vibecoding.agents._engine import (
    ChatResponse,
    MissingAPIKeyError,
)

# --- endpoints / knobs -------------------------------------------------

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_OPENAI_BASE = "https://api.openai.com/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Default output budget. Raised from 8192 (env-overridable): full-file C output
# for SecRepoBench/SecureVibeBench completions was truncating at 8k, corrupting
# the candidate on large source files. Opus 4.8 supports far larger outputs.
_DEFAULT_MAX_TOKENS = int(os.environ.get("GEH_VIBE_MAX_TOKENS", "32000"))
_HTTP_TIMEOUT = 180.0

# Default models (always overridable via ``--model`` / ``GEH_VIBE_MODEL``).
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
_DEFAULT_OPENAI_MODEL = "gpt-5.5"
_DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"

# OpenRouter recommends (optional) attribution headers. The referer must be
# the PUBLIC repo (mkdocs repo_url): every OpenRouter request transmits it, so
# a private/internal URL here would leak the org and point attribution at a
# 404.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Virtue-Research/guard-eval-harness",
    "X-Title": "guard-eval-harness",
}

# --- bounded retry policy (shared by both call seams) -------------------

# Transient failures (transport errors, 429, 5xx): up to 3 attempts total.
_RETRY_MAX_ATTEMPTS = 3
# HTTP 400: exactly ONE extra attempt. Policy: deterministic 400s are
# model/prompt-bound (oversized payload, rejected param), so a second
# identical failure raises instead of burning more attempts.
_RETRY_MAX_ATTEMPTS_400 = 2
# Deterministic exponential backoff, no jitter: 1s after attempt 1, 2s after
# attempt 2. Fixed so tests can assert the exact sleep pattern.
_RETRY_BACKOFF_BASE_S = 1.0
# A 429 Retry-After header is honored but capped so a hostile/buggy header
# cannot stall a worker for minutes.
_RETRY_AFTER_CAP_S = 60.0

# Injectable sleep seam: tests monkeypatch this to capture the backoff
# pattern without real waits.
_sleep = time.sleep


def _retry_after_delay(response: httpx.Response, fallback: float) -> float:
    """Delay for a 429: ``Retry-After`` seconds (capped), else ``fallback``.

    Only the integer/float-seconds form is parsed; the HTTP-date form (rare on
    rate limits) falls back to the computed exponential backoff.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return fallback
    try:
        delay = float(raw)
    except ValueError:
        return fallback
    return min(max(delay, 0.0), _RETRY_AFTER_CAP_S)


def _post_with_retry(
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """``httpx.post`` + ``raise_for_status`` + ``json()`` with bounded retries.

    Used by BOTH provider seams (:func:`_call_anthropic` /
    :func:`_call_openai_chat`). Rationale: :func:`_engine.generate_with`
    degrades any exception to an empty artifact, which scores as a permanent
    in-denominator model failure -- so a transient 429/5xx/timeout must be
    retried here, never silently converted into a lost task.

    Policy:

    - transport errors (timeouts included) and 429/5xx: up to
      :data:`_RETRY_MAX_ATTEMPTS` attempts total, deterministic exponential
      backoff (1s, 2s; no jitter), 429 honoring ``Retry-After`` capped at
      :data:`_RETRY_AFTER_CAP_S`;
    - 400: exactly one extra attempt (deterministic 400s are
      model/prompt-bound; see :data:`_RETRY_MAX_ATTEMPTS_400`);
    - any other 4xx (401/403/404/...): never retried;
    - a successful HTTP response with an undecodable body (``json()`` raises)
      is treated as non-transient and not retried -- the error propagates so
      the engine degrades it the same as any other failure;
    - on exhaustion the last error re-raises unchanged, so the engine's
      empty-artifact degradation stays the in-denominator policy of record.
    """
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
        backoff = _RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
        try:
            response = httpx.post(url, json=json, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except httpx.TransportError:
            # Includes httpx.TimeoutException (a TransportError subclass).
            if attempt >= _RETRY_MAX_ATTEMPTS:
                raise
            _sleep(backoff)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                if attempt >= _RETRY_MAX_ATTEMPTS:
                    raise
                delay = (
                    _retry_after_delay(exc.response, backoff)
                    if status == 429
                    else backoff
                )
                _sleep(delay)
            elif status == 400:
                if attempt >= _RETRY_MAX_ATTEMPTS_400:
                    raise
                _sleep(backoff)
            else:
                raise
    raise AssertionError("unreachable: retry loop must return or raise")


# --- Anthropic (direct) seam + extraction ------------------------------


def _cache_marked(
    messages: list[dict[str, Any]], system: str | list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Apply Anthropic prompt caching (docs: build-with-claude/prompt-caching).

    Adds ONE ``cache_control: {type: ephemeral}`` breakpoint on the last user
    message, which caches the entire prefix (system + messages up to it).
    Within a run the system prompt is constant and the big repo-snapshot user
    prompts are per-task, so the wins are (a) retries of the same request read
    the prefix from cache instead of re-billing it in full, and (b) any
    same-prefix re-ask (regen/empty-retry) is ~90% cheaper on input tokens.
    Sub-1024-token prefixes simply don't cache -- harmless. Returns a
    ``(messages_copy, system_blocks_or_none)`` pair without mutating inputs.
    """
    sys_blocks: list[dict[str, Any]] | None = None
    if system:
        sys_blocks = (
            [{"type": "text", "text": system}]
            if isinstance(system, str)
            else [dict(b) for b in system]
        )
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            content = [dict(b) for b in content]
            content[-1]["cache_control"] = {"type": "ephemeral"}
            m["content"] = content
        break
    return msgs, sys_blocks


def _call_anthropic(
    messages: list[dict[str, Any]],
    model: str,
    *,
    system: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """POST to the Anthropic Messages API; return parsed JSON (monkeypatched).

    Fails fast (before any request) when ``ANTHROPIC_API_KEY`` is unset or
    whitespace-only, so a misconfigured key surfaces as one clear
    :class:`MissingAPIKeyError` instead of a guaranteed-401 round-trip (and,
    via :func:`_engine.generate_with`, an empty artifact per task). The
    network call itself goes through :func:`_post_with_retry`, so transient
    429/5xx/timeouts are retried with bounded backoff instead of degrading
    into a permanent empty-artifact model failure.

    Cost/config knobs (both env-driven so every runner inherits them):

    - ``GEH_VIBE_PROMPT_CACHE`` (default ``1``): prompt caching via
      :func:`_cache_marked`. Set ``0`` to disable.
    - ``GEH_VIBE_THINK_EFFORT`` (e.g. ``high`` / ``max`` / ``xhigh``): adaptive
      extended thinking -- sends ``thinking: {type: adaptive}`` +
      ``output_config: {effort: ...}``. Thinking tokens are billed as output
      and share ``max_tokens``; high efforts at small budgets can starve the
      answer (empty text, ``stop_reason=max_tokens``).

    Never sends ``temperature``/``top_p``: 2026 reasoning models (e.g.
    claude-opus-4-8, claude-fable-5) reject ``temperature`` outright.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set; cannot run --agent claude"
        )
    headers = {
        "content-type": "application/json",
        "anthropic-version": _ANTHROPIC_VERSION,
        "x-api-key": api_key,
    }
    effort = os.environ.get("GEH_VIBE_THINK_EFFORT", "").strip().lower()
    cache_on = os.environ.get("GEH_VIBE_PROMPT_CACHE", "1").strip() != "0"
    system_blocks: str | list[dict[str, Any]] | None = system
    if cache_on:
        messages, system_blocks = _cache_marked(messages, system)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if effort:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
    if system_blocks:
        payload["system"] = system_blocks
    return _post_with_retry(
        _ANTHROPIC_URL, json=payload, headers=headers, timeout=_HTTP_TIMEOUT
    )


def _anthropic_text(response: Any) -> str:
    """Concatenate text blocks out of an Anthropic Messages response."""
    if not isinstance(response, Mapping):
        return ""
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text")
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts)


def _anthropic_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Map Anthropic ``usage`` to (prompt, completion, total) token counts."""
    if not isinstance(response, Mapping):
        return None, None, None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    prompt = prompt if isinstance(prompt, int) else None
    completion = completion if isinstance(completion, int) else None
    total = (
        prompt + completion
        if isinstance(prompt, int) and isinstance(completion, int)
        else None
    )
    return prompt, completion, total


# --- OpenAI-compatible seam (direct OpenAI AND OpenRouter) -------------


def _call_openai_chat(
    messages: list[dict[str, Any]],
    model: str,
    *,
    base_url: str,
    api_key_env: str,
    system: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    token_field: str = "max_tokens",
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST to a ``/chat/completions`` endpoint; return parsed JSON.

    Shared by the direct OpenAI provider and the OpenRouter provider (same
    request/response shape). The caller picks ``token_field``: direct OpenAI
    (gpt-5.x / o-series) requires ``max_completion_tokens`` and rejects
    ``max_tokens``; OpenRouter still accepts ``max_tokens``. This is the single
    monkeypatch seam for both.

    Fails fast (before any request) when the credential named by
    ``api_key_env`` is unset or whitespace-only, mirroring
    :func:`_call_anthropic`: a misconfigured key surfaces as one clear
    :class:`MissingAPIKeyError` rather than an empty ``Bearer`` header, a
    guaranteed 401, and -- via :func:`_engine.generate_with` -- a whole live
    batch silently degraded to per-task empty artifacts (scored as model
    failures). The network call itself goes through :func:`_post_with_retry`,
    so transient 429/5xx/timeouts are retried with bounded backoff.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            f"{api_key_env} is not set; cannot run this OpenAI-compatible "
            "--agent provider"
        )
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)
    chat_messages = list(messages)
    if system:
        chat_messages = [{"role": "system", "content": system}, *chat_messages]
    payload: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        token_field: max_tokens,
    }
    return _post_with_retry(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=_HTTP_TIMEOUT,
    )


def _openai_text(response: Any) -> str:
    """Extract assistant text from an OpenAI/OpenRouter chat response."""
    if not isinstance(response, Mapping):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    # Some OpenRouter reasoning routes put the answer only in ``reasoning`` with
    # an empty ``content``; fall back so it is not a silent empty artifact.
    reasoning = message.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def _openai_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Map OpenAI ``usage`` to (prompt, completion, total) token counts."""
    if not isinstance(response, Mapping):
        return None, None, None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None

    def _int(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    return _int("prompt_tokens"), _int("completion_tokens"), _int("total_tokens")


# --- provider classes --------------------------------------------------


class ChatProvider(ABC):
    """A model provider that turns a chat request into a :class:`ChatResponse`.

    Subclasses carry a ``name`` and ``default_model`` and implement the single
    network-backed :meth:`complete`. Drivers are otherwise provider-agnostic.
    """

    name: str
    default_model: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        model: str,
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Run one chat completion and normalize the result.

        ``max_tokens`` overrides the default output budget (the default is
        ``GEH_VIBE_MAX_TOKENS``, 32000; full-file C rewrites need a large cap
        to avoid truncation).
        """
        raise NotImplementedError


class AnthropicProvider(ChatProvider):
    """Direct Anthropic Messages API (Claude)."""

    name = "anthropic"
    default_model = _DEFAULT_ANTHROPIC_MODEL

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        model: str,
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        response = _call_anthropic(
            messages, model, system=system,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
        )
        prompt, completion, total = _anthropic_usage(response)
        return ChatResponse(
            text=_anthropic_text(response),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            raw=response if isinstance(response, dict) else {},
        )


class OpenAIChatProvider(ChatProvider):
    """OpenAI Chat Completions shape — used for direct OpenAI AND OpenRouter."""

    def __init__(
        self,
        *,
        name: str,
        default_model: str,
        base_url: str,
        api_key_env: str,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.default_model = default_model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.extra_headers = dict(extra_headers) if extra_headers else None

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        model: str,
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        # Direct OpenAI (gpt-5.x / o-series) requires ``max_completion_tokens``;
        # OpenRouter still wants ``max_tokens`` (some third-party models only
        # accept it). The provider knows the endpoint, so it picks the field.
        token_field = (
            "max_completion_tokens"
            if self.base_url.rstrip("/") == _OPENAI_BASE
            else "max_tokens"
        )
        # OpenRouter requires vendor-namespaced ids; normalize bare third-party
        # names (``gemini-3.1-pro`` -> ``google/gemini-3.1-pro``) here so every
        # caller is covered, not just those that pre-resolve via
        # :func:`resolve_model_for_provider`. Raises on an unknown bare name
        # instead of letting OpenRouter 404 into a silent empty artifact.
        if self.base_url.rstrip("/") == _OPENROUTER_BASE:
            model = _namespace_openrouter(model)
        response = _call_openai_chat(
            messages,
            model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            system=system,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            token_field=token_field,
            extra_headers=self.extra_headers,
        )
        prompt, completion, total = _openai_usage(response)
        return ChatResponse(
            text=_openai_text(response),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            raw=response if isinstance(response, dict) else {},
        )


# --- factories + routing ----------------------------------------------


def anthropic_provider() -> AnthropicProvider:
    """Direct Anthropic provider (Claude)."""
    return AnthropicProvider()


def openai_provider(default_model: str = _DEFAULT_OPENAI_MODEL) -> OpenAIChatProvider:
    """Direct OpenAI provider (GPT / Codex)."""
    return OpenAIChatProvider(
        name="openai",
        default_model=default_model,
        base_url=_OPENAI_BASE,
        api_key_env="OPENAI_API_KEY",
    )


def openrouter_provider(
    default_model: str = _DEFAULT_OPENROUTER_MODEL,
) -> OpenAIChatProvider:
    """OpenRouter provider (DeepSeek, Gemini, and all other models)."""
    return OpenAIChatProvider(
        name="openrouter",
        default_model=default_model,
        base_url=_OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
        extra_headers=_OPENROUTER_HEADERS,
    )


# Provider aliases exposed as agent driver names. Each value builds a provider.
_ALIAS_FACTORIES: dict[str, Any] = {
    "anthropic": anthropic_provider,
    "claude": anthropic_provider,
    "openai": openai_provider,
    "gpt": openai_provider,
    "codex": lambda: openai_provider("gpt-5.1-codex"),
    "deepseek": lambda: openrouter_provider("deepseek/deepseek-v4-flash"),
    "gemini": lambda: openrouter_provider("google/gemini-2.5-pro"),
    "openrouter": openrouter_provider,
}


def provider_for_alias(alias: str) -> ChatProvider:
    """Build the provider for a registered driver alias (e.g. ``deepseek``)."""
    factory = _ALIAS_FACTORIES.get(alias)
    if factory is None:
        raise KeyError(
            f"unknown provider alias {alias!r}; "
            f"known: {sorted(_ALIAS_FACTORIES)}"
        )
    return factory()


# Bare vendor prefixes -> their OpenRouter namespace, so the ``llm`` router can
# accept ``gemini-2.5-pro`` and route it as ``google/gemini-2.5-pro``.
_OPENROUTER_VENDORS = {
    "gemini": "google",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "llama": "meta-llama",
    "mistral": "mistralai",
    "mixtral": "mistralai",
    "grok": "x-ai",
}


def _namespace_openrouter(model: str) -> str:
    """Ensure an OpenRouter model id is vendor-namespaced, else raise loudly.

    OpenRouter requires ``vendor/model`` ids; a bare third-party name is
    normalized via :data:`_OPENROUTER_VENDORS`. An unknown bare name raises
    rather than letting OpenRouter 404 into a silent empty artifact.
    """
    if "/" in model:
        return model
    vendor = _OPENROUTER_VENDORS.get(model.split("-", 1)[0].lower())
    if vendor is None:
        raise ValueError(
            "OpenRouter model ids must be vendor-namespaced (e.g. "
            f"google/gemini-2.5-pro); got {model!r}. Pass a namespaced "
            "--model or use a provider alias (deepseek/gemini/openrouter)."
        )
    return f"{vendor}/{model}"


def _is_openai_model(name: str) -> bool:
    """True for direct-OpenAI ids: gpt-*, chatgpt*, codex*, the o<N> family."""
    return name.startswith(
        ("gpt-", "gpt5", "gpt4", "chatgpt", "codex")
    ) or bool(re.match(r"o\d+(-|$)", name))


def provider_for_model(model: str | None) -> ChatProvider:
    """Route a model name to its provider (the ``llm`` driver, by model name).

    ``claude-*`` -> Anthropic; ``gpt-*`` / ``o<N>`` / ``codex*`` / ``chatgpt*``
    -> direct OpenAI; everything else -> OpenRouter. The exact model id (with
    explicit > env > default precedence and OpenRouter namespacing) is computed
    by :func:`resolve_model_for_provider`.
    """
    name = (model or os.environ.get("GEH_VIBE_MODEL") or "").strip().lower()
    if name.startswith("claude"):
        return anthropic_provider()
    if _is_openai_model(name):
        return openai_provider(model or _DEFAULT_OPENAI_MODEL)
    return openrouter_provider(model or _DEFAULT_OPENROUTER_MODEL)


def resolve_model_for_provider(
    provider: ChatProvider, model: str | None
) -> str:
    """Resolve the exact model id to send: explicit > ``GEH_VIBE_MODEL`` >
    provider default. OpenRouter-backed providers get a vendor-namespaced id
    (raising on an unknown bare name) so a request can't silently 404."""
    chosen = model or os.environ.get("GEH_VIBE_MODEL") or provider.default_model
    if (
        isinstance(provider, OpenAIChatProvider)
        and provider.base_url.rstrip("/") == _OPENROUTER_BASE
    ):
        chosen = _namespace_openrouter(chosen)
    return chosen


__all__ = [
    "ChatProvider",
    "AnthropicProvider",
    "OpenAIChatProvider",
    "anthropic_provider",
    "openai_provider",
    "openrouter_provider",
    "provider_for_alias",
    "provider_for_model",
    "resolve_model_for_provider",
    "_call_anthropic",
    "_call_openai_chat",
]
