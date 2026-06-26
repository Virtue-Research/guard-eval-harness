"""Anthropic prompt caching, thinking effort, and retry in the provider seam.

Covers ``providers._cache_marked``, the ``_call_anthropic`` payload knobs
(``GEH_VIBE_PROMPT_CACHE`` / ``GEH_VIBE_THINK_EFFORT``), and the bounded-retry
policy in ``providers._post_with_retry`` shared by both call seams. No network
and no real sleeps: httpx.post is monkeypatched to capture the payload or
script outcomes (same idiom as test_vibecoding_agents), and the injectable
``providers._sleep`` seam records the backoff pattern.
"""

from __future__ import annotations

import httpx
import pytest

import guard_eval_harness.vibecoding.agents.providers as providers_mod
from guard_eval_harness.vibecoding.agents._engine import MissingAPIKeyError


class _Resp:
    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict:
        return {"content": [{"type": "text", "text": "ok"}], "usage": {}}


def _capture_payload(monkeypatch) -> dict:
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json or {})
        return _Resp()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers_mod.httpx, "post", fake_post)
    return captured


# --- _cache_marked unit behavior ----------------------------------------


def test_cache_marked_tags_last_user_message_and_keeps_inputs():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "fix this file"},
    ]
    marked, sys_blocks = providers_mod._cache_marked(messages, "be safe")
    # last user message converted to blocks with the ephemeral breakpoint
    last = marked[-1]["content"]
    assert last == [
        {
            "type": "text",
            "text": "fix this file",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # earlier messages untouched; inputs not mutated
    assert marked[0]["content"] == "first"
    assert messages[-1]["content"] == "fix this file"
    assert sys_blocks == [{"type": "text", "text": "be safe"}]


def test_cache_marked_block_content_gets_breakpoint_on_last_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    marked, _ = providers_mod._cache_marked(messages, None)
    blocks = marked[0]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


# --- _call_anthropic payload knobs ---------------------------------------


def test_call_anthropic_caches_by_default(monkeypatch):
    payload = _capture_payload(monkeypatch)
    monkeypatch.delenv("GEH_VIBE_PROMPT_CACHE", raising=False)
    monkeypatch.delenv("GEH_VIBE_THINK_EFFORT", raising=False)
    providers_mod._call_anthropic(
        [{"role": "user", "content": "hello"}], "claude-fable-5", system="sys"
    )
    assert payload["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert payload["system"] == [{"type": "text", "text": "sys"}]
    # reasoning models reject temperature -- the seam must never send it
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "thinking" not in payload


def test_call_anthropic_cache_opt_out_restores_legacy_payload(monkeypatch):
    payload = _capture_payload(monkeypatch)
    monkeypatch.setenv("GEH_VIBE_PROMPT_CACHE", "0")
    monkeypatch.delenv("GEH_VIBE_THINK_EFFORT", raising=False)
    providers_mod._call_anthropic(
        [{"role": "user", "content": "hello"}], "claude-opus-4-8", system="sys"
    )
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["system"] == "sys"
    assert "cache_control" not in str(payload)


def test_call_anthropic_think_effort_adds_adaptive_thinking(monkeypatch):
    payload = _capture_payload(monkeypatch)
    monkeypatch.setenv("GEH_VIBE_THINK_EFFORT", "high")
    providers_mod._call_anthropic(
        [{"role": "user", "content": "hello"}], "claude-fable-5"
    )
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
    assert "temperature" not in payload


# --- _post_with_retry policy ----------------------------------------------


def _resp(status: int, *, headers: dict | None = None,
          body: dict | None = None) -> httpx.Response:
    """A real ``httpx.Response`` so ``raise_for_status`` raises the real
    ``HTTPStatusError`` (with ``.response``) the retry helper dispatches on."""
    return httpx.Response(
        status,
        headers=headers,
        json=body if body is not None else {"ok": True},
        request=httpx.Request("POST", "https://api.test/v1"),
    )


class _ScriptedPost:
    """httpx.post stand-in that replays a script of responses/exceptions."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_seams(monkeypatch, outcomes: list) -> tuple[_ScriptedPost, list]:
    """Patch httpx.post with a script and the sleep seam with a recorder."""
    post = _ScriptedPost(outcomes)
    sleeps: list[float] = []
    monkeypatch.setattr(providers_mod.httpx, "post", post)
    monkeypatch.setattr(providers_mod, "_sleep", sleeps.append)
    return post, sleeps


def _retry(**kwargs) -> dict:
    return providers_mod._post_with_retry(
        "https://api.test/v1", json={}, headers={}, timeout=1.0, **kwargs
    )


def test_retry_429_then_success(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [_resp(429), _resp(200, body={"ok": 1})])
    assert _retry() == {"ok": 1}
    assert post.calls == 2
    assert sleeps == [1.0]


def test_retry_429_honors_retry_after(monkeypatch):
    post, sleeps = _patch_seams(
        monkeypatch,
        [_resp(429, headers={"retry-after": "5"}), _resp(200)],
    )
    assert _retry() == {"ok": True}
    assert sleeps == [5.0]


def test_retry_429_caps_retry_after_at_60s(monkeypatch):
    post, sleeps = _patch_seams(
        monkeypatch,
        [_resp(429, headers={"retry-after": "600"}), _resp(200)],
    )
    assert _retry() == {"ok": True}
    assert sleeps == [60.0]


def test_retry_429_nonnumeric_retry_after_falls_back_to_backoff(monkeypatch):
    post, sleeps = _patch_seams(
        monkeypatch,
        [_resp(429, headers={"retry-after": "Mon, 09 Jun 2026 00:00:00 GMT"}),
         _resp(200)],
    )
    assert _retry() == {"ok": True}
    assert sleeps == [1.0]


def test_retry_500_then_success(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [_resp(500), _resp(200, body={"ok": 2})])
    assert _retry() == {"ok": 2}
    assert post.calls == 2
    assert sleeps == [1.0]


def test_retry_timeout_then_success(monkeypatch):
    post, sleeps = _patch_seams(
        monkeypatch,
        [httpx.TimeoutException("timed out"), _resp(200, body={"ok": 3})],
    )
    assert _retry() == {"ok": 3}
    assert post.calls == 2
    assert sleeps == [1.0]


def test_retry_5xx_exhaustion_reraises_with_backoff_pattern(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [_resp(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _retry()
    assert excinfo.value.response.status_code == 503
    assert post.calls == 3
    # deterministic exponential backoff, no jitter: 1s then 2s
    assert sleeps == [1.0, 2.0]


def test_retry_transport_exhaustion_reraises_last_error(monkeypatch):
    post, sleeps = _patch_seams(
        monkeypatch, [httpx.ConnectError("down")] * 3
    )
    with pytest.raises(httpx.ConnectError):
        _retry()
    assert post.calls == 3
    assert sleeps == [1.0, 2.0]


def test_retry_400_exactly_one_extra_attempt_then_raises(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [_resp(400)] * 3)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _retry()
    assert excinfo.value.response.status_code == 400
    # one extra attempt only: deterministic 400s are model/prompt-bound
    assert post.calls == 2
    assert sleeps == [1.0]


def test_retry_400_then_success(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [_resp(400), _resp(200, body={"ok": 4})])
    assert _retry() == {"ok": 4}
    assert post.calls == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_retry_other_4xx_never_retried(monkeypatch, status):
    post, sleeps = _patch_seams(monkeypatch, [_resp(status)] * 3)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _retry()
    assert excinfo.value.response.status_code == status
    assert post.calls == 1
    assert sleeps == []


# --- both seams route through the retry helper ---------------------------


def test_call_anthropic_retries_transient_429(monkeypatch):
    body = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    post, sleeps = _patch_seams(monkeypatch, [_resp(429), _resp(200, body=body)])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    out = providers_mod._call_anthropic(
        [{"role": "user", "content": "hello"}], "claude-fable-5"
    )
    assert out == body
    assert post.calls == 2
    assert sleeps == [1.0]


def test_call_openai_chat_retries_transient_500(monkeypatch):
    body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    post, sleeps = _patch_seams(monkeypatch, [_resp(500), _resp(200, body=body)])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = providers_mod._call_openai_chat(
        [{"role": "user", "content": "hello"}],
        "gpt-5.5",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )
    assert out == body
    assert post.calls == 2
    assert sleeps == [1.0]


def test_missing_anthropic_key_raises_before_any_attempt(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        providers_mod._call_anthropic(
            [{"role": "user", "content": "hello"}], "claude-fable-5"
        )
    assert post.calls == 0
    assert sleeps == []


def test_missing_openai_key_raises_before_any_attempt(monkeypatch):
    post, sleeps = _patch_seams(monkeypatch, [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        providers_mod._call_openai_chat(
            [{"role": "user", "content": "hello"}],
            "gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
    assert post.calls == 0
    assert sleeps == []


def test_http_timeout_honors_env_override(monkeypatch):
    """``providers._HTTP_TIMEOUT`` is env-overridable so ``geh vibe run`` (which
    uses these providers directly) honors a long ``GEH_VIBE_HTTP_TIMEOUT`` for
    slow reasoning-model generations instead of timing out at the 180s default.
    """
    import importlib

    monkeypatch.setenv("GEH_VIBE_HTTP_TIMEOUT", "1234")
    try:
        importlib.reload(providers_mod)
        assert providers_mod._HTTP_TIMEOUT == 1234.0
    finally:
        monkeypatch.delenv("GEH_VIBE_HTTP_TIMEOUT", raising=False)
        importlib.reload(providers_mod)
    # Unset -> back to the conservative 180s default (no behavior change).
    assert providers_mod._HTTP_TIMEOUT == 180.0
