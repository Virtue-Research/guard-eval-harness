"""Tests for agent drivers: BYO replay + the provider-agnostic LLM drivers.

All network is mocked at the single per-provider seam in ``agents/providers``
(``_call_anthropic`` for the direct Anthropic path, ``_call_openai_chat`` for
the OpenAI-shape path shared by direct OpenAI and OpenRouter). Tests assert
both the produced artifact AND the routing (which base URL + key env each
alias/model resolves to).
"""

from __future__ import annotations

import pytest

from guard_eval_harness.vibecoding.agents import claude as claude_mod
from guard_eval_harness.vibecoding.agents import providers as providers_mod
from guard_eval_harness.vibecoding.agents._engine import parse_fenced_block
from guard_eval_harness.vibecoding.agents.base import get_agent_driver
from guard_eval_harness.vibecoding.agents.byo import BYOArtifactDriver
from guard_eval_harness.vibecoding.agents.claude import (
    ClaudeAgentDriver,
    MissingAPIKeyError,
)
from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.schema import TaskEnvironmentRef, VibeTask


def _task(task_id: str = "t/1", task_type: str = "repo_patch") -> VibeTask:
    return VibeTask(
        id=task_id,
        source_dataset="t",
        task_type=task_type,
        instructions="fix the vulnerability",
        environment=TaskEnvironmentRef(oracle="t"),
    )


def _anthropic_resp(text: str) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }


def _openai_resp(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }


_DIFF = "```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n```"


def _capture_openai(monkeypatch, text: str = _DIFF) -> dict:
    """Patch the OpenAI-shape seam; return a dict that records call kwargs."""
    seen: dict = {}

    def fake(messages, model, *, base_url, api_key_env, system=None,
             max_tokens=32000, token_field="max_tokens", extra_headers=None):
        seen.update(
            model=model, base_url=base_url, api_key_env=api_key_env,
            system=system, max_tokens=max_tokens, token_field=token_field,
            extra_headers=extra_headers,
        )
        return _openai_resp(text)

    monkeypatch.setattr(providers_mod, "_call_openai_chat", fake)
    return seen


# --- BYO + Claude (back-compat) ----------------------------------------


def test_byo_returns_preloaded_artifact():
    art = AgentArtifact(task_id="t/1", model="m", kind="patch", patch="diff")
    driver = BYOArtifactDriver([art])
    result = driver.generate(_task("t/1"))
    assert result.artifact is art


def test_claude_repo_patch_produces_patch(monkeypatch):
    monkeypatch.setattr(
        providers_mod, "_call_anthropic",
        lambda messages, model, system=None, **_kw: _anthropic_resp(_DIFF),
    )
    result = ClaudeAgentDriver().generate(
        _task("t/1", "repo_patch"), model="claude-opus-4-8"
    )
    assert result.artifact.kind == "patch"
    assert result.artifact.patch.startswith("--- a/x.py")
    assert "+good" in result.artifact.patch
    assert result.model == "claude-opus-4-8"
    assert result.total_tokens == 12


def test_claude_repo_completion_produces_completion(monkeypatch):
    monkeypatch.setattr(
        providers_mod, "_call_anthropic",
        lambda messages, model, system=None, **_kw: _anthropic_resp(
            "```\nint x = 0;\n```"
        ),
    )
    result = ClaudeAgentDriver().generate(_task("t/2", "repo_completion"))
    assert result.artifact.kind == "completion"
    assert "int x = 0;" in result.artifact.completion


def test_claude_garbled_response_is_empty(monkeypatch):
    monkeypatch.setattr(
        providers_mod, "_call_anthropic",
        lambda messages, model, system=None, **_kw: _anthropic_resp("no fence here"),
    )
    result = ClaudeAgentDriver().generate(_task("t/3", "repo_patch"))
    assert result.artifact.metadata.get("empty") is True
    assert result.artifact.patch.strip() == ""


def _no_http(*_args, **_kwargs):
    raise AssertionError("no HTTP request must be made when the key is unset")


def test_call_anthropic_fails_fast_without_key(monkeypatch):
    """The network seam refuses to send an empty ``x-api-key`` -- it raises
    before any request rather than burning a guaranteed-401 round-trip."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Hard-assert the fail-fast happens *before* the network: any httpx.post
    # call would raise AssertionError instead of MissingAPIKeyError.
    monkeypatch.setattr(claude_mod.httpx, "post", _no_http)
    with pytest.raises(MissingAPIKeyError):
        claude_mod._call_anthropic(
            [{"role": "user", "content": "x"}], "claude-opus-4-8"
        )


def test_call_anthropic_treats_blank_key_as_missing(monkeypatch):
    """A whitespace-only key is not a usable credential -> treated as missing,
    still before any request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setattr(claude_mod.httpx, "post", _no_http)
    with pytest.raises(MissingAPIKeyError):
        claude_mod._call_anthropic(
            [{"role": "user", "content": "x"}], "claude-opus-4-8"
        )


def test_call_openai_chat_fails_fast_without_key(monkeypatch):
    """The OpenAI-shape seam refuses to send an empty ``Bearer`` -- it raises
    before any request, parity with the Anthropic path. Otherwise the
    guaranteed 401 is caught by generate_with and a whole misconfigured live
    batch is silently scored as model failures."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(providers_mod.httpx, "post", _no_http)
    with pytest.raises(MissingAPIKeyError):
        providers_mod._call_openai_chat(
            [{"role": "user", "content": "x"}],
            "gpt-5.1",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )


def test_call_openai_chat_treats_blank_key_as_missing(monkeypatch):
    """A whitespace-only key is not a usable credential -> treated as missing,
    still before any request. Exercised via ``api_key_env`` so it also covers
    the OpenRouter path (``OPENROUTER_API_KEY``)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    monkeypatch.setattr(providers_mod.httpx, "post", _no_http)
    with pytest.raises(MissingAPIKeyError):
        providers_mod._call_openai_chat(
            [{"role": "user", "content": "x"}],
            "deepseek/deepseek-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )


def test_generate_surfaces_missing_key_not_empty_artifact(monkeypatch):
    """A misconfigured key must surface once, not degrade every task to an
    empty artifact with the error buried in metadata."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        ClaudeAgentDriver().generate(_task("t/4", "repo_patch"))


def test_missing_key_error_is_cli_user_facing():
    """The error must land in the CLI's user-facing set so `geh vibe run`
    prints one clean `Error: ...` line, not a Python traceback."""
    from guard_eval_harness.cli.main import _USER_FACING_EXCEPTIONS

    assert issubclass(MissingAPIKeyError, _USER_FACING_EXCEPTIONS)


def test_generate_still_degrades_on_network_error(monkeypatch):
    """A genuine network/5xx failure keeps the per-task empty-artifact
    contract (only the missing-key case is escalated)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def _boom(messages, model, *, system=None, **_kwargs):
        raise RuntimeError("simulated 503 from the API")

    # The driver routes through AnthropicProvider().complete -> the provider
    # module's _call_anthropic, so patch the real seam there (not the
    # re-exported claude.* name).
    monkeypatch.setattr(providers_mod, "_call_anthropic", _boom)
    result = ClaudeAgentDriver().generate(_task("t/5", "repo_patch"))
    assert result.artifact.metadata.get("empty") is True
    assert result.artifact.metadata.get("error") == "simulated 503 from the API"


# --- provider routing (the new model-agnostic layer) -------------------


def test_openai_alias_routes_direct_openai(monkeypatch):
    seen = _capture_openai(monkeypatch)
    result = get_agent_driver("openai").generate(
        _task("t/1", "repo_patch"), model="gpt-5.1"
    )
    assert result.artifact.kind == "patch"
    assert "+good" in result.artifact.patch
    assert result.total_tokens == 12
    # Direct OpenAI: api.openai.com + OPENAI_API_KEY + max_completion_tokens
    # (gpt-5.x/o-series reject max_tokens).
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["api_key_env"] == "OPENAI_API_KEY"
    assert seen["model"] == "gpt-5.1"
    assert seen["token_field"] == "max_completion_tokens"
    assert seen["max_tokens"] == 32000


def test_deepseek_alias_routes_openrouter(monkeypatch):
    seen = _capture_openai(monkeypatch)
    result = get_agent_driver("deepseek").generate(_task("t/1", "repo_patch"))
    assert result.artifact.kind == "patch"
    # Third-party model -> OpenRouter, namespaced default model, max_tokens
    # (OpenRouter keeps max_tokens, unlike direct OpenAI).
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key_env"] == "OPENROUTER_API_KEY"
    assert seen["model"] == "deepseek/deepseek-v4-flash"
    assert seen["token_field"] == "max_tokens"
    assert seen["extra_headers"]  # OpenRouter attribution headers present


def test_gemini_alias_routes_openrouter(monkeypatch):
    seen = _capture_openai(monkeypatch)
    get_agent_driver("gemini").generate(_task("t/1", "repo_patch"))
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["model"] == "google/gemini-2.5-pro"


def test_codex_alias_routes_direct_openai(monkeypatch):
    seen = _capture_openai(monkeypatch)
    get_agent_driver("codex").generate(_task("t/1", "repo_patch"))
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["api_key_env"] == "OPENAI_API_KEY"


def test_llm_router_picks_provider_by_model(monkeypatch):
    # gpt-* -> direct OpenAI
    seen = _capture_openai(monkeypatch)
    get_agent_driver("llm").generate(_task("t/1"), model="gpt-4o")
    assert seen["base_url"] == "https://api.openai.com/v1"

    # a namespaced third-party model -> OpenRouter (verbatim model id)
    seen2 = _capture_openai(monkeypatch)
    get_agent_driver("llm").generate(
        _task("t/1"), model="deepseek/deepseek-chat"
    )
    assert seen2["base_url"] == "https://openrouter.ai/api/v1"
    assert seen2["model"] == "deepseek/deepseek-chat"

    # claude-* -> direct Anthropic
    monkeypatch.setattr(
        providers_mod, "_call_anthropic",
        lambda messages, model, system=None, **_kw: _anthropic_resp(_DIFF),
    )
    res = get_agent_driver("llm").generate(
        _task("t/1"), model="claude-opus-4-8"
    )
    assert res.artifact.kind == "patch"
    assert "+good" in res.artifact.patch


def test_llm_router_namespaces_bare_third_party_model(monkeypatch):
    # A bare third-party name -> OpenRouter with the vendor namespace added
    # (OpenRouter would 404 on an un-namespaced id).
    seen = _capture_openai(monkeypatch)
    get_agent_driver("llm").generate(_task("t/1"), model="gemini-2.5-pro")
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["model"] == "google/gemini-2.5-pro"

    seen2 = _capture_openai(monkeypatch)
    get_agent_driver("llm").generate(_task("t/1"), model="qwen-3-235b")
    # Bare qwen -> OpenRouter (never the direct OpenAI key/endpoint).
    assert seen2["base_url"] == "https://openrouter.ai/api/v1"
    assert seen2["api_key_env"] == "OPENROUTER_API_KEY"
    assert seen2["model"] == "qwen/qwen-3-235b"


def test_llm_router_rejects_unknown_bare_model(monkeypatch):
    # An un-namespaced, unknown-vendor model fails fast (loud) instead of
    # silently 404-ing into an empty artifact.
    _capture_openai(monkeypatch)
    with pytest.raises(ValueError):
        get_agent_driver("llm").generate(_task("t/1"), model="frobnicate-9000")


def test_namespaced_model_passthrough_via_llm(monkeypatch):
    # An already-namespaced id is forwarded verbatim.
    seen = _capture_openai(monkeypatch)
    get_agent_driver("llm").generate(
        _task("t/1"), model="mistralai/mistral-large"
    )
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["model"] == "mistralai/mistral-large"


def test_openrouter_complete_namespaces_bare_model(monkeypatch):
    # Callers that route via provider_for_model(...).complete(...) without
    # pre-resolving the id (e.g. the host generation scripts) must still get
    # the vendor namespace -- complete() itself normalizes for OpenRouter.
    seen = _capture_openai(monkeypatch)
    providers_mod.openrouter_provider().complete(
        [{"role": "user", "content": "hi"}], None, "gemini-2.5-pro"
    )
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["model"] == "google/gemini-2.5-pro"

    # Unknown bare vendor still fails fast at the same seam.
    with pytest.raises(ValueError):
        providers_mod.openrouter_provider().complete(
            [{"role": "user", "content": "hi"}], None, "frobnicate-9000"
        )

    # Direct OpenAI is untouched: the id passes through verbatim.
    seen2 = _capture_openai(monkeypatch)
    providers_mod.openai_provider().complete(
        [{"role": "user", "content": "hi"}], None, "gpt-5.5"
    )
    assert seen2["base_url"] != "https://openrouter.ai/api/v1"
    assert seen2["model"] == "gpt-5.5"


def test_openai_completion_task(monkeypatch):
    _capture_openai(monkeypatch, text="```\nint x = 0;\n```")
    result = get_agent_driver("openai").generate(
        _task("t/2", "repo_completion"), model="gpt-5.1"
    )
    assert result.artifact.kind == "completion"
    assert "int x = 0;" in result.artifact.completion


def test_openai_garbled_response_is_empty(monkeypatch):
    _capture_openai(monkeypatch, text="prose, no fenced block")
    result = get_agent_driver("openrouter").generate(_task("t/3"))
    assert result.artifact.metadata.get("empty") is True
    assert result.artifact.patch.strip() == ""


def test_provider_network_error_degrades_to_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(providers_mod, "_call_openai_chat", boom)
    result = get_agent_driver("openai").generate(_task("t/4"))
    assert result.artifact.metadata.get("empty") is True
    assert result.artifact.metadata.get("error")


# --- fenced-body extraction + wrapper-tag sanitation --------------------
#
# A generation wrapper once staged literal <CODE>/</CODE> delimiter tags
# (upstream BaxBench's delivery format) into scored files -> guaranteed
# build failures, i.e. silent score deflation. The extractor must strip
# wrapper-tag lines at the body edges and recover bare <CODE> blocks when
# no fence is present, without touching clean inputs or interior content.


def test_parse_fenced_block_clean_fence_unchanged():
    """Regression: clean fenced bodies pass through byte-identical."""
    body = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good"
    assert parse_fenced_block(f"```diff\n{body}\n```") == body
    assert parse_fenced_block(f"prose before\n```python\n{body}\n```") == body
    assert parse_fenced_block(f"```\n{body}\n```") == body


def test_parse_fenced_block_strips_code_wrapper_lines():
    """Edge-line <CODE>/</CODE> tags inside the fence are wrapper leakage,
    never code -- staging them verbatim guarantees a build failure."""
    text = "```python\n<CODE>\ndef f():\n    return 1\n</CODE>\n```"
    assert parse_fenced_block(text) == "def f():\n    return 1"


def test_parse_fenced_block_strips_leading_file_header():
    """A '### FILE: ...' first line is a staged filename label (upstream
    treats '### ...' headings as filepaths), not code."""
    text = "```python\n### FILE: app/main.py\n<CODE>\nx = 1\n</CODE>\n```"
    assert parse_fenced_block(text) == "x = 1"


def test_parse_fenced_block_recovers_bare_code_tag_block():
    """No fence but an upstream-format <CODE>...</CODE> block: recover the
    body instead of discarding real code as an empty artifact (deflation)."""
    text = "Here is the file.\n<CODE>\ndef f():\n    return 1\n</CODE>\nDone."
    assert parse_fenced_block(text) == "def f():\n    return 1"


def test_parse_fenced_block_prose_only_still_empty():
    assert parse_fenced_block("no code in this response") == ""
    assert parse_fenced_block("") == ""


def test_parse_fenced_block_keeps_interior_code_tag_lines():
    """A literal <CODE> string inside the body (not at the edges) is content
    and must survive untouched -- both as a bare mid-body line and as part
    of a diff hunk line."""
    body = 'print("a")\n<CODE>\nprint("b")'
    assert parse_fenced_block(f"```python\n{body}\n```") == body

    diff = (
        "--- a/t.txt\n+++ b/t.txt\n@@ -1,2 +1,2 @@\n line\n-<CODE>old\n"
        "+<CODE>new"
    )
    assert parse_fenced_block(f"```diff\n{diff}\n```") == diff


def test_bare_code_tag_response_yields_real_artifact(monkeypatch):
    """End to end: a no-fence <CODE>-wrapped response must produce a real
    scored artifact, not an in-denominator empty (deflation class)."""
    _capture_openai(
        monkeypatch,
        text=(
            "<CODE>\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n"
            "</CODE>"
        ),
    )
    result = get_agent_driver("openai").generate(_task("t/5", "repo_patch"))
    assert result.artifact.metadata.get("empty") is None
    assert result.artifact.patch == (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good"
    )
