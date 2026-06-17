"""Live Anthropic-API agent driver (Claude).

A thin wrapper over the shared generation engine (:mod:`._engine`) and the
Anthropic provider (:mod:`.providers`). Kept as its own ``claude`` alias and
re-exporting the legacy ``_call_anthropic`` / ``_resolve_model`` /
``MissingAPIKeyError`` seams for backward compatibility with the in-container
generators and existing tests. All other providers (OpenAI/Codex direct,
DeepSeek/Gemini via OpenRouter) are registered generically in :mod:`.llm`.
"""

from __future__ import annotations

from pathlib import Path

# Imported here purely so the legacy tests can monkeypatch the shared httpx
# seam via ``claude.httpx`` (the same module object the provider uses); the
# real network call lives in :mod:`.providers`.
import httpx  # noqa: F401  (re-exported seam for tests)

from guard_eval_harness.vibecoding.agents._engine import (
    MissingAPIKeyError,
    generate_with,
    resolve_model,
)
from guard_eval_harness.vibecoding.agents.base import AgentDriver, AgentResult
from guard_eval_harness.vibecoding.agents.providers import (
    AnthropicProvider,
    _call_anthropic,  # re-exported for legacy/back-compat callers + tests
)
from guard_eval_harness.vibecoding.interfaces import GenerationSpec
from guard_eval_harness.vibecoding.registry import agent_registry
from guard_eval_harness.vibecoding.schema import VibeTask

_DEFAULT_MODEL = "claude-opus-4-8"


def _resolve_model(model: str | None) -> str:
    """Resolve the Claude model: arg > ``GEH_VIBE_MODEL`` > default (legacy)."""
    return resolve_model(model, _DEFAULT_MODEL)


@agent_registry.register("claude")
class ClaudeAgentDriver(AgentDriver):
    """Real Anthropic-API driver producing patches or completions.

    The default model is taken from ``GEH_VIBE_MODEL`` (falling back to
    ``claude-opus-4-8``) and can be overridden per call. The network seam is
    :func:`providers._call_anthropic` (re-exported here), monkeypatched in
    tests. A missing ``ANTHROPIC_API_KEY`` surfaces once as
    :class:`MissingAPIKeyError` instead of degrading every task to an empty
    artifact.
    """

    name = "claude"

    def generate(
        self,
        task: VibeTask,
        *,
        workdir: str | Path | None = None,
        model: str | None = None,
        gen_spec: GenerationSpec | None = None,
    ) -> AgentResult:
        """Prompt Claude and convert the response into an artifact."""
        return generate_with(
            task,
            workdir=workdir,
            model=model,
            default_model=_DEFAULT_MODEL,
            complete=AnthropicProvider().complete,
            spec=gen_spec,
        )


__all__ = [
    "ClaudeAgentDriver",
    "MissingAPIKeyError",
    "_call_anthropic",
    "_resolve_model",
]
