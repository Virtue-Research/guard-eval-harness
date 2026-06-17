"""Provider-agnostic live agent drivers.

One generic :class:`LLMAgentDriver` runs any :class:`ChatProvider` through the
shared :mod:`._engine`, so every model is evaluated through the exact same path
as Claude. Registered driver aliases:

- ``anthropic`` -> direct Anthropic (Claude); ``claude`` is the same provider
  (registered in :mod:`.claude` for back-compat).
- ``openai`` / ``gpt`` / ``codex`` -> direct OpenAI.
- ``deepseek`` / ``gemini`` / ``openrouter`` -> OpenRouter.
- ``llm`` -> routes by the ``--model`` name (claude-*/gpt-*/everything-else).

So ``geh vibe run --agent openai --model gpt-5.1`` (or ``--agent gemini``,
``--agent deepseek``, ``--agent llm --model deepseek/deepseek-chat``) works
identically to the Claude path.
"""

from __future__ import annotations

from pathlib import Path

from guard_eval_harness.vibecoding.agents._engine import generate_with
from guard_eval_harness.vibecoding.agents.base import AgentDriver, AgentResult
from guard_eval_harness.vibecoding.agents.providers import (
    ChatProvider,
    provider_for_alias,
    provider_for_model,
    resolve_model_for_provider,
)
from guard_eval_harness.vibecoding.interfaces import GenerationSpec
from guard_eval_harness.vibecoding.registry import agent_registry
from guard_eval_harness.vibecoding.schema import VibeTask

# Fixed-provider aliases. ``claude`` is intentionally omitted -- it is owned by
# ``agents/claude.py`` (same Anthropic provider) for backward compatibility.
_PROVIDER_ALIASES = (
    "anthropic",
    "openai",
    "gpt",
    "codex",
    "deepseek",
    "gemini",
    "openrouter",
)


class LLMAgentDriver(AgentDriver):
    """Generic driver over any :class:`ChatProvider`.

    The base driver is the ``llm`` router: it resolves the provider from the
    ``--model`` name at call time. Fixed-provider subclasses (one per alias)
    pin a provider regardless of the model. Both delegate to the shared engine,
    so a failed generation degrades to an empty artifact rather than aborting.
    """

    name = "llm"

    def _provider(self, model: str | None) -> ChatProvider:
        """Resolve the provider for ``model`` (router: by model name)."""
        return provider_for_model(model)

    def generate(
        self,
        task: VibeTask,
        *,
        workdir: str | Path | None = None,
        model: str | None = None,
        gen_spec: GenerationSpec | None = None,
    ) -> AgentResult:
        provider = self._provider(model)
        # Resolve + namespace the exact model id BEFORE generate_with, so a
        # config error (unknown bare OpenRouter id) raises loudly here rather
        # than being swallowed into a silent empty artifact by the engine.
        final_model = resolve_model_for_provider(provider, model)
        return generate_with(
            task,
            workdir=workdir,
            model=final_model,
            default_model=provider.default_model,
            complete=provider.complete,
            spec=gen_spec,
        )


def _register_alias(alias: str) -> type[LLMAgentDriver]:
    """Create + register a fixed-provider driver for ``alias``."""

    class _AliasDriver(LLMAgentDriver):
        name = alias

        def _provider(self, model: str | None) -> ChatProvider:
            return provider_for_alias(alias)

    _AliasDriver.__name__ = f"{alias.capitalize()}AgentDriver"
    _AliasDriver.__qualname__ = _AliasDriver.__name__
    return agent_registry.register(alias)(_AliasDriver)


# Register the model-routing ``llm`` driver and one driver per fixed alias.
agent_registry.register("llm")(LLMAgentDriver)
for _alias in _PROVIDER_ALIASES:
    _register_alias(_alias)


__all__ = ["LLMAgentDriver"]
