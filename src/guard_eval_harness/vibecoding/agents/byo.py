"""Bring-your-own-predictions agent driver.

Wraps a preloaded collection of :class:`AgentArtifact` objects (keyed by
task id) and replays them through the standard :meth:`AgentDriver.generate`
seam. This lets the runner score externally generated predictions through the
exact same path as a live agent, with no model call and no network -- the core
of ``patch_eval`` / BYO mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from guard_eval_harness.vibecoding.agents.base import AgentDriver, AgentResult
from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import GenerationSpec
from guard_eval_harness.vibecoding.registry import agent_registry
from guard_eval_harness.vibecoding.schema import VibeTask


@agent_registry.register("byo_predictions")
class BYOArtifactDriver(AgentDriver):
    """Replay preloaded artifacts instead of generating live.

    Accepts either a mapping ``{task_id: AgentArtifact}`` or any iterable of
    artifacts (indexed internally by ``artifact.task_id``). ``generate`` looks
    up the artifact for the requested task and returns it verbatim; an unknown
    task id raises :class:`KeyError` so a misaligned predictions file fails
    loudly rather than silently scoring nothing.
    """

    name = "byo_predictions"

    def __init__(
        self,
        artifacts: Mapping[str, AgentArtifact]
        | Iterable[AgentArtifact]
        | None = None,
    ) -> None:
        self._by_task: dict[str, AgentArtifact] = {}
        if artifacts is None:
            return
        if isinstance(artifacts, Mapping):
            items: Iterable[AgentArtifact] = artifacts.values()
        else:
            items = artifacts
        for artifact in items:
            self._by_task[artifact.task_id] = artifact

    def add(self, artifact: AgentArtifact) -> None:
        """Register/replace the artifact served for ``artifact.task_id``."""
        self._by_task[artifact.task_id] = artifact

    def generate(
        self,
        task: VibeTask,
        *,
        workdir: str | Path | None = None,
        model: str | None = None,
        gen_spec: GenerationSpec | None = None,
    ) -> AgentResult:
        """Return the preloaded artifact for ``task.id``.

        ``workdir``, ``model``, and ``gen_spec`` are ignored: BYO mode replays
        externally generated artifacts and never frames a new generation.
        """
        try:
            artifact = self._by_task[task.id]
        except KeyError as exc:
            available = ", ".join(sorted(self._by_task)) or "<none>"
            raise KeyError(
                f"no preloaded artifact for task_id={task.id!r}; "
                f"loaded: {available}"
            ) from exc
        return AgentResult(
            artifact=artifact,
            model=artifact.model,
            metadata={"source": "byo_predictions"},
        )


__all__ = ["BYOArtifactDriver"]
