"""VibeCoding Safety Bench subsystem.

A Level-2 agentic benchmark family that evaluates coding agents on
repository-level secure-coding tasks by wrapping upstream secure-coding
benchmarks as out-of-process oracles.

This subsystem is separate from the classification flow: it has its own
runner, registry, schema, and CLI under
``guard_eval_harness.vibecoding``. The core public schema/result symbols
and registries are re-exported here for convenience.
"""

from __future__ import annotations

from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.registry import (
    agent_registry,
    ensure_vibe_registrations,
    oracle_registry,
    task_source_registry,
)
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    VibeTask,
)

__all__ = [
    "AgentArtifact",
    "EnvSpec",
    "OracleCapabilities",
    "VibeTask",
    "VibeTaskResult",
    "agent_registry",
    "ensure_vibe_registrations",
    "oracle_registry",
    "task_source_registry",
]
