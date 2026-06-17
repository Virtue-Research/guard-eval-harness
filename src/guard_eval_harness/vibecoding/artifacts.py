"""The artifact crossing the generation-to-evaluation boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from guard_eval_harness.execution.artifacts import sha256_payload
from guard_eval_harness.vibecoding.schema import VibeModel, VibeTask

ArtifactKind = Literal[
    "patch",
    "full_file",
    "completion",
    "repo_dir",
    "staged_chain",
]

# Reserved ``metadata`` sub-key holding per-generation telemetry (token
# counts, cost, the resolved model) that the live runner folds onto an
# artifact for provenance. It is excluded from the *scoring* identity
# (:func:`artifact_scoring_sha256`) so volatile usage never perturbs the
# oracle cache key.
TELEMETRY_METADATA_KEY = "agent_telemetry"


class AgentArtifact(VibeModel):
    """A candidate produced by an agent (or loaded from predictions).

    Exactly one payload field is meaningful per ``kind``; the model
    validator enforces that the matching field is populated so downstream
    adapters never have to guess.
    """

    task_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    kind: ArtifactKind
    patch: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    completion: str | None = None
    worktree: str | None = None
    base_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_payload_for_kind(self) -> "AgentArtifact":
        if self.kind == "patch":
            if not self.patch:
                raise ValueError("kind='patch' requires a non-empty patch")
        elif self.kind == "full_file":
            if not self.files:
                raise ValueError("kind='full_file' requires non-empty files")
        elif self.kind == "completion":
            if not self.completion:
                raise ValueError(
                    "kind='completion' requires a non-empty completion"
                )
        elif self.kind == "repo_dir":
            if not self.worktree:
                raise ValueError(
                    "kind='repo_dir' requires a non-empty worktree path"
                )
        elif self.kind == "staged_chain":
            if not self.metadata:
                raise ValueError(
                    "kind='staged_chain' requires non-empty metadata"
                )
        return self


def artifact_sha256(artifact: AgentArtifact) -> str:
    """Fingerprint an artifact via canonical sorted JSON.

    Key ordering is normalized by ``sha256_payload`` (sort_keys=True), so the
    digest is stable across dict reorderings and changes only when payload
    content changes.
    """
    return sha256_payload(artifact.model_dump(mode="json"))


def artifact_scoring_sha256(artifact: AgentArtifact) -> str:
    """Fingerprint the *scoreable* artifact, excluding volatile telemetry.

    Identical to :func:`artifact_sha256` except the reserved
    ``metadata[TELEMETRY_METADATA_KEY]`` block (per-generation token counts,
    cost, resolved model -- folded on by the live runner for provenance) is
    dropped, so it never perturbs the oracle scoring cache key: two runs that
    produce the same candidate score-identically regardless of usage/cost.

    Artifacts without that reserved key (every BYO / ``patch_eval``
    prediction) hash identically to :func:`artifact_sha256`, so cache keys
    for the offline path are unchanged.
    """
    payload = artifact.model_dump(mode="json")
    meta = payload.get("metadata")
    if isinstance(meta, dict) and TELEMETRY_METADATA_KEY in meta:
        payload["metadata"] = {
            k: v for k, v in meta.items() if k != TELEMETRY_METADATA_KEY
        }
    return sha256_payload(payload)


def task_sha256(task: VibeTask) -> str:
    """Fingerprint a task's metadata via canonical sorted JSON."""
    return sha256_payload(task.model_dump(mode="json"))
