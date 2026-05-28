"""Backend contract: how a Guard talks to an inference engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from guard_eval_harness.registry import Registry
from guard_eval_harness.schemas import Message


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Resolved backend configuration."""

    kind: str
    model: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Base inference backend."""

    kind: str = ""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: BackendConfig) -> "Backend":
        """Build a backend from a resolved config."""
        return cls(config)


class GenerationBackend(Backend):
    """Backend that produces raw text outputs from chat messages.

    Generation defaults (``max_new_tokens``, ``temperature``) are read
    from ``BackendConfig.args`` so a profile can tune them per-model.
    Subclasses must honor ``self.max_new_tokens`` / ``self.temperature``
    inside their ``generate()`` implementation.
    """

    DEFAULT_MAX_NEW_TOKENS: int = 128
    DEFAULT_TEMPERATURE: float = 0.0

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.max_new_tokens: int = int(
            config.args.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        )
        self.temperature: float = float(
            config.args.get("temperature", self.DEFAULT_TEMPERATURE)
        )

    @abstractmethod
    def generate(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[str]:
        """Generate raw text for each conversation in the batch."""


class ClassifierBackend(Backend):
    """Backend that produces per-label probabilities."""

    @abstractmethod
    def classify(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[dict[str, float]]:
        """Score each conversation; keys are label names."""


backend_registry: Registry[type[Backend]] = Registry("backend")


def get_backend_cls(kind: str) -> type[Backend]:
    """Look up a backend class by `kind`."""
    return backend_registry.get(kind)


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return backend_registry.keys()
