"""Guard contract: model-family-specific prompt + parser."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from guard_eval_harness.output_formats import OutputFormat, ParsedLabel
from guard_eval_harness.policies import Policy
from guard_eval_harness.registry import Registry
from guard_eval_harness.schemas import Message, PredictSample


BackendKind = Literal["generate", "classify"]


class Guard(ABC):
    """Backend-agnostic guard implementation.

    Subclasses own the prompt construction and the output parsing.
    They must always accept a ``policy`` kwarg in ``build_messages``
    (fixed-taxonomy guards may ignore it; the runner always passes it).
    """

    name: str = ""
    backend_kind: BackendKind = "generate"
    accepts_policy: bool = False
    accepts_output_format: bool = False
    default_policy: Policy | None = None
    default_output_format: OutputFormat | None = None

    @abstractmethod
    def build_messages(
        self,
        sample: PredictSample,
        *,
        policy: Policy | None = None,
        output_format: OutputFormat | None = None,
    ) -> list[Message]:
        """Construct the wire-format messages for one sample."""

    @abstractmethod
    def parse(self, output: Any) -> ParsedLabel:
        """Parse the backend's raw output into a normalized label."""


guard_registry: Registry[type[Guard]] = Registry("guard")


def get_guard_cls(name: str) -> type[Guard]:
    """Look up a Guard class by name."""
    return guard_registry.get(name)


def list_guards() -> list[str]:
    """Return the names of all registered guards."""
    return guard_registry.keys()
