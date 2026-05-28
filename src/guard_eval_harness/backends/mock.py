"""Mock backend for tests and the bundled demo."""

from typing import Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    ClassifierBackend,
    GenerationBackend,
    backend_registry,
)
from guard_eval_harness.schemas import Message, TextPart


@backend_registry.register("mock")
class MockGenerationBackend(GenerationBackend):
    """Deterministic backend that fakes guard outputs from message text.

    A conversation is classified UNSAFE if any user message contains any
    of the configured trigger words; otherwise SAFE. The default output
    format is ``safe_unsafe_first_line`` (one of ``safe`` / ``unsafe``).

    Configurable via ``args``:
      - ``triggers``: list[str], default ``["unsafe", "harm", "attack"]``
      - ``response_format``: one of ``"safe_unsafe"`` (default),
        ``"yes_no"``, ``"json_rating"``
      - ``fail_substring``: optional str; if present in a message, the
        backend raises ``RuntimeError`` for that batch element. Used by
        tests to exercise the per-sample error path.
    """

    kind = "mock"

    _DEFAULT_TRIGGERS: tuple[str, ...] = (
        "unsafe",
        "harm",
        "attack",
    )

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        raw_triggers = config.args.get("triggers")
        if raw_triggers is None:
            self.triggers: tuple[str, ...] = self._DEFAULT_TRIGGERS
        else:
            self.triggers = tuple(str(t).lower() for t in raw_triggers)
        self.response_format = str(
            config.args.get("response_format", "safe_unsafe")
        )
        self.fail_substring = config.args.get("fail_substring")

    @staticmethod
    def _flatten_text(messages: Sequence[Message]) -> str:
        """Concatenate text from non-system turns.

        The system prompt is the judge's instruction, not user content,
        and frequently contains trigger words like "unsafe" itself.
        Including it would make every sample trip the trigger heuristic.
        """
        chunks: list[str] = []
        for message in messages:
            if message.role == "system":
                continue
            if isinstance(message.content, str):
                chunks.append(message.content)
                continue
            for part in message.content:
                if isinstance(part, TextPart):
                    chunks.append(part.text)
        return " ".join(chunks).lower()

    def _render(self, unsafe: bool) -> str:
        """Render a fake guard response in the requested output format."""
        if self.response_format == "safe_unsafe":
            return "unsafe\nS1" if unsafe else "safe"
        if self.response_format == "yes_no":
            return "yes" if unsafe else "no"
        if self.response_format == "json_rating":
            score = 0.92 if unsafe else 0.05
            categories = '["S1"]' if unsafe else "[]"
            return (
                '{"unsafe_score": '
                f"{score}"
                ', "categories": '
                f"{categories}"
                "}"
            )
        raise ValueError(
            f"unknown mock response_format: {self.response_format!r}"
        )

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[str]:
        """Return a fake guard label per conversation."""
        outputs: list[str] = []
        for conversation in batch:
            text = self._flatten_text(conversation)
            if self.fail_substring and self.fail_substring.lower() in text:
                raise RuntimeError(
                    f"mock backend forced failure: substring "
                    f"{self.fail_substring!r} in input"
                )
            unsafe = any(trigger in text for trigger in self.triggers)
            outputs.append(self._render(unsafe))
        return outputs


@backend_registry.register("mock_classifier")
class MockClassifierBackend(ClassifierBackend):
    """Deterministic classifier backend that returns per-label probs.

    Trigger-word match on user-turn text returns ``{"unsafe": p,
    "safe": 1-p}`` (with p configurable via ``args.unsafe_prob``);
    otherwise returns ``{"unsafe": 1-p, "safe": p}``.
    """

    kind = "mock_classifier"

    _DEFAULT_TRIGGERS: tuple[str, ...] = (
        "unsafe",
        "harm",
        "attack",
    )

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        raw_triggers = config.args.get("triggers")
        if raw_triggers is None:
            self.triggers: tuple[str, ...] = self._DEFAULT_TRIGGERS
        else:
            self.triggers = tuple(str(t).lower() for t in raw_triggers)
        self.unsafe_prob = float(config.args.get("unsafe_prob", 0.92))

    def classify(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[dict[str, float]]:
        """Return mock per-label probability dicts."""
        results: list[dict[str, float]] = []
        for conversation in batch:
            text = MockGenerationBackend._flatten_text(conversation)
            unsafe = any(trigger in text for trigger in self.triggers)
            if unsafe:
                results.append(
                    {
                        "unsafe": self.unsafe_prob,
                        "safe": 1.0 - self.unsafe_prob,
                    }
                )
            else:
                results.append(
                    {
                        "unsafe": 1.0 - self.unsafe_prob,
                        "safe": self.unsafe_prob,
                    }
                )
        return results
