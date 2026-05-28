"""Guard contracts and built-in implementations."""

from guard_eval_harness.guards.base import (
    Guard,
    get_guard_cls,
    guard_registry,
    list_guards,
)
from guard_eval_harness.output_formats import ParsedLabel

# Side-effect imports: register built-in guards.
from guard_eval_harness.guards import (  # noqa: F401
    granite_guardian,
    hf_image_classifier,
    llama_guard,
    llavaguard,
    llm,
    md_judge,
    prompt_guard,
    qwen3guard,
    shieldgemma,
    wildguard,
)

__all__ = [
    "Guard",
    "ParsedLabel",
    "get_guard_cls",
    "guard_registry",
    "list_guards",
]
