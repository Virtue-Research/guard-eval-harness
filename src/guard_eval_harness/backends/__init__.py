"""Backend contracts + built-in implementations."""

from guard_eval_harness.backends.base import (
    Backend,
    BackendConfig,
    ClassifierBackend,
    GenerationBackend,
    backend_registry,
    get_backend_cls,
    list_backends,
)

# Side-effect imports: register built-in backends.
from guard_eval_harness.backends import (  # noqa: F401
    hf_generate,
    hf_image_classifier,
    hf_text_classifier,
    hf_vlm,
    mock,
    openai_compat,
)

__all__ = [
    "Backend",
    "BackendConfig",
    "ClassifierBackend",
    "GenerationBackend",
    "backend_registry",
    "get_backend_cls",
    "list_backends",
]
