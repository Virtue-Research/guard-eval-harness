"""Guard profile registry — bundled YAML configs for known-good guards.

A *profile* is a complete ``model:`` block (guard + backend + args)
ready to drop into a run config. Users reference a profile by slug:

.. code-block:: yaml

    model:
      profile: llama-guard-3-8b   # ← resolves from profiles/llama-guard-3-8b.yaml

    # Optional overrides — deep-merged onto the profile:
    model:
      profile: llama-guard-3-8b
      backend:
        args:
          device: cuda

Each profile YAML mirrors the ``model:`` schema exactly. Loading a
profile is a single read; merge happens in the config loader.
"""

import importlib.resources
from typing import Any

import yaml

_PROFILES_PACKAGE = "guard_eval_harness.guards.profiles"


def _profiles_dir() -> Any:
    """Return the package-data traversable pointing at this directory."""
    return importlib.resources.files(_PROFILES_PACKAGE)


def list_profiles() -> list[str]:
    """Return the slugs of all bundled profiles, sorted."""
    slugs: list[str] = []
    for entry in _profiles_dir().iterdir():
        name = entry.name
        if name.endswith(".yaml") and not name.startswith("_"):
            slugs.append(name[: -len(".yaml")])
    return sorted(slugs)


def load_profile(slug: str) -> dict[str, Any]:
    """Read profile ``<slug>.yaml`` from package data into a dict.

    Raises ``KeyError`` if the slug is not bundled. Raises
    ``ValueError`` if the YAML doesn't parse as a mapping.
    """
    candidate = _profiles_dir() / f"{slug}.yaml"
    if not candidate.is_file():
        available = ", ".join(list_profiles()) or "<none>"
        raise KeyError(
            f"profile {slug!r} is not bundled. Available: {available}"
        )
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"profile {slug!r}: top-level must be a mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


def deep_merge(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``overrides`` onto ``base`` (overrides win).

    Lists and scalars are replaced wholesale; dicts merge per-key.
    Returns a new dict; inputs are not mutated.
    """
    result: dict[str, Any] = dict(base)
    for key, override_value in overrides.items():
        base_value = result.get(key)
        if (
            isinstance(base_value, dict)
            and isinstance(override_value, dict)
        ):
            result[key] = deep_merge(base_value, override_value)
        else:
            result[key] = override_value
    return result


__all__ = [
    "deep_merge",
    "list_profiles",
    "load_profile",
]
