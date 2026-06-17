"""Load packaged ``EnvSpec`` catalog entries for vibecoding oracles.

Mirrors the model-catalog convention (``models/catalog/*.yaml`` scanned at
import) so adding a dataset is config-driven: drop an EnvSpec-shaped YAML under
``vibecoding/catalog/`` and it is discoverable by name. The YAML must contain
ONLY :class:`EnvSpec` fields (the model is ``extra="forbid"``); adapter-specific
knobs belong as oracle class constants, not in the EnvSpec.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from guard_eval_harness.vibecoding.schema import EnvSpec

_CATALOG_DIR = Path(__file__).parent / "catalog"


def catalog_dir() -> Path:
    """Return the packaged catalog directory."""
    return _CATALOG_DIR


def available() -> list[str]:
    """List catalog entry names (YAML file stems), sorted."""
    if not _CATALOG_DIR.is_dir():
        return []
    return sorted(p.stem for p in _CATALOG_DIR.glob("*.yaml"))


def load_env_spec(name: str) -> EnvSpec:
    """Load and validate the ``EnvSpec`` named ``name`` from the catalog."""
    path = _CATALOG_DIR / f"{name}.yaml"
    if not path.exists():
        names = ", ".join(available()) or "<none>"
        raise KeyError(
            f"Unknown vibecoding catalog entry {name!r}. Available: {names}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EnvSpec.model_validate(data)


__all__ = ["catalog_dir", "available", "load_env_spec"]
