"""Conformance tests for vibecoding oracle adapters.

Parametrizes the reusable eight checks (see ``tests/_vibe_conformance.py``)
over each registered oracle that ships committed fixtures under
``tests/fixtures/vibecoding/<name>/``. The mock oracle is covered today; the
Stage-D adapters (susvibes / secrepobench / ase / securevibebench) plug in
automatically once they register and drop fixtures next to it.

Parser and staging checks run in CI with no Docker. Real out-of-process /
Docker execution is opt-in behind the ``vibe_smoke`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    oracle_registry,
)

from tests import _vibe_conformance as conformance

_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "vibecoding"

# Adapters that have committed conformance fixtures. Stage-D adapters extend
# this list as they land (each ships its own fixtures dir + registration).
_FIXTURED_ADAPTERS = ["mock"]


def _adapter_instance(name: str):
    """Resolve and instantiate a registered oracle adapter by name."""
    ensure_vibe_registrations()
    return oracle_registry.get(name)()


def _check_id(check) -> str:
    """Stable, readable parametrization id for a check function."""
    return check.__name__


def test_mock_oracle_is_registered() -> None:
    """The mock oracle resolves after ``ensure_vibe_registrations()``."""
    ensure_vibe_registrations()
    assert "mock" in oracle_registry.keys()
    adapter = oracle_registry.get("mock")()
    assert adapter.name == "mock"


@pytest.mark.parametrize("adapter_name", _FIXTURED_ADAPTERS)
@pytest.mark.parametrize(
    "check", conformance.CONFORMANCE_CHECKS, ids=_check_id
)
def test_oracle_conformance(adapter_name: str, check) -> None:
    """Run one conformance check against one fixtured adapter."""
    fixtures_dir = _FIXTURES_ROOT / adapter_name
    if not fixtures_dir.exists():
        pytest.skip(f"no fixtures for adapter {adapter_name!r}")
    adapter = _adapter_instance(adapter_name)
    check(adapter, fixtures_dir)


@pytest.mark.vibe_smoke
@pytest.mark.parametrize("adapter_name", _FIXTURED_ADAPTERS)
def test_oracle_smoke(adapter_name: str) -> None:
    """Opt-in real-environment smoke (Docker/venv); skipped in CI.

    Placeholder hook so Stage-D adapters can attach a real end-to-end run
    behind ``GEH_VIBE_SMOKE=1`` without touching the in-process checks above.
    """
    adapter = _adapter_instance(adapter_name)
    assert adapter.name == adapter_name
