"""Coverage guard: every vibecoding dataset adapter must be fully wired + tested.

This is the structural safety net for the PR. It fails loudly if anyone adds
or re-introduces a dataset oracle without (a) a registered task source,
(b) a committed catalog YAML, (c) a dedicated standalone smoke-test file, or
(d) visibility through the ``geh vibe datasets`` CLI. It also sanity-checks
that each adapter declares a coherent capability/parallelism surface.

Keeping this green means "every dataset we adapt is reproducible from geh
commands and covered by smoke tests" is enforced, not aspirational.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from guard_eval_harness.cli.main import main
from guard_eval_harness.vibecoding.catalog_loader import load_env_spec
from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    oracle_registry,
    task_source_registry,
)
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    OracleParallelism,
)

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "src" / "guard_eval_harness" / "vibecoding"
_TESTS = _REPO / "tests"
_CATALOG = _PKG / "catalog"

# The "mock" adapter is the in-process reference oracle: it is covered by the
# conformance + runner + cli tests rather than a dataset-specific file, and it
# ships no catalog YAML. Every *dataset* adapter must satisfy the full matrix.
_NON_DATASET = {"mock"}

# The dataset adapters that make up the shipped suite. Update this set when a
# dataset is intentionally added or removed so coverage stays explicit.
EXPECTED_DATASETS = {
    "susvibes",
    "secrepobench",
    "ase",
    "securevibebench",
    "baxbench",
    "seccodebench",
}


def _dataset_oracles() -> list[str]:
    ensure_vibe_registrations()
    return [n for n in oracle_registry.keys() if n not in _NON_DATASET]


def test_expected_datasets_are_registered() -> None:
    """All shipped dataset oracles resolve from the registry."""
    registered = set(_dataset_oracles())
    missing = EXPECTED_DATASETS - registered
    assert not missing, f"dataset oracles not registered: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_dataset_has_task_source(name: str) -> None:
    """Each dataset oracle has a matching registered task source."""
    ensure_vibe_registrations()
    assert name in task_source_registry.keys(), (
        f"no task source registered for {name!r}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_dataset_has_standalone_smoke_test(name: str) -> None:
    """Each dataset adapter ships a dedicated smoke-test file."""
    test_file = _TESTS / f"test_vibecoding_{name}.py"
    assert test_file.is_file(), f"missing smoke test {test_file.name}"
    # A non-trivial test file (mirrors the existing adapters' depth).
    n_tests = test_file.read_text(encoding="utf-8").count("def test_")
    assert n_tests >= 6, (
        f"{test_file.name} has only {n_tests} tests; dataset adapters need a "
        f"comprehensive smoke suite (load/stage/parse/infra/unsupported/...)"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_dataset_has_catalog_yaml(name: str) -> None:
    """Each dataset adapter ships a reference catalog YAML (packaged)."""
    stems = {p.stem for p in _CATALOG.glob("*.yaml")}
    assert any(s == name or s.startswith(name) for s in stems), (
        f"no catalog/*.yaml for {name!r}; have {sorted(stems)}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_catalog_ref_matches_inline_env(name: str) -> None:
    """The packaged catalog pins the SAME upstream_ref as the runtime EnvSpec.

    EnvProvider verifies the checked-out HEAD startswith(upstream_ref), so a
    branch name in the catalog (while the inline EnvSpec pins a SHA) silently
    breaks catalog-driven acquisition. Keep the two in lockstep.
    """
    ensure_vibe_registrations()
    inline_ref = oracle_registry.get(name)().env.upstream_ref
    if inline_ref is None:
        pytest.skip(f"{name} pins no upstream_ref inline")
    stems = [
        p.stem
        for p in _CATALOG.glob("*.yaml")
        if p.stem == name or p.stem.startswith(name)
    ]
    assert stems, f"no catalog yaml for {name!r}"
    catalog_ref = load_env_spec(stems[0]).upstream_ref
    assert catalog_ref == inline_ref, (
        f"catalog {stems[0]}.yaml pins upstream_ref={catalog_ref!r} but the "
        f"inline EnvSpec pins {inline_ref!r}; EnvProvider requires a SHA match"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_dataset_oracle_surface_is_coherent(name: str) -> None:
    """Each oracle declares a valid capability + parallelism surface."""
    ensure_vibe_registrations()
    oracle = oracle_registry.get(name)()
    assert isinstance(oracle.capabilities, OracleCapabilities)
    assert isinstance(oracle.parallelism, OracleParallelism)
    assert oracle.granularity in {"per_task", "batch"}
    assert oracle.artifact_kinds, f"{name} declares no artifact_kinds"
    assert oracle.task_types, f"{name} declares no task_types"
    assert oracle.env is not None and oracle.env.name


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_dataset_has_fixtures_dir(name: str) -> None:
    """Each dataset adapter ships committed fixtures for its smoke test."""
    fixtures = _TESTS / "fixtures" / "vibecoding" / name
    assert fixtures.is_dir() and any(fixtures.rglob("*")), (
        f"missing/empty fixtures dir for {name!r}"
    )


def test_geh_vibe_datasets_lists_every_adapter() -> None:
    """`geh vibe datasets` is the reproducible entry point and lists all."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["vibe", "datasets"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    listed = {row["oracle"] for row in payload["oracles"]}
    missing = EXPECTED_DATASETS - listed
    assert not missing, f"`geh vibe datasets` omits: {sorted(missing)}"
    # Task sources are listed too (reproducibility from the CLI).
    sources = set(payload["task_sources"])
    assert EXPECTED_DATASETS <= sources
