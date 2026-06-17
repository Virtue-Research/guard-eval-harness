"""BaxBench task source.

Loads the upstream BaxBench scenarios (``src/scenarios/<id>/``) into normalized
:class:`VibeTask` records. BaxBench is a ``project_scaffold`` benchmark: the
model is asked to generate a complete backend web application that implements a
functional spec (OpenAPI / text), and the upstream harness then builds the app
in a per-(scenario, env) Docker image and runs functional + security tests
against the running server.

Each BaxBench *task* is the cross product of a scenario (e.g. ``Calculator``)
and a backend env/framework (e.g. ``Python-Flask``), so the normalized task id
is ``baxbench/<scenario>__<env>``. The upstream scenario modules are Python, so
GEH does not import them; instead the task source reads a small per-scenario
descriptor (``scenario.json``) that mirrors the upstream
:class:`scenarios.base.Scenario` fields the adapter actually needs:

    {
      "id": "Calculator",
      "short_app_description": "calculator web app",
      "instructions": "<functional spec text>",
      "cwes": ["CWE-94", "CWE-400"],   # from the security tests (optional)
      "envs": ["Python-Flask", "Python-FastAPI"]   # optional restriction
    }

The ``envs`` field is optional; when absent the source enumerates every env in
the configured env list (constructor ``envs=`` / module default). Tests pass a
``scenarios_dir`` override pointing at a tiny fixture so loading needs neither
Docker nor the full upstream checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guard_eval_harness.execution.artifacts import atomic_text_writer
from guard_eval_harness.vibecoding.envs import EnvProvider
from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# Per-scenario descriptor file the source reads inside ``<scenarios_dir>/<id>/``.
_SCENARIO_FILE = "scenario.json"

# All 14 upstream backend env ids (``Env.id == "<language>-<framework>"``). The
# descriptor export pins each scenario to this full list so the source
# enumerates the paper-parity 28 scenarios x 14 envs = 392 tasks.
_ALL_ENVS = (
    "Python-aiohttp",
    "Python-Django",
    "Python-FastAPI",
    "Python-Flask",
    "JavaScript-express",
    "JavaScript-fastify",
    "JavaScript-koa",
    "JavaScript-nest",
    "Go-Fiber",
    "Go-Gin",
    "Go-net/http",
    "PHP-Lumen",
    "Ruby-Rails",
    "Rust-Actix",
)

# Manifest the materializer writes (atomically, last) beside the descriptors.
# It records the env list and descriptor count of the completed export, so
# :meth:`BaxBenchTaskSource._descriptors_current` can tell a CURRENT set from
# a stale or partial one: descriptors written by an older exporter (no
# manifest, or a different env list) and interrupted exports (manifest count
# disagrees with the files on disk) both trigger a refresh instead of being
# silently enumerated.
_DESCRIPTOR_MANIFEST = ".geh_descriptors.json"

# Upstream ships Python scenario modules, not the small ``scenario.json`` this
# source reads, so a fresh checkout materializes zero tasks. This snippet runs
# IN the provisioned venv (where the upstream ``scenarios`` package and its
# ``docker`` / ``yaml`` deps import) to export each ``scenarios.all_scenarios``
# entry into ``<scenarios_dir>/<id>/scenario.json``, carrying only the fields
# the adapter needs (id, short description, the OpenAPI/text spec as the task
# instructions, and the scenario's potential CWEs). Parameters arrive via env
# vars (not string interpolation) so the script body stays literal. Idempotent
# and concurrency-safe: a descriptor whose content is unchanged is left
# untouched, and changed files are replaced via a same-dir temp + os.replace
# so a concurrent reader never sees a torn write.
_DESCRIPTOR_EXPORT = r"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["BAX_SRC"])
import scenarios

scen_dir = Path(os.environ["BAX_SCEN_DIR"])
env_ids = json.loads(os.environ["BAX_ENV_IDS"])
written = 0
for s in scenarios.all_scenarios:
    cwes = sorted("CWE-%d" % c.value["num"] for c in s.potential_cwes)
    instructions = (s.api_spec or s.text_spec or "").strip()
    payload = {
        "id": s.id,
        "short_app_description": s.short_app_description,
        "instructions": instructions,
        "cwes": cwes,
        "envs": list(env_ids),
    }
    d = scen_dir / s.id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "scenario.json"
    new = json.dumps(payload, indent=2, sort_keys=True)
    old = p.read_text(encoding="utf-8") if p.exists() else None
    if old != new:
        tmp = p.with_name(".scenario.json.tmp-%d" % os.getpid())
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, p)
        written += 1
# Prune descriptors whose scenario no longer exists upstream: a pin refresh
# does not clean untracked files, so a removed/renamed scenario would
# otherwise keep loading as a ghost task.
current_ids = {s.id for s in scenarios.all_scenarios}
removed = 0
for stale in scen_dir.glob("*/scenario.json"):
    if stale.parent.name not in current_ids:
        stale.unlink()
        removed += 1
print(json.dumps(
    {
        "n_scenarios": len(scenarios.all_scenarios),
        "n_written": written,
        "n_removed": removed,
    }
))
"""

# Default backend envs (``Env.id == "<language>-<framework>"`` upstream). Kept
# as a small representative slice so a scenario with no explicit ``envs`` field
# still expands to a deterministic set of (scenario, env) tasks.
_DEFAULT_ENVS = (
    "Python-Flask",
    "Python-FastAPI",
    "Python-Django",
    "Python-aiohttp",
    "JavaScript-express",
    "Go-Gin",
)


def _as_list(value: Any) -> list[str]:
    """Coerce a scalar/list label field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def task_id_for(scenario: str, env: str) -> str:
    """Build the normalized ``baxbench/<scenario>__<env>`` task id."""
    return f"baxbench/{scenario}__{env}"


def split_task_id(task_id: str) -> tuple[str, str]:
    """Recover ``(scenario, env)`` from a ``baxbench/<scenario>__<env>`` id."""
    bare = task_id.split("/", 1)[-1]
    scenario, _, env = bare.partition("__")
    return scenario, env


@task_source_registry.register("baxbench")
class BaxBenchTaskSource(TaskSource):
    """Yield ``project_scaffold`` tasks for each (scenario, env) pair."""

    name = "baxbench"

    def __init__(
        self,
        *,
        scenarios_dir: str | Path | None = None,
        dataset_path: str | Path | None = None,
        envs: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Optionally override the scenarios dir / env list (used by fixtures).

        ``scenarios_dir`` (or its alias ``dataset_path``) points at a directory
        of ``<scenario_id>/scenario.json`` descriptors. ``envs`` overrides the
        default backend-env enumeration.
        """
        root = scenarios_dir if scenarios_dir is not None else dataset_path
        self._scenarios_dir = Path(root) if root is not None else None
        self._envs = list(envs) if envs is not None else list(_DEFAULT_ENVS)

    def _env_provider(self, cache_dir: str | Path | None) -> EnvProvider:
        """Build the BaxBench env provider (clone + venv under the cache)."""
        from guard_eval_harness.vibecoding.oracles.baxbench import (
            BaxBenchOracle,
        )

        return EnvProvider(BaxBenchOracle.env, cache_dir=cache_dir)

    def _resolved_scenarios_dir(
        self, cache_dir: str | Path | None = None
    ) -> Path:
        """Explicit override, else the env provider's .geh checkout."""
        if self._scenarios_dir is not None:
            return self._scenarios_dir
        provider = self._env_provider(cache_dir)
        return Path(provider.resolve().upstream_dir) / "src" / "scenarios"

    def _descriptors_current(
        self, scenarios_dir: Path, expected_ref: str
    ) -> bool:
        """True when the on-disk descriptor set is a COMPLETE current export.

        A single ``scenario.json`` is no proof of currency: an older exporter
        (different env list), an interrupted export, or a concurrent partial
        write all leave files behind that would silently load a wrong task
        set (e.g. 28 single-env tasks instead of 392). Currency therefore
        requires ALL of: the manifest the materializer wrote LAST (existing,
        carrying exactly the env list this source materializes and the
        CURRENT pinned ``upstream_ref`` -- a pin bump must invalidate
        descriptors exported from the previous checkout -- with a descriptor
        count matching the files on disk) AND every on-disk descriptor's own
        ``envs`` field agreeing with that list -- a legacy exporter can
        overwrite the descriptors with a single-env list while leaving the
        manifest untouched, so the manifest alone could vouch for files it
        did not write.
        """
        manifest_path = scenarios_dir / _DESCRIPTOR_MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(manifest, dict):
            return False
        if manifest.get("env_ids") != list(_ALL_ENVS):
            return False
        if manifest.get("upstream_ref") != expected_ref:
            return False
        descriptors = list(scenarios_dir.glob(f"*/{_SCENARIO_FILE}"))
        if not descriptors or len(descriptors) != manifest.get(
            "descriptor_count"
        ):
            return False
        for path in descriptors:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(data, dict) or data.get("envs") != list(
                _ALL_ENVS
            ):
                return False
        return True

    def _ensure_descriptors(
        self, provider: EnvProvider, scenarios_dir: Path
    ) -> None:
        """Materialize ``<id>/scenario.json`` from the upstream checkout.

        Upstream ships Python scenario modules, so the descriptors the source
        reads do not exist until exported. Unless the on-disk set verifies as
        a complete current export for the CURRENT upstream pin (see
        :meth:`_descriptors_current`),
        provision the env (clone + venv; idempotent) and run
        :data:`_DESCRIPTOR_EXPORT` through the venv to (re)write them --
        refreshing stale descriptors from older exporters, completing
        partial sets, and pruning descriptors for scenarios the current
        upstream pin no longer ships (untracked leftovers survive a
        checkout refresh), not just filling an empty dir. The manifest recording
        the export is written atomically LAST, so an interrupted export is
        re-run on the next load. A checkout whose manifest verifies -- the
        common provisioned-host case after the first load -- is a fast no-op
        that never provisions.
        """
        expected_ref = str(
            getattr(
                getattr(provider, "env_spec", None), "upstream_ref", ""
            )
            or ""
        )
        if self._descriptors_current(scenarios_dir, expected_ref):
            return
        resolved = provider.ensure_ready()
        result = provider.run(
            [resolved.venv_python, "-c", _DESCRIPTOR_EXPORT],
            run_dir=resolved.cache_dir,
            timeout_s=300,
            extra_env={
                "BAX_SRC": str(Path(resolved.upstream_dir) / "src"),
                "BAX_SCEN_DIR": str(scenarios_dir),
                "BAX_ENV_IDS": json.dumps(list(_ALL_ENVS)),
            },
        )
        if result.returncode != 0:
            raise RuntimeError(
                "BaxBench descriptor materialization failed "
                f"(rc={result.returncode}); see "
                f"{result.stderr_path or result.stderr[-2000:]}"
            )
        count = len(list(scenarios_dir.glob(f"*/{_SCENARIO_FILE}")))
        if count == 0:
            raise RuntimeError(
                "BaxBench descriptor materialization wrote no descriptors "
                f"under {scenarios_dir}"
            )
        with atomic_text_writer(scenarios_dir / _DESCRIPTOR_MANIFEST) as fh:
            fh.write(
                json.dumps(
                    {
                        "env_ids": list(_ALL_ENVS),
                        "descriptor_count": count,
                        "upstream_ref": expected_ref,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[VibeTask]:
        """Return one task per (scenario, env), sorted by task id."""
        if self._scenarios_dir is not None:
            scenarios_dir = self._scenarios_dir
        else:
            provider = self._env_provider(cache_dir)
            scenarios_dir = (
                Path(provider.resolve().upstream_dir) / "src" / "scenarios"
            )
            self._ensure_descriptors(provider, scenarios_dir)
        descriptors = sorted(
            scenarios_dir.glob(f"*/{_SCENARIO_FILE}"),
            key=lambda p: p.parent.name,
        )
        tasks: list[VibeTask] = []
        for path in descriptors:
            tasks.extend(self._load_scenario(path))
        tasks.sort(key=lambda t: t.id)
        if limit is not None:
            return tasks[: max(0, int(limit))]
        return tasks

    def _load_scenario(self, path: Path) -> list[VibeTask]:
        """Expand one ``scenario.json`` into its (scenario, env) tasks."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []

        scenario_id = str(data.get("id") or path.parent.name)
        # Functional spec the model must implement. Prefer an explicit
        # ``instructions``; fall back to the upstream text/openapi spec fields.
        instructions = (
            data.get("instructions")
            or data.get("text_spec")
            or data.get("api_spec")
            or ""
        )
        short_desc = data.get("short_app_description")
        if short_desc and short_desc not in str(instructions):
            instructions = f"{short_desc}\n\n{instructions}"

        cwe = _as_list(data.get("cwes") or data.get("cwe"))

        scenario_envs = _as_list(data.get("envs")) or self._envs
        tasks: list[VibeTask] = []
        for env in scenario_envs:
            tasks.append(
                VibeTask(
                    id=task_id_for(scenario_id, env),
                    source_dataset="baxbench",
                    task_type="project_scaffold",
                    instructions=str(instructions),
                    # The candidate is a freshly generated backend app, not a
                    # checked-out repo, so there is no upstream repo/commit.
                    repo=RepoSpec(workdir="code"),
                    labels=TaskLabels(cwe=cwe),
                    environment=TaskEnvironmentRef(
                        oracle="baxbench",
                        requires_docker=True,
                    ),
                )
            )
        return tasks
