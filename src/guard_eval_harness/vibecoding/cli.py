"""`geh vibe ...` command dispatch for the VibeCoding Safety Bench.

The argparse subparsers are built in :mod:`guard_eval_harness.cli.main`
(``_build_vibe``); this module owns only the runtime behavior of each action.
Imports of the heavier runner/report stack happen inside the action handlers
so that classification users never pay the cost and an OSS checkout missing
this subpackage degrades cleanly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def dispatch(args: argparse.Namespace) -> int:
    """Route a parsed ``geh vibe`` namespace to its action handler."""
    action = getattr(args, "vibe_action", None)
    handlers = {
        "run": _run,
        "eval": _eval,
        "datasets": _datasets,
        "acquire": _acquire,
        "doctor": _doctor,
        "report": _report,
    }
    handler = handlers.get(action)
    if handler is None:
        print("Usage: geh vibe {run,eval,datasets,acquire,doctor,report}")
        return 1
    return handler(args)


def _registries() -> tuple[Any, Any]:
    """Import + load the vibecoding registries (sources, oracles)."""
    from guard_eval_harness.vibecoding.registry import (
        ensure_vibe_registrations,
        oracle_registry,
        task_source_registry,
    )

    ensure_vibe_registrations()
    return task_source_registry, oracle_registry


def _datasets(args: argparse.Namespace) -> int:
    """List registered task sources and oracles with their capabilities."""
    sources, oracles = _registries()
    rows: list[dict[str, Any]] = []
    for name, oracle_cls in oracles.items():
        oracle = oracle_cls()
        env = getattr(oracle, "env", None)
        rows.append(
            {
                "oracle": name,
                "task_types": sorted(oracle.task_types),
                "artifact_kinds": sorted(oracle.artifact_kinds),
                "granularity": oracle.granularity,
                "parallelism": oracle.parallelism.model,
                "capabilities": oracle.capabilities.model_dump(mode="json"),
                "license_policy": getattr(env, "license_policy", None),
                "requires_docker": bool(
                    getattr(env, "requires_docker", False)
                ),
            }
        )
    payload = {
        "task_sources": sources.keys(),
        "oracles": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _acquire(args: argparse.Namespace) -> int:
    """Clone + build a dataset's upstream env so ``run``/``eval`` can use it.

    This is the missing setup step: an upstream-backed source reads task files
    from the checkout, so without it ``run``/``eval`` load zero tasks. We call
    :meth:`EnvProvider.ensure_ready` (clone the pinned upstream + build its
    isolated venv) then the oracle's :meth:`prepare_acquisition` post-fixups
    (descriptor materialization / shared ground-truth restore). Idempotent:
    re-running is a no-op unless ``--force``. Inline oracles (mock) have
    nothing to acquire.
    """
    _, oracles = _registries()
    dataset = args.dataset
    try:
        oracle = oracles.get(dataset)()
    except KeyError as exc:
        print(f"Error: {exc}")
        return 1
    env = getattr(oracle, "env", None)
    if (
        env is None
        or getattr(env, "kind", "inline") == "inline"
        or not getattr(env, "upstream_url", None)
    ):
        print(json.dumps({
            "dataset": dataset,
            "status": "inline",
            "detail": "inline oracle: no external upstream to acquire",
        }, indent=2, sort_keys=True))
        return 0

    from guard_eval_harness.vibecoding.envs import EnvProvider

    provider = EnvProvider(env, cache_dir=getattr(args, "cache_dir", None))
    try:
        resolved = provider.ensure_ready(
            force=bool(getattr(args, "force", False))
        )
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        # ensure_ready raises RuntimeError on a clone/checkout/venv-install
        # failure, which is not in the CLI's user-facing exception set; report
        # a structured error (like prepare_acquisition below) so scripts get a
        # clean status + exit 1 instead of a traceback.
        print(json.dumps({
            "dataset": dataset,
            "status": "error",
            "stage": "ensure_ready",
            "detail": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    prepare = getattr(oracle, "prepare_acquisition", None)
    if callable(prepare):
        try:
            prepare(resolved)
        except Exception as exc:  # noqa: BLE001 - report, don't traceback
            print(json.dumps({
                "dataset": dataset,
                "status": "error",
                "stage": "prepare_acquisition",
                "detail": str(exc),
            }, indent=2, sort_keys=True))
            return 1
    print(json.dumps({
        "dataset": dataset,
        "status": "ready",
        "upstream_dir": resolved.upstream_dir,
        "upstream_ref": resolved.upstream_ref,
        "venv_python": resolved.venv_python,
    }, indent=2, sort_keys=True))
    return 0


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read a BYO predictions JSONL file into a list of artifact dicts."""
    entries: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _select_env_provider(dataset: str, cache_dir: str | None):
    """Pick the env provider for ``dataset``'s oracle.

    Inline oracles (the mock) score in-process; upstream-backed oracles get
    the real subprocess :class:`EnvProvider`, which resolves the upstream
    checkout + venv under ``.geh/`` (or ``GEH_CACHE_DIR`` / ``cache_dir``).
    """
    _, oracles = _registries()
    oracle = oracles.get(dataset)()
    env = getattr(oracle, "env", None)
    if env is None or getattr(env, "kind", "inline") == "inline":
        from guard_eval_harness.vibecoding.oracles.mock import (
            InMemoryEnvProvider,
        )

        return InMemoryEnvProvider()
    from guard_eval_harness.vibecoding.envs import EnvProvider

    return EnvProvider(env, cache_dir=cache_dir)


def _eval(args: argparse.Namespace) -> int:
    """Score BYO predictions through an oracle (no live agent)."""
    from guard_eval_harness.vibecoding.interfaces import OracleRunConfig
    from guard_eval_harness.vibecoding.report import (
        build_markdown_report,
        build_vibe_summary,
        write_markdown_report,
        write_vibe_summary,
    )
    from guard_eval_harness.vibecoding.runner import VibeRunner

    dataset = args.dataset
    run_id = getattr(args, "run_id", None) or f"vibe-{dataset}-{int(time.time())}"
    run_dir = getattr(args, "run_dir", None) or str(
        Path("runs") / "vibecoding" / run_id
    )
    predictions = _load_predictions(args.predictions)
    cache_dir = getattr(args, "cache_dir", None)
    env_provider = _select_env_provider(dataset, cache_dir)

    runner = VibeRunner(cache_dir=cache_dir)
    run_config = OracleRunConfig(run_id=run_id, run_dir=run_dir)
    result = runner.run(
        dataset,
        dataset,
        predictions=predictions,
        limit=getattr(args, "limit", None),
        run_config=run_config,
        env_provider=env_provider,
        run_dir=run_dir,
        no_cache=bool(getattr(args, "no_cache", False)),
        allow_empty=bool(getattr(args, "allow_empty", False)),
    )

    manifest = {
        "run_id": result.run_id,
        "source": result.source,
        "oracle": result.oracle,
        "mode": "patch_eval",
        "model": getattr(args, "model", None),
    }
    summary = build_vibe_summary(manifest, result.results, result.tasks)
    write_vibe_summary(result.run_dir, summary)
    write_markdown_report(result.run_dir, build_markdown_report(summary))

    print(json.dumps({
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "n_results": len(result.results),
        "leaderboard": list(summary["leaderboard"].keys()),
    }, indent=2, sort_keys=True))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Check upstream env + adapter conformance for a dataset."""
    _, oracles = _registries()
    dataset = getattr(args, "dataset", None)
    names = [dataset] if dataset else oracles.keys()
    # Respect each dataset's declared Docker requirement by default; only skip
    # the Docker probes when the caller explicitly asks (--skip-docker).
    require_docker = False if getattr(args, "skip_docker", False) else None
    had_error = False
    report: list[dict[str, Any]] = []
    for name in names:
        try:
            oracle = oracles.get(name)()
        except KeyError as exc:
            print(f"Error: {exc}")
            return 1
        checks: list[dict[str, Any]] = []
        env = getattr(oracle, "env", None)
        provider = _maybe_env_provider(env)
        if provider is not None:
            for check in provider.check(require_docker=require_docker):
                ok = bool(getattr(check, "ok", False))
                severity = getattr(check, "severity", "info")
                checks.append({
                    "name": getattr(check, "name", "?"),
                    "ok": ok,
                    "severity": severity,
                    "detail": getattr(check, "detail", ""),
                })
                if not ok and severity == "error":
                    had_error = True
        else:
            checks.append({
                "name": "env",
                "ok": True,
                "severity": "info",
                "detail": "inline oracle: no external environment to check",
            })
        report.append({"oracle": name, "checks": checks})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if had_error else 0


def _maybe_env_provider(env: Any) -> Any:
    """Return an EnvProvider for ``env`` if it targets a real upstream."""
    if env is None or not getattr(env, "upstream_url", None):
        return None
    try:
        from guard_eval_harness.vibecoding.envs import EnvProvider
    except Exception:  # noqa: BLE001 - envs is optional at doctor time
        return None
    return EnvProvider(env)


def _report(args: argparse.Namespace) -> int:
    """Rebuild ``summary.json`` + ``report.md`` from a run dir."""
    from guard_eval_harness.vibecoding.report import rebuild_vibe_summary

    summary = rebuild_vibe_summary(args.run_dir)
    print(json.dumps({
        "run_dir": args.run_dir,
        "summary": str(Path(args.run_dir) / "summary.json"),
        "scored_rows": summary.get("totals", {}).get("scored_rows"),
    }, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    """Drive a live agent over a dataset, then score through the oracle."""
    trials = int(getattr(args, "trials", 1) or 1)
    if trials != 1:
        # A live run scores one candidate per task; looping here would emit
        # rows that still say trial 0 / seed null until trial stamping +
        # aggregation land. Fail loudly rather than silently under-sample.
        print(
            f"--trials={trials} is not supported yet: a live run scores one "
            "candidate per task. For trials / pass@k, launch separate runs "
            "with distinct --run-id (multi-trial aggregation lands with the "
            "trial-stamping work)."
        )
        return 2

    concurrency = int(getattr(args, "concurrency", 1) or 1)
    if concurrency != 1:
        # A live run generates and scores tasks serially; nothing threads this
        # value into generation or evaluation yet (and Docker-backed oracles
        # like SecureVibeBench run one container at a time by design). Reject
        # rather than report success while silently ignoring the request.
        print(
            f"--concurrency={concurrency} is not supported yet: a live run "
            "generates and scores tasks serially (Docker-backed oracles like "
            "SecureVibeBench run one container at a time). Re-run with "
            "--concurrency 1, or shard the dataset across separate runs with "
            "distinct --run-id."
        )
        return 2
    from guard_eval_harness.vibecoding.interfaces import OracleRunConfig
    from guard_eval_harness.vibecoding.report import (
        build_markdown_report,
        build_vibe_summary,
        write_markdown_report,
        write_vibe_summary,
    )
    from guard_eval_harness.vibecoding.runner import VibeRunner

    dataset = args.dataset
    run_id = getattr(args, "run_id", None) or f"vibe-{dataset}-{int(time.time())}"
    run_dir = getattr(args, "run_dir", None) or str(
        Path("runs") / "vibecoding" / run_id
    )
    cache_dir = getattr(args, "cache_dir", None)
    env_provider = _select_env_provider(dataset, cache_dir)

    runner = VibeRunner(cache_dir=cache_dir)
    run_config = OracleRunConfig(run_id=run_id, run_dir=run_dir)
    result = runner.run(
        dataset,
        dataset,
        mode="live_agent",
        agent=args.agent,
        agent_model=getattr(args, "model", None),
        limit=getattr(args, "limit", None),
        run_config=run_config,
        env_provider=env_provider,
        run_dir=run_dir,
        no_cache=bool(getattr(args, "no_cache", False)),
        allow_empty=bool(getattr(args, "allow_empty", False)),
    )

    manifest = {
        "run_id": result.run_id,
        "source": result.source,
        "oracle": result.oracle,
        "mode": "live_agent",
        "agent": args.agent,
        "model": getattr(args, "model", None),
    }
    summary = build_vibe_summary(manifest, result.results, result.tasks)
    write_vibe_summary(result.run_dir, summary)
    write_markdown_report(result.run_dir, build_markdown_report(summary))

    print(json.dumps({
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "agent": args.agent,
        "n_results": len(result.results),
        "leaderboard": list(summary["leaderboard"].keys()),
    }, indent=2, sort_keys=True))
    return 0


__all__ = ["dispatch"]
