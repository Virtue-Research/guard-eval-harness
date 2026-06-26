"""Capability-scoped reporting for the VibeCoding Safety Bench.

This module arranges the metrics produced by :mod:`metrics` into capability
scoped leaderboard sections, auxiliary rates, breakdowns, and a failure
taxonomy, then renders ``summary.json`` and ``report.md``. The metric math
lives in :mod:`metrics`; ``report.py`` only arranges and renders, so the
classification reporter (``reports/summary.py``) is intentionally untouched --
its flat accuracy/F1 shape cannot represent tri-state nulls, infra exclusions,
or capability-scoped tracks.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from guard_eval_harness.execution.artifacts import atomic_text_writer, dump_json
from guard_eval_harness.vibecoding.metrics import compute_vibe_metrics
from guard_eval_harness.vibecoding.results import VibeTaskResult
from guard_eval_harness.vibecoding.schema import VibeTask

SUMMARY_SCHEMA_VERSION = "vibe-summary/1"

# Headline, capability-scoped leaderboards (doc: "target-secure by track").
# Each headline track requires functional tests + target-vuln detection and
# scores ``target_secure_success`` over only the listed datasets.
HEADLINE_TRACKS: tuple[dict[str, Any], ...] = (
    {
        "key": "vibecoding_safety_repo_patch_v0",
        "tier": "target_secure",
        "metric": "target_secure_success",
        "datasets": ("susvibes", "securevibebench"),
        "required_capabilities": (
            "runs_functional_tests",
            "detects_target_vuln",
        ),
    },
    {
        "key": "vibecoding_safety_repo_completion_v0",
        "tier": "target_secure",
        "metric": "target_secure_success",
        "datasets": ("secrepobench",),
        "required_capabilities": (
            "runs_functional_tests",
            "detects_target_vuln",
        ),
    },
    # Multi-turn (compositional-chain) family. MT-Sec is a DETERMINISTIC,
    # fully-executable target-secure adapter (``llm_judge=False`` +
    # ``deterministic=True``, ``runs_functional_tests`` + ``detects_target_vuln``)
    # -- the adapter, catalog, and its docstring all place it on the
    # target-secure leaderboard, so it earns a headline track. It is its OWN
    # track (not pooled into the repo patch/completion families above) because a
    # track scores ``target_secure_success`` over its datasets pooled together,
    # and mixing function-level multi-turn rates with repo-level patch/completion
    # rates would conflate distinct task families. In v0 MT-Sec is an
    # offline / BYO adapter (the live driver does not yet emit
    # ``compositional_chain`` artifacts), so this track is populated by
    # offline / BYO-scored rows; with none present it reports ``n_scored=0``
    # like any unrun track rather than being silently absent.
    {
        "key": "vibecoding_safety_multiturn_v0",
        "tier": "target_secure",
        "metric": "target_secure_success",
        "datasets": ("mtsec",),
        "required_capabilities": (
            "runs_functional_tests",
            "detects_target_vuln",
        ),
    },
)

# Secondary, capability-scoped leaderboards reported where supported but never
# mixed into the headline score.
SECONDARY_TRACKS: tuple[dict[str, Any], ...] = (
    {
        "key": "strict_secure",
        "tier": "strict_secure",
        "metric": "strict_secure_success",
        "datasets": ("securevibebench",),
        "required_capabilities": (
            "runs_functional_tests",
            "detects_target_vuln",
            "detects_new_vuln",
        ),
    },
)


def _track_section(
    track: dict[str, Any],
    results: Sequence[VibeTaskResult],
    tasks: Sequence[VibeTask],
) -> dict[str, Any]:
    """Score one capability-scoped track over its own datasets only."""
    datasets = set(track["datasets"])
    subset = [r for r in results if r.source_dataset in datasets]
    subset_ids = {r.task_id for r in subset}
    subset_tasks = [t for t in tasks if t.id in subset_ids]
    metrics = compute_vibe_metrics(subset, subset_tasks)
    cell = metrics["cells"][track["metric"]]
    return {
        "tier": track["tier"],
        "metric": track["metric"],
        "datasets": list(track["datasets"]),
        "required_capabilities": list(track["required_capabilities"]),
        "rate": cell["rate"],
        "n_scored": cell["n_scored"],
        "excluded_null_verdict": cell["excluded_null"],
        "n_in_denominator": metrics["n_in_denominator"],
        "excluded_infra": metrics["excluded_infra"],
        "excluded_unsupported": metrics["excluded_unsupported"],
        "cheating_detected": metrics["cheating_detected"],
    }


def build_vibe_summary(
    manifest: dict[str, Any],
    results: Sequence[VibeTaskResult],
    tasks: Sequence[VibeTask],
) -> dict[str, Any]:
    """Arrange results + manifest into the capability-scoped summary dict.

    The headline leaderboards are scored per track (never mixing
    target-vuln-only and new-vuln datasets), with an ``overall`` view across
    all in-denominator rows for at-a-glance reading. Every rate carries its
    denominator and excluded counts; no generic ``secure_success`` alias is
    emitted.
    """
    base = compute_vibe_metrics(list(results), list(tasks))
    cells = base["cells"]

    by_origin = Counter(r.failure_origin for r in results)
    by_reason = Counter(
        r.failure_reason for r in results if r.failure_reason is not None
    )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run": {
            "run_id": manifest.get("run_id"),
            "source": manifest.get("source"),
            "oracle": manifest.get("oracle"),
            "mode": manifest.get("mode"),
            "model": manifest.get("model"),
        },
        "totals": {
            "tasks": len(tasks),
            "result_rows": base["n_total"],
            "scored_rows": base["n_in_denominator"],
            "excluded": {
                "infra_failure": base["excluded_infra"],
                "unsupported": base["excluded_unsupported"],
            },
            "cheating_detected": base["cheating_detected"],
        },
        "leaderboard": {
            track["key"]: _track_section(track, results, tasks)
            for track in HEADLINE_TRACKS
        },
        "secondary_leaderboards": {
            track["key"]: _track_section(track, results, tasks)
            for track in SECONDARY_TRACKS
        },
        "overall": cells,
        "auxiliary_rates": {
            "functional_only": cells["functional_only"],
            "oracle_secure": cells["oracle_secure"],
            "functional_to_secure_gap": base["functional_to_secure_gap"],
        },
        "breakdowns": {
            "per_cwe": base["by_cwe"],
            "per_dataset": base["by_dataset"],
            "per_task_type": base["by_task_type"],
        },
        "failures": {
            "by_origin": dict(by_origin),
            "by_reason": dict(by_reason),
        },
        "quality_gate": base["quality_gate"],
    }


def write_vibe_summary(
    run_dir: str | Path, summary: dict[str, Any]
) -> Path:
    """Write ``summary.json`` under ``run_dir`` and return its path."""
    path = Path(run_dir) / "summary.json"
    dump_json(path, summary)
    return path


def _fmt_rate(value: Any) -> str:
    """Render a rate (or ``None``) for a Markdown table cell."""
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from the summary dict."""
    run = summary.get("run", {})
    totals = summary.get("totals", {})
    lines: list[str] = []
    lines.append("# VibeCoding Safety Bench Report")
    lines.append("")
    lines.append(f"- Run: `{run.get('run_id')}`")
    lines.append(f"- Source / oracle: `{run.get('source')}` / "
                 f"`{run.get('oracle')}`")
    lines.append(f"- Tasks: {totals.get('tasks')} | "
                 f"result rows: {totals.get('result_rows')} | "
                 f"scored: {totals.get('scored_rows')}")
    excluded = totals.get("excluded", {})
    lines.append(f"- Excluded: infra={excluded.get('infra_failure')} "
                 f"unsupported={excluded.get('unsupported')} | "
                 f"cheating={totals.get('cheating_detected')}")
    lines.append("")

    lines.append("## Headline leaderboards (target-secure by track)")
    lines.append("")
    lines.append("| Track | Rate | Scored | Excl. null | Excl. infra |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key, section in summary.get("leaderboard", {}).items():
        lines.append(
            f"| {key} | {_fmt_rate(section.get('rate'))} | "
            f"{section.get('n_scored')} | "
            f"{section.get('excluded_null_verdict')} | "
            f"{section.get('excluded_infra')} |"
        )
    lines.append("")

    lines.append("## Auxiliary rates")
    lines.append("")
    aux = summary.get("auxiliary_rates", {})
    fo = aux.get("functional_only", {})
    osec = aux.get("oracle_secure", {})
    lines.append(f"- Functional-only: {_fmt_rate(fo.get('rate'))} "
                 f"(n={fo.get('n_scored')})")
    lines.append(f"- Oracle-secure: {_fmt_rate(osec.get('rate'))} "
                 f"(n={osec.get('n_scored')})")
    lines.append("- Functional-to-secure gap: "
                 f"{_fmt_rate(aux.get('functional_to_secure_gap'))}")
    lines.append("")

    failures = summary.get("failures", {})
    lines.append("## Failure taxonomy")
    lines.append("")
    lines.append(f"- By origin: {failures.get('by_origin', {})}")
    lines.append(f"- By reason: {failures.get('by_reason', {})}")
    lines.append("")

    gate = summary.get("quality_gate", {})
    lines.append(f"Quality gate: **{gate}**")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    run_dir: str | Path, markdown: str
) -> Path:
    """Write ``report.md`` under ``run_dir`` and return its path."""
    path = Path(run_dir) / "report.md"
    with atomic_text_writer(path) as handle:
        handle.write(markdown)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (empty if absent)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rebuild_vibe_summary(run_dir: str | Path) -> dict[str, Any]:
    """Recompute ``summary.json`` + ``report.md`` from on-disk run artifacts.

    Reads ``manifest.json``, ``results.jsonl``, and ``tasks.jsonl`` under
    ``run_dir`` and rewrites the summary/report so a run can be re-reported
    without re-evaluating.
    """
    base = Path(run_dir)
    manifest_path = base / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    results = [
        VibeTaskResult.model_validate(row)
        for row in _read_jsonl(base / "results.jsonl")
    ]
    tasks = [
        VibeTask.model_validate(row)
        for row in _read_jsonl(base / "tasks.jsonl")
    ]
    summary = build_vibe_summary(manifest, results, tasks)
    write_vibe_summary(base, summary)
    write_markdown_report(base, build_markdown_report(summary))
    return summary


__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "HEADLINE_TRACKS",
    "SECONDARY_TRACKS",
    "build_vibe_summary",
    "write_vibe_summary",
    "build_markdown_report",
    "write_markdown_report",
    "rebuild_vibe_summary",
]
