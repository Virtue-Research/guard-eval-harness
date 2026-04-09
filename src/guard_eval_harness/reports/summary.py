"""Summary and comparison helpers built from disk artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from guard_eval_harness.judgment import partial_evaluation_judgment

CORE_COMPARE_METRICS = (
    "accuracy",
    "auprc",
    "auroc",
    "precision",
    "recall",
    "f1",
    "count",
)


def _load_run_artifacts(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load the manifest and per-dataset artifacts for a run."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    dataset_results = []
    datasets_dir = root / "datasets"
    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        dataset_results.append(
            {
                "metadata": json.loads(
                    (dataset_dir / "dataset-manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "metrics": json.loads(
                    (dataset_dir / "metrics.json").read_text(encoding="utf-8")
                ),
            }
        )
    return root, manifest, dataset_results


def _format_metric(value: Any) -> str:
    """Render numeric metrics compactly for human-facing outputs."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_html_report(summary: dict[str, Any]) -> str:
    """Build a small static HTML report from a summary payload."""
    rows = []
    for dataset in summary["datasets"]:
        metrics = dataset["metrics"]
        evaluated = dataset.get("evaluated_sample_count")
        sample_count = dataset.get("sample_count")
        scored_display = _format_metric(
            evaluated if evaluated is not None else metrics.get("count"),
        )
        if (
            evaluated is not None
            and sample_count is not None
            and evaluated < sample_count
        ):
            scored_display += f"/{_format_metric(sample_count)}"
        note = dataset.get("note", "")
        note_row = ""
        if note:
            cols = 13
            note_row = (
                f'<tr><td colspan="{cols}" '
                f'style="color:var(--muted);font-size:0.85em;'
                f'padding:4px 14px 10px">'
                f"{html.escape(note)}</td></tr>"
            )
        rows.append(
            "".join(
                (
                    "<tr>",
                    f"<td>{html.escape(dataset['display_name'])}</td>",
                    f"<td>{html.escape(dataset['name'])}</td>",
                    f"<td>{_format_metric(sample_count)}</td>",
                    f"<td>{_format_metric(dataset.get('unsafe_count'))}</td>",
                    f"<td>{scored_display}</td>",
                    f"<td>{_format_metric(metrics.get('accuracy'))}</td>",
                    f"<td>{_format_metric(metrics.get('auroc'))}</td>",
                    f"<td>{_format_metric(metrics.get('auprc'))}</td>",
                    f"<td>{_format_metric(metrics.get('precision'))}</td>",
                    f"<td>{_format_metric(metrics.get('recall'))}</td>",
                    f"<td>{_format_metric(metrics.get('f1'))}</td>",
                    f"<td>{_format_metric(metrics.get('fpr'))}</td>",
                    f"<td>{_format_metric(metrics.get('fnr'))}</td>",
                    "</tr>",
                    note_row,
                )
            )
        )

    model_name = summary.get("model_name") or "-"
    adapter_name = summary.get("adapter") or "-"
    title = html.escape(summary["run_name"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f2e8;
      --panel: #fffdf9;
      --ink: #162117;
      --muted: #5b6655;
      --accent: #215f4b;
      --line: #d7d2c8;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top, #fffdf7, var(--bg));
      color: var(--ink);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, #fbf7ef, #eef6ef);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 16px 40px rgba(22, 33, 23, 0.08);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 2.2rem;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 24px;
      color: var(--muted);
    }}
    .meta strong {{
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      background: #f1ebde;
      color: var(--accent);
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.75rem;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{title}</h1>
      <div class="meta">
        <span><strong>Status:</strong> {html.escape(summary["status"])}</span>
        <span><strong>Threshold:</strong> {_format_metric(summary["threshold"])}</span>
        <span><strong>Adapter:</strong> {html.escape(adapter_name)}</span>
        <span><strong>Model:</strong> {html.escape(model_name)}</span>
        <span><strong>Datasets:</strong> {summary["dataset_count"]}</span>
      </div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Artifact Name</th>
          <th>Samples</th>
          <th>Unsafe</th>
          <th>Scored</th>
          <th>Accuracy</th>
          <th>AUROC</th>
          <th>AUPRC</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>FPR</th>
          <th>FNR</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def write_html_report(run_dir: str | Path, summary: dict[str, Any]) -> str:
    """Write the static HTML report next to the JSON summary."""
    destination = Path(run_dir) / "report.html"
    destination.write_text(build_html_report(summary), encoding="utf-8")
    return destination.as_posix()


def _metric_delta(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    metric: str,
) -> float | int | None:
    """Return a metric delta only when both inputs are numeric."""
    if left is None or right is None:
        return None
    left_value = left.get(metric)
    right_value = right.get(metric)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value


def build_summary(
    manifest: dict[str, Any],
    dataset_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a summary payload from manifest and dataset metrics."""
    datasets = []
    for result in dataset_results:
        meta = result["metadata"]
        extra_meta = meta.get("metadata", {})
        sample_count = meta.get("sample_count")
        evaluated = extra_meta.get("evaluated_sample_count")
        dropped = extra_meta.get("dropped_sample_count", 0)
        evaluation_judgment = extra_meta.get("evaluation_judgment")
        if evaluation_judgment is None:
            evaluation_judgment = partial_evaluation_judgment(
                dropped_count=dropped,
                total_count=sample_count or 0,
            )

        entry: dict[str, Any] = {
            "name": meta["name"],
            "display_name": meta["display_name"],
            "sample_count": sample_count,
            "unsafe_count": meta.get("unsafe_count"),
            "evaluated_sample_count": evaluated,
            "metrics": result["metrics"],
        }
        if evaluation_judgment is not None:
            entry["evaluation_judgment"] = evaluation_judgment

        if dropped and sample_count:
            rate = dropped / sample_count
            entry["note"] = (
                f"{dropped} of {sample_count} samples "
                f"({rate:.1%}) could not be scored and were "
                f"excluded from metrics"
            )

        datasets.append(entry)
    model = manifest.get("model", {})
    return {
        "run_name": manifest["run_name"],
        "status": manifest["status"],
        "threshold": manifest["threshold"],
        "tool_version": manifest.get("tool_version"),
        "adapter": manifest.get("adapter_capabilities", {}).get("adapter_name"),
        "model_name": model.get("model_name"),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def rebuild_summary(run_dir: str | Path) -> dict[str, Any]:
    """Rebuild a run summary from artifact files on disk."""
    root, manifest, dataset_results = _load_run_artifacts(run_dir)
    summary = build_summary(manifest, dataset_results)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_html_report(root, summary)
    return summary


def load_or_build_summary(run_dir: str | Path) -> dict[str, Any]:
    """Build a summary from disk artifacts without writing anything."""
    _root, manifest, dataset_results = _load_run_artifacts(run_dir)
    return build_summary(manifest, dataset_results)


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """Compare two summaries at a coarse per-dataset level."""
    summary_a = load_or_build_summary(run_a)
    summary_b = load_or_build_summary(run_b)
    metrics_a = {
        item["name"]: item["metrics"] for item in summary_a["datasets"]
    }
    metrics_b = {
        item["name"]: item["metrics"] for item in summary_b["datasets"]
    }

    datasets = sorted(set(metrics_a) | set(metrics_b))
    comparison = {
        "run_a": summary_a["run_name"],
        "run_b": summary_b["run_name"],
        "datasets": [],
    }
    for dataset_name in datasets:
        left = metrics_a.get(dataset_name)
        right = metrics_b.get(dataset_name)
        dataset_comparison = {"name": dataset_name}
        for metric in CORE_COMPARE_METRICS:
            dataset_comparison[f"{metric}_delta"] = _metric_delta(
                left,
                right,
                metric,
            )
        comparison["datasets"].append(dataset_comparison)
    return comparison
