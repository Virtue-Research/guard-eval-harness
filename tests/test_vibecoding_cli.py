"""End-to-end tests for the ``geh vibe`` CLI against the mock oracle.

These exercise the full foundation path -- argparse wiring in
``cli/main.py`` -> ``vibecoding.cli.dispatch`` -> ``VibeRunner`` -> reporter --
without any Docker, upstream checkout, or live agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from guard_eval_harness.cli.main import main


def _write_predictions(path: Path) -> None:
    """Write a small BYO predictions JSONL for three mock tasks."""
    rows = [
        {
            "task_id": "mock/default-0",
            "model": "m",
            "kind": "patch",
            "patch": "diff --git a/x b/x\n+secure\n",
            "metadata": {"mock_outcome": "secure_pass"},
        },
        {
            "task_id": "mock/default-1",
            "model": "m",
            "kind": "patch",
            "patch": "diff --git a/y b/y\n+broken\n",
            "metadata": {"mock_outcome": "model_failure"},
        },
        {
            "task_id": "mock/default-2",
            "model": "m",
            "kind": "patch",
            "patch": "diff --git a/z b/z\n+flaky\n",
            "metadata": {"mock_outcome": "infra"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_vibe_run_rejects_multi_trial(tmp_path, capsys):
    """`geh vibe run --trials N>1` fails fast instead of under-sampling.

    A live run scores one candidate per task; accepting --trials 3 while only
    sampling once would silently corrupt trial/pass@k experiments. The guard
    fires before any generation, so no agent/Docker is needed.
    """
    code = main([
        "vibe", "run",
        "--dataset", "mock",
        "--agent", "claude",
        "--trials", "3",
        "--run-dir", str(tmp_path / "run"),
    ])
    out = capsys.readouterr().out
    assert code == 2
    assert "trials" in out.lower()


def test_vibe_run_rejects_concurrency(tmp_path, capsys):
    """`geh vibe run --concurrency N>1` fails fast instead of pretending.

    Nothing threads --concurrency into live generation or oracle evaluation
    yet, so honoring it would report success while still running one task (and
    one Docker container) at a time. The guard fires before any generation, so
    no agent/Docker is needed.
    """
    code = main([
        "vibe", "run",
        "--dataset", "mock",
        "--agent", "claude",
        "--concurrency", "8",
        "--run-dir", str(tmp_path / "run"),
    ])
    out = capsys.readouterr().out
    assert code == 2
    assert "concurrency" in out.lower()


def test_vibe_datasets_lists_mock(capsys):
    """`geh vibe datasets` exits 0 and lists the mock source/oracle."""
    code = main(["vibe", "datasets"])
    out = capsys.readouterr().out
    assert code == 0
    assert "mock" in out
    payload = json.loads(out)
    assert "mock" in payload["task_sources"]
    assert any(row["oracle"] == "mock" for row in payload["oracles"])


def test_vibe_eval_end_to_end(tmp_path, capsys):
    """`geh vibe eval` writes the full run dir and a capability summary."""
    preds = tmp_path / "preds.jsonl"
    _write_predictions(preds)
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"

    # Load exactly the three tasks the predictions cover (a deliberate subset
    # expressed via --limit): the loaded set then matches the predictions, so
    # coverage reconciliation is a no-op and only these three rows are scored.
    code = main([
        "vibe", "eval",
        "--dataset", "mock",
        "--predictions", str(preds),
        "--limit", "3",
        "--run-id", "smoke",
        "--run-dir", str(run_dir),
        "--cache-dir", str(cache_dir),
    ])
    assert code == 0
    capsys.readouterr()  # drain

    for name in ("manifest.json", "results.jsonl", "summary.json", "report.md"):
        assert (run_dir / name).exists(), f"missing {name}"

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == "vibe-summary/1"
    assert "vibecoding_safety_repo_patch_v0" in summary["leaderboard"]
    assert "vibecoding_safety_repo_completion_v0" in summary["leaderboard"]
    # The infra row is excluded from the model denominator; the two model rows
    # stay in.
    assert summary["totals"]["excluded"]["infra_failure"] == 1
    assert summary["totals"]["scored_rows"] == 2
    assert "report.md" in {p.name for p in run_dir.iterdir()}


def test_vibe_eval_cache_hit_on_rerun(tmp_path, capsys):
    """A second identical run reuses cached completed/model rows."""
    preds = tmp_path / "preds.jsonl"
    _write_predictions(preds)
    cache_dir = tmp_path / "cache"

    code1 = main([
        "vibe", "eval",
        "--dataset", "mock",
        "--predictions", str(preds),
        "--run-id", "r1",
        "--run-dir", str(tmp_path / "r1"),
        "--cache-dir", str(cache_dir),
    ])
    assert code1 == 0
    capsys.readouterr()

    cache_files = list((cache_dir / "cache" / "vibecoding").glob("*.json"))
    # secure_pass + model_failure are cacheable; infra is not.
    assert len(cache_files) >= 1

    code2 = main([
        "vibe", "eval",
        "--dataset", "mock",
        "--predictions", str(preds),
        "--run-id", "r2",
        "--run-dir", str(tmp_path / "r2"),
        "--cache-dir", str(cache_dir),
    ])
    assert code2 == 0
    capsys.readouterr()

    s1 = json.loads((tmp_path / "r1" / "summary.json").read_text())
    s2 = json.loads((tmp_path / "r2" / "summary.json").read_text())
    assert s1["totals"]["scored_rows"] == s2["totals"]["scored_rows"]
