"""Main CLI for the guard eval harness (YAML-driven)."""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError

from guard_eval_harness.config import (
    detect_config_version,
    load_config_from_path,
)
from guard_eval_harness.runner import run_from_config
from guard_eval_harness.registry import (
    dataset_registry,
    ensure_builtin_registrations,
)


_AVAILABLE_METRICS = (
    "accuracy",
    "auprc",
    "auroc",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "tp",
    "tn",
    "fp",
    "fn",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="geh",
        description="Guard Eval Harness — evaluate safety guards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  geh run --config examples/mock-jsonl.yaml\n"
            "  geh list datasets\n"
            "  geh validate --config my-run.yaml\n"
            "  geh inspect --run-dir out/my-run"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full Python tracebacks.",
    )
    parser.set_defaults(func=lambda _args: parser.print_help() or 0)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    _build_run(subparsers)
    _build_list(subparsers)
    _build_validate(subparsers)
    _build_inspect(subparsers)
    return parser


def _build_run(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Execute a benchmark run from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to a YAML config.")
    p.add_argument("--output-dir", help="Override output.run_dir.")
    p.add_argument("--threshold", type=float, help="Override threshold.")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; require an empty output dir.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe the output dir before running.",
    )
    p.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Skip inference; just recompute metrics from existing predictions.",
    )
    p.set_defaults(func=_cmd_run)


def _build_list(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "list",
        help="List available datasets / guards / backends / policies / metrics.",
    )
    p.add_argument(
        "what",
        choices=("datasets", "guards", "backends", "policies",
                 "profiles", "output_formats", "metrics"),
    )
    p.set_defaults(func=_cmd_list)


def _build_validate(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "validate",
        help="Load a config and confirm all datasets can be materialized.",
    )
    p.add_argument("--config", required=True, help="Path to a YAML config.")
    p.set_defaults(func=_cmd_validate)


def _build_inspect(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect",
        help="Inspect manifest, summary, and per-dataset artifacts.",
    )
    p.add_argument("--run-dir", required=True, help="Run directory.")
    p.set_defaults(func=_cmd_inspect)


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute `geh run --config <yaml>`."""
    if detect_config_version(args.config) != 2:
        raise ValueError(
            f"{args.config}: only YAML configs are supported "
            "(top-level `version: 2`)."
        )
    if args.no_resume and args.overwrite:
        raise ValueError("--no-resume and --overwrite are mutually exclusive")

    config = load_config_from_path(
        args.config,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    if args.no_resume:
        config = config.model_copy(
            update={"output": config.output.model_copy(update={"resume": False})}
        )
    if args.overwrite:
        config = config.model_copy(
            update={"output": config.output.model_copy(
                update={"resume": False, "overwrite": True}
            )}
        )
    result = run_from_config(
        config,
        recompute_metrics_only=args.recompute_metrics,
    )
    _emit_json({
        "run_dir": result.run_dir,
        "manifest_path": result.manifest_path,
        "summary_path": result.summary_path,
        "completed_predictions": result.completed_predictions,
        "failed_predictions": result.failed_predictions,
    })
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Execute `geh list <what>`."""
    ensure_builtin_registrations()
    if args.what == "datasets":
        payload = {"datasets": dataset_registry.keys()}
    elif args.what == "guards":
        from guard_eval_harness.guards import list_guards
        payload = {"guards": list_guards()}
    elif args.what == "backends":
        from guard_eval_harness.backends import list_backends
        payload = {"backends": list_backends()}
    elif args.what == "policies":
        from guard_eval_harness.policies import list_policies
        payload = {"policies": list_policies()}
    elif args.what == "profiles":
        from guard_eval_harness.guards.profiles import list_profiles
        payload = {"profiles": list_profiles()}
    elif args.what == "output_formats":
        from guard_eval_harness.output_formats import list_output_formats
        payload = {"output_formats": list_output_formats()}
    else:
        payload = {"metrics": list(_AVAILABLE_METRICS)}
    _emit_json(payload)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Execute `geh validate --config <yaml>`."""
    config = load_config_from_path(args.config)
    ensure_builtin_registrations()
    validated = []
    for selection in config.datasets:
        from guard_eval_harness.config import ResolvedDatasetConfig

        dataset_cls = dataset_registry.get(selection.adapter)
        dataset_config = ResolvedDatasetConfig(
            name=selection.name,
            adapter=selection.adapter,
            path=selection.path,
            split=selection.split,
            options=dict(selection.options),
        )
        dataset = dataset_cls.from_config(dataset_config)
        samples = dataset.load()
        if not samples:
            raise ValueError(
                f"dataset {selection.name!r} loaded zero samples"
            )
        validated.append({
            "name": selection.name,
            "adapter": selection.adapter,
            "split": selection.split,
            "sample_count": len(samples),
        })
    _emit_json({
        "config": Path(args.config).resolve().as_posix(),
        "datasets": validated,
        "status": "valid",
    })
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Execute `geh inspect --run-dir <dir>`."""
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )

    datasets_dir = run_dir / "datasets"
    dataset_details = []
    if datasets_dir.exists():
        for d in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
            preds = d / "predictions.jsonl"
            metrics = d / "metrics.json"
            dataset_details.append({
                "name": d.name,
                "files": sorted(f.name for f in d.iterdir()),
                "prediction_count": _count_jsonl_rows(preds),
                "prediction_preview": _load_jsonl_preview(preds),
                "metrics": (
                    json.loads(metrics.read_text(encoding="utf-8"))
                    if metrics.exists()
                    else None
                ),
            })
    _emit_json({
        "run_dir": run_dir.as_posix(),
        "manifest": {
            "run_name": manifest.get("run_name"),
            "status": manifest.get("status"),
            "config_hash": manifest.get("config_hash"),
            "dataset_count": len(manifest.get("datasets", [])),
            "totals": manifest.get("totals"),
        },
        "datasets": dataset_details,
        "summary": summary,
    })
    return 0


def _count_jsonl_rows(path: Path) -> int | None:
    """Count lines in a JSONL artifact when it exists."""
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_jsonl_preview(
    path: Path,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Load the first N rows of a JSONL artifact when it exists."""
    if not path.exists():
        return []
    preview: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in itertools.islice(handle, limit):
            stripped = line.strip()
            if stripped:
                preview.append(json.loads(stripped))
    return preview


def _emit_json(payload: object) -> None:
    """Print a JSON payload to stdout in stable form."""
    print(json.dumps(payload, indent=2, sort_keys=True))


_USER_FACING_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    KeyError,
    ValueError,
    HTTPError,
    URLError,
)


def _format_user_error(exc: BaseException) -> str:
    """Format expected user errors without traceback noise."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    if isinstance(exc, HTTPError):
        if exc.code in {401, 403}:
            return (
                f"HTTP {exc.code} from {exc.url}: authentication failed. "
                "Check the configured API key environment variable."
            )
        return f"HTTP {exc.code} from {exc.url}: {exc.reason}"
    if isinstance(exc, URLError):
        return f"connection error: {exc.reason}"
    return str(exc)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("guard_eval_harness").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        os.environ["GEH_DEBUG"] = "1"
    try:
        return int(args.func(args))
    except _USER_FACING_EXCEPTIONS as exc:
        if getattr(args, "debug", False):
            raise
        print(f"Error: {_format_user_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
