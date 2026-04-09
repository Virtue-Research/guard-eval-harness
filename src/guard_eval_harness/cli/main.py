"""Main CLI for the guard eval harness."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

from guard_eval_harness.benchmarks.packs import (
    build_pack_run_payload,
    get_pack,
    list_packs,
)
from guard_eval_harness.benchmarks.presets import list_presets
from guard_eval_harness.config import (
    ResolvedRunConfig,
    load_config,
    load_config_from_path,
)
from guard_eval_harness.execution import run_benchmark
from guard_eval_harness.exports import export_summary
from guard_eval_harness.registry import ensure_builtin_registrations
from guard_eval_harness.registry import dataset_registry, model_registry
from guard_eval_harness.reports import compare_runs, rebuild_summary


AVAILABLE_METRICS = (
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


class HarnessCLI:
    """Main CLI parser and dispatcher."""

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            prog="geh",
            description="Guard Eval Harness",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Quick start:\n"
                "  geh run --dataset xstest --model mock\n"
                "  geh run --dataset xstest,toxic_chat "
                "--model hf --model-name X\n"
                "  geh run --pack core --model mock\n"
                "  geh list datasets\n\n"
                "Common workflow:\n"
                "  geh list datasets\n"
                "  geh run --config examples/run-mock-jsonl.yaml\n"
                "  geh inspect --run-dir out/mock-jsonl\n"
                "  geh report --run-dir out/mock-jsonl\n"
                "  geh export --run-dir out/mock-jsonl "
                "--format csv --output "
                "out/mock-jsonl/summary.csv\n\n"
                "Artifacts are the source of truth. Each run "
                "writes a manifest,\n"
                "a resolved config snapshot, summary JSON, and "
                "one dataset\n"
                "folder per benchmarked dataset."
            ),
        )
        self._parser.set_defaults(func=self._print_help)
        subparsers = self._parser.add_subparsers(
            dest="command", metavar="COMMAND",
        )
        self._build_run(subparsers)
        self._build_compare(subparsers)
        self._build_report(subparsers)
        self._build_list(subparsers)
        self._build_validate(subparsers)
        self._build_inspect(subparsers)
        self._build_export(subparsers)
        self._build_cache(subparsers)

    def _print_help(self, args: argparse.Namespace) -> int:
        self._parser.print_help()
        return 0

    def _build_run(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "run",
            help="Run a benchmark and write artifacts to disk.",
            description=(
                "Execute a benchmark run and persist the resolved "
                "config,\nmanifest, summary, and dataset artifacts "
                "on disk.\n\nUse --config for full YAML configs, "
                "or --pack + --model for\nquick pack-based runs:\n"
                "  geh run --pack core --model mock\n"
                "  geh run --pack jailbreak --model mock"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        source = parser.add_mutually_exclusive_group(
            required=True,
        )
        source.add_argument(
            "--config",
            help="Path to a YAML config.",
        )
        source.add_argument(
            "--pack",
            help=(
                "Named benchmark pack to run "
                "(e.g. core, jailbreak, toxicity, "
                "hate_harassment, prompt_injection)."
            ),
        )
        source.add_argument(
            "--dataset",
            help=(
                "Comma-separated dataset names for "
                "inline runs (e.g. xstest, "
                "xstest,toxic_chat). Requires --model."
            ),
        )
        parser.add_argument(
            "--model",
            dest="model_adapter",
            help=(
                "Model catalog slug "
                "(e.g. llama-guard-3-8b) or raw "
                "adapter name (e.g. hf, vllm, mock)."
            ),
        )
        parser.add_argument(
            "--model-name",
            help=(
                "Model name / HF repo ID for "
                "pack runs."
            ),
        )
        parser.add_argument(
            "--model-args",
            help=(
                "JSON string of extra model adapter "
                "args for pack runs."
            ),
        )
        parser.add_argument(
            "--backend",
            choices=("hf", "vllm"),
            help=(
                "Override the inference backend "
                "(e.g. --model llama-guard-3-8b "
                "--backend vllm)."
            ),
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size for pack runs (default: 1).",
        )
        parser.add_argument(
            "--output-dir",
            help="Override output.run_dir.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Override threshold.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit samples per dataset.",
        )
        parser.add_argument(
            "--no-sample-cache",
            action="store_true",
            default=False,
            help="Disable the sample cache for image datasets.",
        )
        parser.set_defaults(func=self._run)

    def _build_compare(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "compare",
            help="Compare two prior run directories.",
            description=(
                "Compare two existing run directories and "
                "report coarse\nper-dataset metric deltas "
                "from the stored artifacts."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--run-a", required=True,
            help="First run directory.",
        )
        parser.add_argument(
            "--run-b", required=True,
            help="Second run directory.",
        )
        parser.set_defaults(func=self._compare)

    def _build_report(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "report",
            help=(
                "Rebuild summary artifacts from an "
                "existing run directory."
            ),
            description=(
                "Rebuild summary artifacts from the "
                "manifest and dataset\noutputs already on "
                "disk, including a static HTML report."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--run-dir", required=True,
            help="Run directory.",
        )
        parser.set_defaults(func=self._report)

    def _build_list(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "list",
            help=(
                "List built-in datasets, backends, "
                "metrics, packs, presets, or plugins."
            ),
            description=(
                "List discovered objects from the local "
                "registry and the\nbuilt-in metric names "
                "used by the harness."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "what",
            choices=(
                "datasets",
                "backends",
                "metrics",
                "models",
                "packs",
                "presets",
                "plugins",
            ),
            help="What to list.",
        )
        parser.set_defaults(func=self._list)

    def _build_validate(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "validate",
            help=(
                "Validate config loading and local "
                "dataset resolution."
            ),
            description=(
                "Load a config, materialize dataset "
                "adapters, and confirm the\nconfigured "
                "inputs can be normalized successfully."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--config", required=True,
            help="Path to a YAML config.",
        )
        parser.set_defaults(func=self._validate)

    def _build_inspect(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "inspect",
            help=(
                "Inspect manifest and summary artifacts "
                "from a run directory."
            ),
            description=(
                "Inspect an existing run directory, "
                "including the manifest,\nsummary, and "
                "per-dataset artifact layout."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--run-dir", required=True,
            help="Run directory.",
        )
        parser.set_defaults(func=self._inspect)

    def _build_export(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "export",
            help=(
                "Export summary artifacts to a simpler "
                "stakeholder format."
            ),
            description=(
                "Export the rebuilt summary to a JSON, "
                "CSV, or XLSX file."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--run-dir", required=True,
            help="Run directory.",
        )
        parser.add_argument(
            "--format",
            choices=("json", "csv", "xlsx"),
            default="json",
            help="Export format.",
        )
        parser.add_argument(
            "--output", required=True,
            help="Destination file path.",
        )
        parser.set_defaults(func=self._export)

    def _build_cache(
        self, subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "cache",
            help="Manage the sample cache.",
            description=(
                "View or clear the disk cache that stores "
                "normalized samples for image datasets."
            ),
        )
        cache_sub = parser.add_subparsers(
            dest="cache_action", metavar="ACTION",
        )
        clear_p = cache_sub.add_parser(
            "clear",
            help="Remove cached sample entries.",
        )
        clear_p.add_argument(
            "--dataset",
            help="Only clear cache for this adapter name.",
        )
        cache_sub.add_parser(
            "status",
            help="Show cache size and entry count.",
        )
        parser.set_defaults(func=self._cache)

    def parse_args(
        self, argv: Sequence[str] | None = None,
    ) -> argparse.Namespace:
        """Parse CLI arguments."""
        return self._parser.parse_args(argv)

    def execute(
        self, argv: Sequence[str] | None = None,
    ) -> int:
        """Parse args and dispatch a command."""
        args = self.parse_args(argv)
        return int(args.func(args))

    def _run(self, args: argparse.Namespace) -> int:
        if args.pack:
            config = self._resolve_pack_config(args)
        elif args.dataset:
            config = self._resolve_dataset_config(args)
        else:
            config = load_config_from_path(
                args.config,
                output_dir=args.output_dir,
                threshold=args.threshold,
                limit=args.limit,
            )
        if getattr(args, "no_sample_cache", False):
            for dc in config.datasets:
                dc.options["no_sample_cache"] = True
        result = run_benchmark(config)
        print(
            json.dumps(
                {
                    "run_dir": result.run_dir,
                    "manifest_path": result.manifest_path,
                    "summary_path": result.summary_path,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def _resolve_model(
        self,
        args: argparse.Namespace,
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Resolve --model to (adapter, model_name, args).

        Checks the model catalog first; falls back to
        treating --model as a raw adapter name.  User-supplied
        --model-args are merged on top of catalog defaults.
        """
        from guard_eval_harness.models.catalog import (
            resolve_catalog,
        )

        user_args: dict[str, Any] = {}
        if args.model_args:
            user_args = json.loads(args.model_args)

        backend = getattr(args, "backend", None)
        entry = resolve_catalog(
            args.model_adapter, backend=backend,
        )
        if entry is not None:
            merged = dict(entry.args)
            merged.update(user_args)
            return (
                entry.adapter,
                args.model_name or entry.model_name,
                merged,
            )

        return (
            args.model_adapter,
            args.model_name,
            user_args,
        )

    def _resolve_pack_config(
        self,
        args: argparse.Namespace,
    ) -> ResolvedRunConfig:
        """Build a ResolvedRunConfig from --pack flags."""
        if not args.model_adapter:
            self._parser.error(
                "--model is required when using --pack"
            )

        pack = get_pack(args.pack)

        adapter, model_name, model_args = (
            self._resolve_model(args)
        )

        payload = build_pack_run_payload(
            pack=pack,
            model_adapter=adapter,
            model_name=model_name,
            model_args=model_args,
            threshold=(
                args.threshold
                if args.threshold is not None
                else 0.5
            ),
            batch_size=args.batch_size,
            output_root=(
                args.output_dir
                if args.output_dir
                else "out/packs"
            ),
        )

        if args.limit is not None:
            payload["execution"]["limit"] = args.limit

        return load_config(payload)

    def _resolve_dataset_config(
        self,
        args: argparse.Namespace,
    ) -> ResolvedRunConfig:
        """Build a ResolvedRunConfig from --dataset flags."""
        if not args.model_adapter:
            self._parser.error(
                "--model is required when using --dataset"
            )

        dataset_names = [
            name.strip()
            for name in args.dataset.split(",")
            if name.strip()
        ]
        if not dataset_names:
            self._parser.error(
                "--dataset requires at least one name"
            )

        adapter, model_name, model_args = (
            self._resolve_model(args)
        )

        model_slug = (
            model_name or args.model_adapter
        ).replace("/", "_")
        dataset_slug = "-".join(dataset_names[:3])
        if len(dataset_names) > 3:
            dataset_slug += (
                f"-and-{len(dataset_names) - 3}-more"
            )

        threshold = (
            args.threshold
            if args.threshold is not None
            else 0.5
        )
        output_dir = (
            args.output_dir
            or f"out/{model_slug}/{dataset_slug}"
        )

        payload: dict[str, Any] = {
            "version": 1,
            "run_name": (
                f"{model_slug}-{dataset_slug}"
            ),
            "threshold": threshold,
            "model": {
                "adapter": adapter,
                "model_name": model_name,
                "args": model_args,
            },
            "datasets": [
                {"name": name, "adapter": name}
                for name in dataset_names
            ],
            "execution": {
                "batch_size": args.batch_size,
                "concurrency": 1,
                "retries": 0,
            },
            "output": {
                "run_dir": output_dir,
                "overwrite": False,
            },
        }

        if args.limit is not None:
            payload["execution"]["limit"] = args.limit

        return load_config(payload)

    def _compare(self, args: argparse.Namespace) -> int:
        comparison = compare_runs(args.run_a, args.run_b)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0

    def _report(self, args: argparse.Namespace) -> int:
        summary = rebuild_summary(args.run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    def _list(self, args: argparse.Namespace) -> int:
        ensure_builtin_registrations()
        if args.what == "datasets":
            payload = {"datasets": dataset_registry.keys()}
        elif args.what == "backends":
            payload = {"backends": model_registry.keys()}
        elif args.what == "metrics":
            payload = {"metrics": list(AVAILABLE_METRICS)}
        elif args.what == "models":
            from guard_eval_harness.models.catalog import (
                list_catalog,
            )
            payload = {"models": list_catalog()}
        elif args.what == "packs":
            pack_details = []
            for name in list_packs():
                p = get_pack(name)
                pack_details.append(
                    p.as_manifest_dict()
                )
            payload = {"packs": pack_details}
        elif args.what == "presets":
            payload = {"presets": list_presets()}
        else:
            payload = {
                "plugins": {
                    "datasets": dataset_registry.keys(),
                    "models": model_registry.keys(),
                }
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    def _validate(self, args: argparse.Namespace) -> int:
        config = load_config_from_path(args.config)
        ensure_builtin_registrations()
        validated = []
        for dataset_config in config.datasets:
            dataset_cls = dataset_registry.get(
                dataset_config.adapter,
            )
            dataset = dataset_cls.from_config(dataset_config)
            samples = dataset.load()
            if not samples:
                raise ValueError(
                    f"dataset {dataset_config.name} "
                    f"loaded zero samples"
                )
            validated.append(
                {
                    "adapter": dataset_config.adapter,
                    "name": dataset_config.name,
                    "sample_count": len(samples),
                    "split": dataset_config.split,
                }
            )
        print(
            json.dumps(
                {
                    "config": (
                        Path(args.config)
                        .resolve()
                        .as_posix()
                    ),
                    "datasets": validated,
                    "status": "valid",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def _load_jsonl_preview(
        self, path: Path, limit: int = 3,
    ) -> list[dict[str, object]]:
        """Load a small JSONL preview from an artifact."""
        if not path.exists():
            return []

        preview: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in itertools.islice(handle, limit):
                stripped = line.strip()
                if not stripped:
                    continue
                preview.append(json.loads(stripped))
        return preview

    def _count_jsonl_rows(
        self, path: Path,
    ) -> int | None:
        """Count JSONL rows when the artifact exists."""
        if not path.exists():
            return None

        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def _inspect(self, args: argparse.Namespace) -> int:
        run_dir = Path(args.run_dir)
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(
                encoding="utf-8",
            )
        )
        summary = json.loads(
            (run_dir / "summary.json").read_text(
                encoding="utf-8",
            )
        )
        dataset_details = []
        datasets_dir = run_dir / "datasets"
        if datasets_dir.exists():
            for dataset_dir in sorted(
                entry
                for entry in datasets_dir.iterdir()
                if entry.is_dir()
            ):
                predictions_path = (
                    dataset_dir / "predictions.jsonl"
                )
                metrics_path = (
                    dataset_dir / "metrics.json"
                )
                manifest_path = (
                    dataset_dir / "dataset-manifest.json"
                )
                dataset_details.append(
                    {
                        "artifact_dir": dataset_dir.name,
                        "files": sorted(
                            entry.name
                            for entry
                            in dataset_dir.iterdir()
                        ),
                        "metrics": (
                            json.loads(
                                metrics_path.read_text(
                                    encoding="utf-8",
                                )
                            )
                            if metrics_path.exists()
                            else None
                        ),
                        "prediction_count": (
                            self._count_jsonl_rows(
                                predictions_path,
                            )
                        ),
                        "prediction_preview": (
                            self._load_jsonl_preview(
                                predictions_path,
                            )
                        ),
                        "sample_manifest": (
                            json.loads(
                                manifest_path.read_text(
                                    encoding="utf-8",
                                )
                            )
                            if manifest_path.exists()
                            else None
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "manifest": {
                        "dataset_names": [
                            item["name"]
                            for item in manifest["datasets"]
                        ],
                        "run_name": manifest["run_name"],
                        "status": manifest["status"],
                        "dataset_count": len(
                            manifest["datasets"]
                        ),
                    },
                    "artifacts": {
                        "dataset_dirs": dataset_details,
                        "files": sorted(
                            entry.name
                            for entry in run_dir.iterdir()
                        ),
                        "run_dir": run_dir.as_posix(),
                    },
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def _export(self, args: argparse.Namespace) -> int:
        destination = export_summary(
            args.run_dir,
            fmt=args.format,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "format": args.format,
                    "output": destination,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def _cache(self, args: argparse.Namespace) -> int:
        from guard_eval_harness.datasets.sample_cache import (
            clear_sample_cache,
            sample_cache_base_dir,
        )

        action = getattr(args, "cache_action", None)
        if action == "clear":
            removed = clear_sample_cache(
                adapter_name=getattr(args, "dataset", None),
            )
            print(f"Cleared {removed} cache entries.")
            return 0

        if action == "status":
            base = sample_cache_base_dir()
            if not base.exists():
                print("No sample cache found.")
                return 0
            total_bytes = 0
            entry_count = 0
            for p in base.rglob("samples.jsonl"):
                total_bytes += p.stat().st_size
                entry_count += 1
            mb = total_bytes / (1024 * 1024)
            print(
                f"Sample cache: {entry_count} entries, "
                f"{mb:.1f} MB"
            )
            print(f"Location: {base}")
            return 0

        print("Usage: geh cache {clear,status}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
    )
    logging.getLogger("guard_eval_harness").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    return HarnessCLI().execute(argv)
