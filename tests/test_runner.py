"""Integration tests for the minimal runner."""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.config import load_config, load_config_from_path
from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.datasets.source_backed import SourceBackedDatasetAdapter
from guard_eval_harness.execution import run_benchmark
from guard_eval_harness.execution.artifacts import sha256_payload
from guard_eval_harness.execution.runner import _resolve_runtime_config
from guard_eval_harness.models.base import ModelAdapter
from guard_eval_harness.registry import dataset_registry, model_registry
from guard_eval_harness.reports import rebuild_summary
from guard_eval_harness.reports.summary import compare_runs
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    NormalizedSample,
    PredictSample,
)


class IntegrityTestAdapter(ModelAdapter):
    """Adapter used to exercise prediction-set validation paths."""

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="integrity_test",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            requires_ground_truth=True,
            notes=("test-only",),
        )

    @property
    def allow_partial_predictions(self) -> bool:
        return bool(self.config.args.get("allow_partial_predictions", False))

    def predict_batch(
        self,
        samples: list[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        strategy = self.config.args.get("strategy")
        predictions = [
            NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=0.9 if sample.label.unsafe else 0.1,
                unsafe_label=sample.label.unsafe,
                threshold=threshold,
                latency_ms=0.0,
                metadata={"adapter": "integrity_test"},
            )
            for sample in samples
        ]
        if strategy == "drop_last":
            return predictions[:-1]
        if strategy == "duplicate_first" and predictions:
            return [predictions[0], predictions[0], *predictions[1:]]
        if strategy == "unknown_id" and predictions:
            first = predictions[0].model_copy(
                update={"sample_id": "unknown-sample"}
            )
            return [first, *predictions[1:]]
        return predictions


class LimitAwareDataset(DatasetAdapter):
    """Test-only dataset that records execution limit propagation."""

    observed_limit: int | None = None

    def load(self) -> list[NormalizedSample]:
        self.__class__.observed_limit = self.config.options.get("limit")
        return [
            NormalizedSample(
                id="limit-aware-1",
                dataset=self.config.name,
                split=self.config.split,
                messages=[{"role": "user", "content": "Hello"}],
                label={"unsafe": False},
            )
        ]


class EmptyDataset(DatasetAdapter):
    """Test-only dataset that simulates a dataset load failure mode."""

    def load(self) -> list[NormalizedSample]:
        return []


class CodeMetricEligibleDataset(DatasetAdapter):
    """Test-only dataset that opts into code-vuln metrics."""

    def load(self) -> list[NormalizedSample]:
        return [
            NormalizedSample(
                id="code-metric-1",
                dataset=self.config.name,
                split=self.config.split,
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-79",),
            ),
            NormalizedSample(
                id="code-metric-2",
                dataset=self.config.name,
                split=self.config.split,
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-79",),
            ),
        ]

    def describe(self, samples):
        meta = super().describe(samples)
        eligibility = dict(meta.metric_eligibility)
        eligibility["code_vuln"] = True
        return meta.model_copy(
            update={
                "input_modalities": ("code",),
                "metric_eligibility": eligibility,
            }
        )


class BuiltinPredictMetadataDataset(SourceBackedDatasetAdapter):
    """Source-backed dataset with built-in model-visible metadata."""

    metadata_fields_to_preserve = ("target_role",)

    def load_source_rows(self):
        return [
            {
                "id": "builtin-predict-metadata-1",
                "prompt": "hello",
                "unsafe": False,
                "target_role": "assistant",
            }
        ]


class ExecutionEngineTestAdapter(ModelAdapter):
    """Adapter used to exercise resume and auto-batch behavior."""

    observed_batches: list[list[str]] = []
    batch_call_count: int = 0
    fail_on_call_number: int | None = None

    @classmethod
    def reset(cls) -> None:
        """Clear the observed batch history."""
        cls.observed_batches = []
        cls.batch_call_count = 0
        cls.fail_on_call_number = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="execution_engine_test",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            requires_ground_truth=True,
            notes=("test-only",),
        )

    def predict_batch(
        self,
        samples: list[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        self.__class__.batch_call_count += 1
        self.__class__.observed_batches.append(
            [sample.id for sample in samples]
        )
        if (
            self.__class__.fail_on_call_number
            == self.__class__.batch_call_count
        ):
            raise RuntimeError("synthetic adapter failure")
        fail_above_batch_size = self.config.args.get("fail_above_batch_size")
        if (
            isinstance(fail_above_batch_size, int)
            and len(samples) > fail_above_batch_size
        ):
            raise RuntimeError(
                str(
                    self.config.args.get(
                        "fail_message",
                        "CUDA out of memory",
                    )
                )
            )
        return [
            NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=0.9 if sample.label.unsafe else 0.1,
                unsafe_label=sample.label.unsafe,
                threshold=threshold,
                latency_ms=0.0,
                metadata={"adapter": "execution_engine_test"},
            )
            for sample in samples
        ]


class CategoryEchoAdapter(ModelAdapter):
    """Test-only adapter that emits code-vuln categories."""

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="category_echo",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            supported_input_modalities=("text", "code"),
            supports_category_outputs=True,
            notes=("test-only",),
        )

    @property
    def allow_partial_predictions(self) -> bool:
        return bool(self.config.args.get("allow_partial_predictions", False))

    def predict_batch(
        self,
        samples: list[NormalizedSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        predictions = [
            NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=0.9,
                unsafe_label=True,
                threshold=threshold,
                latency_ms=0.0,
                predicted_categories=("CWE-79",),
                metadata={"adapter": "category_echo"},
            )
            for sample in samples
        ]
        if self.config.args.get("drop_last_prediction", False):
            return predictions[:-1]
        return predictions


class PredictViewObserverAdapter(ModelAdapter):
    """Test-only production-style adapter that records predict samples."""

    observed_samples: list[PredictSample] = []

    @classmethod
    def reset(cls) -> None:
        """Clear observed samples."""
        cls.observed_samples = []

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="predict_view_observer",
            probability_scores=True,
            batching=True,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            notes=("test-only",),
        )

    def predict_batch(
        self,
        samples: list[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        self.__class__.observed_samples.extend(samples)
        return [
            NormalizedPrediction(
                sample_id=sample.id,
                unsafe_score=(
                    0.9
                    if "unsafe" in sample.messages[0].text_content
                    else 0.1
                ),
                unsafe_label=(
                    0.9
                    if "unsafe" in sample.messages[0].text_content
                    else 0.1
                )
                >= threshold,
                threshold=threshold,
                latency_ms=0.0,
                metadata={"adapter": "predict_view_observer"},
            )
            for sample in samples
        ]


if "integrity_test" not in model_registry:
    model_registry.register("integrity_test", target=IntegrityTestAdapter)

if "limit_aware" not in dataset_registry:
    dataset_registry.register("limit_aware", target=LimitAwareDataset)

if "empty_dataset" not in dataset_registry:
    dataset_registry.register("empty_dataset", target=EmptyDataset)

if "code_metric_eligible" not in dataset_registry:
    dataset_registry.register(
        "code_metric_eligible",
        target=CodeMetricEligibleDataset,
    )

if "builtin_predict_metadata" not in dataset_registry:
    dataset_registry.register(
        "builtin_predict_metadata",
        target=BuiltinPredictMetadataDataset,
    )

if "execution_engine_test" not in model_registry:
    model_registry.register(
        "execution_engine_test",
        target=ExecutionEngineTestAdapter,
    )

if "category_echo" not in model_registry:
    model_registry.register(
        "category_echo",
        target=CategoryEchoAdapter,
    )

if "predict_view_observer" not in model_registry:
    model_registry.register(
        "predict_view_observer",
        target=PredictViewObserverAdapter,
    )


class RunnerIntegrationTest(unittest.TestCase):
    """Validate artifact layout and deterministic mock execution."""

    def _write_config(
        self,
        root: Path,
        *,
        model_block: str,
        dataset_path: Path | None = None,
        batch_size: int = 10,
    ) -> Path:
        if dataset_path is None:
            dataset_path = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "datasets"
                / "mock_samples.jsonl"
            )
        config_path = root / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""
                version: 1
                run_name: runner-test
                threshold: 0.5
                model:
                {textwrap.indent(model_block.strip(), "  ")}
                datasets:
                  - name: mock_jsonl
                    adapter: local_jsonl
                    path: {dataset_path.as_posix()}
                    split: test
                output:
                  run_dir: {(root / "run").as_posix()}
                execution:
                  batch_size: {batch_size}
                  concurrency: 1
                  retries: 0
                """
            ).strip(),
            encoding="utf-8",
        )
        return config_path

    def test_mock_run_writes_expected_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config_from_path(
                config_path,
                output_dir=output_dir.as_posix(),
            )
            result = run_benchmark(config)

            self.assertEqual(result.run_dir, output_dir.as_posix())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "resolved-config.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "report.html").exists())
            self.assertTrue(
                (
                    output_dir / "datasets" / "mock_jsonl" / "predictions.jsonl"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir / "datasets" / "mock_jsonl" / "metrics.json"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "datasets"
                    / "mock_jsonl"
                    / "dataset-manifest.json"
                ).exists()
            )

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["dataset_count"], 1)
            self.assertEqual(summary["datasets"][0]["metrics"]["accuracy"], 1.0)
            self.assertEqual(
                summary["datasets"][0]["evaluated_sample_count"], 16
            )
            self.assertNotIn("note", summary["datasets"][0])

            rebuilt = rebuild_summary(output_dir)
            self.assertEqual(rebuilt["datasets"][0]["name"], "mock_jsonl")
            self.assertTrue((output_dir / "report.html").exists())

    def test_production_adapter_receives_predict_sample_allowlist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "samples.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "hello",
                                "verdict": "safe",
                                "category": "benign",
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "unsafe request",
                                "verdict": "unsafe",
                                "category": "policy",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            PredictViewObserverAdapter.reset()
            config = load_config(
                {
                    "version": 1,
                    "run_name": "predict-view",
                    "model": {"adapter": "predict_view_observer"},
                    "datasets": [
                        {
                            "name": "predict_view_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "id_field": None,
                            "label_field": "verdict",
                            "metadata_fields": ("category", "verdict"),
                            "predict_metadata_fields": ("category",),
                        }
                    ],
                    "output": {"run_dir": (root / "run").as_posix()},
                },
                base_dir=root.as_posix(),
            )

            run_benchmark(config)

        observed = PredictViewObserverAdapter.observed_samples
        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all(isinstance(sample, PredictSample) for sample in observed)
        )
        self.assertTrue(all(not hasattr(sample, "label") for sample in observed))
        self.assertEqual(
            [sample.metadata for sample in observed],
            [{"category": "benign"}, {"category": "policy"}],
        )

    def test_runner_uses_adapter_resolved_predict_metadata_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            PredictViewObserverAdapter.reset()
            config = load_config(
                {
                    "version": 1,
                    "run_name": "builtin-predict-metadata",
                    "model": {"adapter": "predict_view_observer"},
                    "datasets": [
                        {
                            "name": "builtin_predict_metadata",
                            "adapter": "builtin_predict_metadata",
                        }
                    ],
                    "output": {"run_dir": (root / "run").as_posix()},
                },
                base_dir=root.as_posix(),
            )

            run_benchmark(config)

        self.assertEqual(
            [
                sample.metadata
                for sample in PredictViewObserverAdapter.observed_samples
            ],
            [{"target_role": "assistant"}],
        )

    def test_runtime_config_exposes_model_name_to_dataset_options(self) -> None:
        config = load_config(
            {
                "version": 1,
                "run_name": "runtime-model-name",
                "model": {
                    "adapter": "hf",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                },
                "datasets": [
                    {
                        "name": "demo",
                        "adapter": "limit_aware",
                    }
                ],
                "output": {"run_dir": "out/runtime-model-name"},
            },
            base_dir=".",
        )

        resolved = _resolve_runtime_config(config)

        self.assertEqual(
            resolved.datasets[0].options["model_name"],
            "Qwen/Qwen2.5-7B-Instruct",
        )

    def test_runtime_config_preserves_explicit_dataset_model_name(self) -> None:
        config = load_config(
            {
                "version": 1,
                "run_name": "runtime-model-name-explicit",
                "model": {
                    "adapter": "hf",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                },
                "datasets": [
                    {
                        "name": "demo",
                        "adapter": "limit_aware",
                        "options": {"model_name": "custom-model"},
                    }
                ],
                "output": {"run_dir": "out/runtime-model-name-explicit"},
            },
            base_dir=".",
        )

        resolved = _resolve_runtime_config(config)

        self.assertEqual(
            resolved.datasets[0].options["model_name"],
            "custom-model",
        )

    def test_runner_uses_dataset_metric_eligibility_for_code_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_config(
                {
                    "version": 1,
                    "run_name": "code-metric-test",
                    "threshold": 0.5,
                    "model": {"adapter": "category_echo"},
                    "datasets": [
                        {
                            "name": "code_metric_demo",
                            "adapter": "code_metric_eligible",
                        }
                    ],
                    "output": {"run_dir": (root / "run").as_posix()},
                },
                base_dir=root.as_posix(),
            )

            result = run_benchmark(config)

            metrics = json.loads(
                (
                    Path(result.run_dir)
                    / "datasets"
                    / "code_metric_demo"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIn("code_vuln", metrics)
            self.assertEqual(metrics["code_vuln"]["tp"], 2)

    def test_runner_code_vuln_excludes_dropped_predictions(
        self,
    ) -> None:
        """code_vuln metrics should only count evaluated samples.

        Dropped predictions (e.g. from API rate limiting) must not be
        penalized as false negatives — they should be excluded from
        the metric computation entirely.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_config(
                {
                    "version": 1,
                    "run_name": "code-metric-missing",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "category_echo",
                        "args": {
                            "allow_partial_predictions": True,
                            "drop_last_prediction": True,
                        },
                    },
                    "datasets": [
                        {
                            "name": "code_metric_missing",
                            "adapter": "code_metric_eligible",
                        }
                    ],
                    "execution": {"batch_size": 2},
                    "output": {"run_dir": (root / "run").as_posix()},
                },
                base_dir=root.as_posix(),
            )

            result = run_benchmark(config)

            metrics = json.loads(
                (
                    Path(result.run_dir)
                    / "datasets"
                    / "code_metric_missing"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            # Only the 1 evaluated sample should be counted;
            # the dropped sample must not appear as fn.
            self.assertEqual(metrics["code_vuln"]["count"], 1)
            self.assertEqual(metrics["code_vuln"]["tp"], 1)

    def test_run_redacts_sensitive_values_in_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(
                root,
                model_block="""
                adapter: mock
                args:
                  strategy: label_echo
                  api_key: ${GEH_TEST_SECRET}
                  api_key_env: ${GEH_TEST_SECRET}
                  trace_env: GEH_TRACE_HEADER
                  headers:
                    Authorization: Bearer ${GEH_TEST_SECRET}
                    Cookie: session=${GEH_TEST_SECRET}
                    X-Amz-Security-Token: ${GEH_TEST_SECRET}
                    X-Session-Token: ${GEH_TEST_SECRET}
                    X-Trace: keep-me
                """,
            )
            os.environ["GEH_TEST_SECRET"] = "super-secret-token"
            config = load_config_from_path(config_path)
            result = run_benchmark(config)

            resolved_config = json.loads(
                Path(result.run_dir, "resolved-config.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                Path(result.run_dir, "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                resolved_config["model"]["args"]["api_key"],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["api_key_env"],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["trace_env"],
                "GEH_TRACE_HEADER",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["headers"]["Authorization"],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["headers"]["Cookie"],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["headers"][
                    "X-Amz-Security-Token"
                ],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["headers"]["X-Session-Token"],
                "***REDACTED***",
            )
            self.assertEqual(
                resolved_config["model"]["args"]["headers"]["X-Trace"],
                "keep-me",
            )
            self.assertEqual(
                manifest["model"]["args"]["api_key"],
                "***REDACTED***",
            )
            self.assertEqual(
                manifest["resolved_config_sha256"],
                sha256_payload(resolved_config),
            )
            self.assertNotIn(
                "super-secret-token",
                json.dumps(resolved_config, sort_keys=True),
            )
            self.assertNotIn(
                "super-secret-token",
                json.dumps(manifest, sort_keys=True),
            )

    def test_run_fails_when_predictions_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(
                Path(tmpdir),
                model_block="""
                adapter: integrity_test
                args:
                  strategy: drop_last
                """,
            )
            config = load_config_from_path(config_path)

            with self.assertRaisesRegex(
                ValueError,
                "missing prediction sample ids",
            ):
                run_benchmark(config)

    def test_run_marks_partial_when_adapter_allows_missing_predictions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(
                root,
                model_block="""
                adapter: integrity_test
                args:
                  strategy: drop_last
                  allow_partial_predictions: true
                """,
            )
            config = load_config_from_path(config_path)
            result = run_benchmark(config)

            summary = json.loads(
                Path(result.run_dir, "summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                Path(result.run_dir, "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            dataset_manifest = json.loads(
                Path(
                    result.run_dir,
                    "datasets",
                    "mock_jsonl",
                    "dataset-manifest.json",
                ).read_text(encoding="utf-8")
            )
            prediction_count = sum(
                1
                for line in Path(
                    result.run_dir,
                    "datasets",
                    "mock_jsonl",
                    "predictions.jsonl",
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(summary["datasets"][0]["sample_count"], 16)
            self.assertEqual(summary["datasets"][0]["metrics"]["count"], 14)
            self.assertEqual(prediction_count, 14)
            self.assertEqual(
                dataset_manifest["metadata"]["dropped_sample_count"], 2
            )
            self.assertEqual(
                dataset_manifest["metadata"]["evaluation_judgment"],
                "flagged_partial",
            )
            self.assertEqual(
                dataset_manifest["metadata"]["evaluated_sample_count"],
                14,
            )
            self.assertEqual(
                manifest["warnings"],
                [
                    "dropped 2 of 16 samples (12.50%) from dataset "
                    "mock_jsonl after adapter errors"
                ],
            )
            self.assertEqual(
                summary["datasets"][0]["evaluated_sample_count"], 14
            )
            self.assertIn("note", summary["datasets"][0])
            self.assertEqual(
                summary["datasets"][0]["evaluation_judgment"],
                "flagged_partial",
            )
            self.assertIn("12.5%", summary["datasets"][0]["note"])

    def test_run_marks_acceptable_partial_below_one_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "large_mock.jsonl"
            rows = []
            for index in range(200):
                rows.append(
                    json.dumps(
                        {
                            "id": f"sample-{index:03d}",
                            "prompt": f"Prompt {index}",
                            "unsafe": bool(index % 2),
                        }
                    )
                )
            dataset_path.write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            config_path = self._write_config(
                root,
                model_block="""
                adapter: integrity_test
                args:
                  strategy: drop_last
                  allow_partial_predictions: true
                """,
                dataset_path=dataset_path,
                batch_size=200,
            )
            config = load_config_from_path(config_path)
            result = run_benchmark(config)

            summary = json.loads(
                Path(result.run_dir, "summary.json").read_text(encoding="utf-8")
            )
            dataset_manifest = json.loads(
                Path(
                    result.run_dir,
                    "datasets",
                    "mock_jsonl",
                    "dataset-manifest.json",
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["datasets"][0]["sample_count"], 200)
            self.assertEqual(summary["datasets"][0]["metrics"]["count"], 199)
            self.assertEqual(
                dataset_manifest["metadata"]["dropped_sample_count"],
                1,
            )
            self.assertEqual(
                dataset_manifest["metadata"]["evaluation_judgment"],
                "acceptable_partial",
            )
            self.assertEqual(
                summary["datasets"][0]["evaluation_judgment"],
                "acceptable_partial",
            )
            self.assertIn("0.5%", summary["datasets"][0]["note"])

    def test_run_fails_when_prediction_ids_are_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(
                Path(tmpdir),
                model_block="""
                adapter: integrity_test
                args:
                  strategy: duplicate_first
                """,
            )
            config = load_config_from_path(config_path)

            with self.assertRaisesRegex(
                ValueError,
                "duplicate prediction sample ids",
            ):
                run_benchmark(config)

    def test_run_fails_when_prediction_ids_are_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(
                Path(tmpdir),
                model_block="""
                adapter: integrity_test
                args:
                  strategy: unknown_id
                """,
            )
            config = load_config_from_path(config_path)

            with self.assertRaisesRegex(
                ValueError,
                "unknown prediction sample ids",
            ):
                run_benchmark(config)

    def test_run_fails_when_dataset_loads_zero_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(
                {
                    "version": 1,
                    "run_name": "empty-dataset",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {
                            "name": "empty_dataset",
                            "adapter": "empty_dataset",
                        }
                    ],
                    "output": {"run_dir": str(Path(tmpdir) / "run")},
                },
                base_dir=tmpdir,
            )

            with self.assertRaisesRegex(
                ValueError,
                "dataset empty_dataset loaded zero samples",
            ):
                run_benchmark(config)

    def test_runner_passes_explicit_dataset_limit_into_dataset_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(
                {
                    "version": 1,
                    "run_name": "limit-aware",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {
                            "name": "limit-aware",
                            "adapter": "limit_aware",
                            "options": {"limit": 2},
                        }
                    ],
                    "output": {"run_dir": str(Path(tmpdir) / "run")},
                },
                base_dir=tmpdir,
            )

            LimitAwareDataset.observed_limit = None
            run_benchmark(config)

            self.assertEqual(LimitAwareDataset.observed_limit, 2)

    def test_compare_runs_does_not_mutate_run_directories(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = Path(tmpdir) / "run_a"
            dir_b = Path(tmpdir) / "run_b"
            config_a = load_config_from_path(
                config_path, output_dir=dir_a.as_posix()
            )
            config_b = load_config_from_path(
                config_path, output_dir=dir_b.as_posix()
            )
            run_benchmark(config_a)
            run_benchmark(config_b)

            summary_a_before = (dir_a / "summary.json").read_bytes()
            report_a_before = (dir_a / "report.html").read_bytes()
            summary_b_before = (dir_b / "summary.json").read_bytes()
            report_b_before = (dir_b / "report.html").read_bytes()

            compare_runs(dir_a, dir_b)

            self.assertEqual(
                (dir_a / "summary.json").read_bytes(), summary_a_before
            )
            self.assertEqual(
                (dir_a / "report.html").read_bytes(), report_a_before
            )
            self.assertEqual(
                (dir_b / "summary.json").read_bytes(), summary_b_before
            )
            self.assertEqual(
                (dir_b / "report.html").read_bytes(), report_b_before
            )

    def test_overwrite_preserves_prior_artifacts_on_failure(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config_from_path(
                config_path, output_dir=output_dir.as_posix()
            )
            run_benchmark(config)

            prior_metrics = (
                output_dir / "datasets" / "mock_jsonl" / "metrics.json"
            ).read_bytes()

            failing_config = load_config(
                {
                    "version": 1,
                    "run_name": "failing-run",
                    "model": {
                        "adapter": "integrity_test",
                        "args": {"strategy": "drop_last"},
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": (
                                root
                                / "examples"
                                / "datasets"
                                / "mock_samples.jsonl"
                            ).as_posix(),
                        }
                    ],
                    "output": {
                        "run_dir": output_dir.as_posix(),
                        "overwrite": True,
                    },
                    "execution": {"batch_size": 10},
                },
                base_dir=tmpdir,
            )
            with self.assertRaises(ValueError):
                run_benchmark(failing_config)

            self.assertTrue((output_dir / "datasets" / "mock_jsonl").exists())
            self.assertEqual(
                (
                    output_dir / "datasets" / "mock_jsonl" / "metrics.json"
                ).read_bytes(),
                prior_metrics,
            )
            resolved_config = json.loads(
                (output_dir / "resolved-config.json").read_text(
                    encoding="utf-8"
                )
            )
            resume_signature = json.loads(
                (output_dir / "resume-signature.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resolved_config["run_name"], "failing-run")
            self.assertEqual(
                resolved_config["model"]["adapter"],
                "integrity_test",
            )
            self.assertEqual(
                resume_signature["model"]["adapter"],
                "integrity_test",
            )

    def test_mock_run_with_batch_size_greater_than_one(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config(
                {
                    "version": 1,
                    "run_name": "batch-smoke",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "mock",
                        "args": {
                            "strategy": "label_echo",
                            "safe_score": 0.1,
                            "unsafe_score": 0.9,
                        },
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "split": "test",
                        }
                    ],
                    "output": {"run_dir": output_dir.as_posix()},
                    "execution": {
                        "batch_size": 16,
                        "concurrency": 1,
                        "retries": 0,
                    },
                },
                base_dir=root.as_posix(),
            )
            run_benchmark(config)

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["datasets"][0]["metrics"]["accuracy"], 1.0)
            self.assertEqual(
                summary["datasets"][0]["evaluated_sample_count"], 16
            )

            predictions_path = (
                output_dir / "datasets" / "mock_jsonl" / "predictions.jsonl"
            )
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            self.assertEqual(len(predictions), 16)
            ids = {p["sample_id"] for p in predictions}
            self.assertEqual(len(ids), 16)

    def test_auto_batch_size_reduces_local_batches_on_capacity_error(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config(
                {
                    "version": 1,
                    "run_name": "auto-batch",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "execution_engine_test",
                        "args": {"fail_above_batch_size": 4},
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "split": "test",
                        }
                    ],
                    "output": {"run_dir": output_dir.as_posix()},
                    "execution": {"batch_size": "auto"},
                },
                base_dir=root.as_posix(),
            )

            ExecutionEngineTestAdapter.reset()
            run_benchmark(config)

            self.assertEqual(
                [
                    len(batch)
                    for batch in ExecutionEngineTestAdapter.observed_batches
                ],
                [16, 8, 4, 4, 4, 4],
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["datasets"][0]["metrics"]["accuracy"], 1.0)

    def test_resume_reuses_cached_predictions_from_run_directory(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-test",
                "threshold": 0.5,
                "model": {"adapter": "execution_engine_test"},
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            predictions_path = (
                output_dir / "datasets" / "mock_jsonl" / "predictions.jsonl"
            )
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            cached_predictions = predictions[:12]
            predictions_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in cached_predictions
                ),
                encoding="utf-8",
            )

            resume_payload = dict(base_payload)
            resume_payload["execution"] = {"batch_size": 8, "resume": True}
            ExecutionEngineTestAdapter.reset()
            result = run_benchmark(
                load_config(resume_payload, base_dir=root.as_posix())
            )

            self.assertEqual(
                ExecutionEngineTestAdapter.observed_batches,
                [[row["sample_id"] for row in predictions[12:]]],
            )
            dataset_manifest = json.loads(
                Path(
                    result.run_dir,
                    "datasets",
                    "mock_jsonl",
                    "dataset-manifest.json",
                ).read_text(encoding="utf-8")
            )
            summary = json.loads(
                Path(result.run_dir, "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                dataset_manifest["metadata"]["cached_prediction_count"],
                12,
            )
            self.assertEqual(
                dataset_manifest["metadata"]["executed_sample_count"],
                4,
            )
            self.assertTrue(dataset_manifest["metadata"]["resume_enabled"])
            self.assertIn(
                "resume_signature_sha256",
                dataset_manifest["metadata"],
            )
            self.assertEqual(
                summary["datasets"][0]["metrics"]["count"],
                16,
            )

    def test_resume_works_after_interrupted_multi_dataset_run(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            payload = {
                "version": 1,
                "run_name": "resume-interrupted",
                "threshold": 0.5,
                "model": {"adapter": "execution_engine_test"},
                "datasets": [
                    {
                        "name": "mock_jsonl_a",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    },
                    {
                        "name": "mock_jsonl_b",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    },
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 16},
            }

            ExecutionEngineTestAdapter.reset()
            ExecutionEngineTestAdapter.fail_on_call_number = 2
            with self.assertRaisesRegex(
                RuntimeError, "synthetic adapter failure"
            ):
                run_benchmark(load_config(payload, base_dir=root.as_posix()))

            self.assertTrue((output_dir / "resolved-config.json").exists())
            self.assertTrue(
                (
                    output_dir
                    / "datasets"
                    / "mock_jsonl_a"
                    / "predictions.jsonl"
                ).exists()
            )

            resume_payload = dict(payload)
            resume_payload["execution"] = {
                "batch_size": 16,
                "resume": True,
            }
            ExecutionEngineTestAdapter.reset()
            run_benchmark(load_config(resume_payload, base_dir=root.as_posix()))

            self.assertEqual(
                len(ExecutionEngineTestAdapter.observed_batches), 1
            )
            self.assertEqual(
                len(ExecutionEngineTestAdapter.observed_batches[0]), 16
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["dataset_count"], 2)

    def test_resume_cleans_stale_dataset_directories_after_success(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-clean-stale",
                "threshold": 0.5,
                "model": {"adapter": "execution_engine_test"},
                "datasets": [
                    {
                        "name": "mock_jsonl_a",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    },
                    {
                        "name": "mock_jsonl_b",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    },
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 16},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            resume_payload = {
                **base_payload,
                "datasets": [base_payload["datasets"][0]],
                "execution": {"batch_size": 16, "resume": True},
            }
            run_benchmark(load_config(resume_payload, base_dir=root.as_posix()))

            self.assertTrue((output_dir / "datasets" / "mock_jsonl_a").exists())
            self.assertFalse(
                (output_dir / "datasets" / "mock_jsonl_b").exists()
            )

    def test_resume_rejects_threshold_mismatch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-threshold",
                "threshold": 0.5,
                "model": {"adapter": "execution_engine_test"},
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            resume_payload = dict(base_payload)
            resume_payload["threshold"] = 0.7
            resume_payload["execution"] = {"batch_size": 8, "resume": True}

            with self.assertRaisesRegex(
                ValueError,
                "cannot resume dataset mock_jsonl",
            ):
                run_benchmark(
                    load_config(resume_payload, base_dir=root.as_posix())
                )

    def test_resume_mismatch_with_overwrite_preserves_prior_signatures(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-overwrite-mismatch",
                "threshold": 0.5,
                "model": {"adapter": "mock"},
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            resolved_before = (output_dir / "resolved-config.json").read_bytes()
            signature_before = (
                output_dir / "resume-signature.json"
            ).read_bytes()

            resume_payload = dict(base_payload)
            resume_payload["threshold"] = 0.7
            resume_payload["output"] = {
                "run_dir": output_dir.as_posix(),
                "overwrite": True,
            }
            resume_payload["execution"] = {
                "batch_size": 8,
                "resume": True,
            }

            with self.assertRaisesRegex(
                ValueError,
                "cannot resume dataset mock_jsonl",
            ):
                run_benchmark(
                    load_config(resume_payload, base_dir=root.as_posix())
                )

            self.assertEqual(
                (output_dir / "resolved-config.json").read_bytes(),
                resolved_before,
            )
            self.assertEqual(
                (output_dir / "resume-signature.json").read_bytes(),
                signature_before,
            )

    def test_resume_rejects_stale_cached_predictions_when_dataset_manifest_differs(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-stale-cache",
                "threshold": 0.5,
                "model": {"adapter": "execution_engine_test"},
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            dataset_manifest_path = (
                output_dir / "datasets" / "mock_jsonl" / "dataset-manifest.json"
            )
            dataset_manifest = json.loads(
                dataset_manifest_path.read_text(encoding="utf-8")
            )
            dataset_manifest["metadata"]["resume_signature_sha256"] = (
                "sha256:stale"
            )
            dataset_manifest_path.write_text(
                json.dumps(dataset_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            resume_payload = dict(base_payload)
            resume_payload["execution"] = {"batch_size": 8, "resume": True}

            with self.assertRaisesRegex(
                ValueError,
                "cached predictions were produced by a different",
            ):
                run_benchmark(
                    load_config(resume_payload, base_dir=root.as_posix())
                )

    def test_resume_rejects_non_remote_model_arg_changes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-model-args",
                "threshold": 0.5,
                "model": {
                    "adapter": "execution_engine_test",
                    "args": {"concurrency": 1},
                },
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            resume_payload = dict(base_payload)
            resume_payload["model"] = {
                "adapter": "execution_engine_test",
                "args": {"concurrency": 2},
            }
            resume_payload["execution"] = {"batch_size": 8, "resume": True}

            with self.assertRaisesRegex(
                ValueError,
                "cannot resume dataset mock_jsonl",
            ):
                run_benchmark(
                    load_config(resume_payload, base_dir=root.as_posix())
                )

    def test_resume_rejects_changed_secret_like_model_args(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            base_payload = {
                "version": 1,
                "run_name": "resume-secret-args",
                "threshold": 0.5,
                "model": {
                    "adapter": "mock",
                    "args": {
                        "strategy": "label_echo",
                        "api_key": "tenant-a",
                    },
                },
                "datasets": [
                    {
                        "name": "mock_jsonl",
                        "adapter": "local_jsonl",
                        "path": dataset_path.as_posix(),
                        "split": "test",
                    }
                ],
                "output": {"run_dir": output_dir.as_posix()},
                "execution": {"batch_size": 8},
            }
            run_benchmark(load_config(base_payload, base_dir=root.as_posix()))

            resume_payload = dict(base_payload)
            resume_payload["model"] = {
                "adapter": "mock",
                "args": {
                    "strategy": "label_echo",
                    "api_key": "tenant-b",
                },
            }
            resume_payload["execution"] = {"batch_size": 8, "resume": True}

            with self.assertRaisesRegex(
                ValueError,
                "cannot resume dataset mock_jsonl",
            ):
                run_benchmark(
                    load_config(resume_payload, base_dir=root.as_posix())
                )

    def test_runner_injects_execution_defaults_into_http_model_args(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config(
                {
                    "version": 1,
                    "run_name": "http-defaults",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "http",
                        "args": {
                            "url": "https://example.test/score",
                            "score_path": "unsafe_score",
                        },
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "split": "test",
                        }
                    ],
                    "output": {"run_dir": output_dir.as_posix()},
                    "execution": {
                        "batch_size": 2,
                        "concurrency": 4,
                        "retries": 3,
                        "retry_backoff": 2.5,
                        "limit": 2,
                    },
                },
                base_dir=root.as_posix(),
            )

            with patch(
                "guard_eval_harness.models.http.json_post_with_retry",
                return_value={"unsafe_score": 0.1},
            ) as mock_post:
                run_benchmark(config)

            resolved_config = json.loads(
                (output_dir / "resolved-config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(mock_post.call_count, 2)
            self.assertEqual(mock_post.call_args.kwargs["retries"], 3)
            self.assertEqual(mock_post.call_args.kwargs["backoff"], 2.5)
            self.assertEqual(
                resolved_config["model"]["args"]["concurrency"],
                4,
            )
            self.assertEqual(
                resolved_config["model"]["args"]["retries"],
                3,
            )
            self.assertEqual(
                resolved_config["model"]["args"]["retry_backoff"],
                2.5,
            )

    def test_runner_injects_execution_defaults_into_openai_moderation_args(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config(
                {
                    "version": 1,
                    "run_name": "openai-moderation-defaults",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "openai_moderation",
                        "args": {
                            "url": "https://example.test/v1/moderations",
                        },
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "split": "test",
                        }
                    ],
                    "output": {"run_dir": output_dir.as_posix()},
                    "execution": {
                        "batch_size": 2,
                        "concurrency": 4,
                        "retries": 3,
                        "retry_backoff": 2.5,
                        "limit": 2,
                    },
                },
                base_dir=root.as_posix(),
            )

            with patch(
                "guard_eval_harness.models.openai_moderation.json_post_with_retry",
                return_value={
                    "results": [
                        {
                            "flagged": False,
                            "categories": {},
                            "category_scores": {},
                        }
                    ]
                },
            ) as mock_post:
                run_benchmark(config)

            resolved_config = json.loads(
                (output_dir / "resolved-config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(mock_post.call_count, 2)
            self.assertEqual(mock_post.call_args.kwargs["retries"], 3)
            self.assertEqual(mock_post.call_args.kwargs["backoff"], 2.5)
            self.assertEqual(
                resolved_config["model"]["args"]["concurrency"],
                4,
            )
            self.assertEqual(
                resolved_config["model"]["args"]["retries"],
                3,
            )
            self.assertEqual(
                resolved_config["model"]["args"]["retry_backoff"],
                2.5,
            )

    def test_auto_batch_keeps_remote_batches_intact(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / "examples" / "datasets" / "mock_samples.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config(
                {
                    "version": 1,
                    "run_name": "http-auto-batch",
                    "threshold": 0.5,
                    "model": {
                        "adapter": "http",
                        "args": {
                            "url": "https://example.test/score",
                            "score_path": "unsafe_score",
                        },
                    },
                    "datasets": [
                        {
                            "name": "mock_jsonl",
                            "adapter": "local_jsonl",
                            "path": dataset_path.as_posix(),
                            "split": "test",
                        }
                    ],
                    "output": {"run_dir": output_dir.as_posix()},
                    "execution": {
                        "batch_size": "auto",
                        "concurrency": 4,
                        "limit": 4,
                    },
                },
                base_dir=root.as_posix(),
            )

            observed_batch_sizes: list[int] = []

            def fake_predict_batch(_self, samples, *, threshold):
                observed_batch_sizes.append(len(samples))
                return [
                    NormalizedPrediction(
                        sample_id=sample.id,
                        unsafe_score=0.1,
                        unsafe_label=False,
                        threshold=threshold,
                        latency_ms=0.0,
                    )
                    for sample in samples
                ]

            with patch(
                "guard_eval_harness.models.http.HttpAdapter.predict_batch",
                autospec=True,
                side_effect=fake_predict_batch,
            ):
                run_benchmark(config)

            self.assertEqual(observed_batch_sizes, [4])

    def test_runner_emits_progress_events_when_callback_provided(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            config = load_config_from_path(
                config_path,
                output_dir=output_dir.as_posix(),
            )
            events: list[dict[str, object]] = []

            run_benchmark(config, progress_callback=events.append)

            event_names = [str(event["event"]) for event in events]
            self.assertEqual(event_names[0], "dataset_load_started")
            self.assertIn("dataset_load_completed", event_names)
            self.assertIn("prediction_started", event_names)
            self.assertIn("prediction_progress", event_names)
            self.assertIn("prediction_completed", event_names)
            self.assertEqual(event_names[-1], "dataset_completed")

            dataset_loaded = next(
                event
                for event in events
                if event["event"] == "dataset_load_completed"
            )
            self.assertEqual(dataset_loaded["dataset_name"], "mock_jsonl")
            self.assertEqual(dataset_loaded["sample_count"], 16)

            prediction_started = next(
                event
                for event in events
                if event["event"] == "prediction_started"
            )
            self.assertEqual(prediction_started["batch_size"], 8)
            self.assertEqual(prediction_started["total_batches"], 2)

    def test_runner_does_not_inject_execution_limit_into_dataset_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(
                {
                    "version": 1,
                    "run_name": "limit-aware",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {"name": "limit-aware", "adapter": "limit_aware"}
                    ],
                    "execution": {"limit": 2},
                    "output": {"run_dir": str(Path(tmpdir) / "run")},
                },
                base_dir=tmpdir,
            )
            dataset_config = config.datasets[0]

            self.assertEqual(dataset_config.options, {})


if __name__ == "__main__":
    unittest.main()
