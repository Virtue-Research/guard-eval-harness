"""Focused CLI command tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


class CliCommandTest(unittest.TestCase):
    """Validate command output and artifact-driven flows."""

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = (self.root / "src").as_posix()

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "guard_eval_harness", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
            env=self.env,
        )

    def test_list_metrics_and_presets(self) -> None:
        metrics = json.loads(self._run_cli("list", "metrics").stdout)
        presets = json.loads(self._run_cli("list", "presets").stdout)

        self.assertIn("accuracy", metrics["metrics"])
        self.assertIsInstance(presets["presets"], list)
        self.assertGreater(len(presets["presets"]), 0)

    def test_validate_inspect_report_and_export(self) -> None:
        config_path = self.root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            export_path = Path(tmpdir) / "summary.csv"
            xlsx_path = Path(tmpdir) / "summary.xlsx"

            validate = json.loads(
                self._run_cli(
                    "validate",
                    "--config",
                    config_path.as_posix(),
                ).stdout
            )
            self.assertEqual("valid", validate["status"])
            self.assertEqual(1, len(validate["datasets"]))
            self.assertEqual(16, validate["datasets"][0]["sample_count"])

            self._run_cli(
                "run",
                "--config",
                config_path.as_posix(),
                "--output-dir",
                output_dir.as_posix(),
            )

            inspect = json.loads(
                self._run_cli(
                    "inspect",
                    "--run-dir",
                    output_dir.as_posix(),
                ).stdout
            )
            self.assertEqual("mock-jsonl", inspect["manifest"]["run_name"])
            self.assertEqual(1, inspect["manifest"]["dataset_count"])
            self.assertEqual(16, inspect["artifacts"]["dataset_dirs"][0]["prediction_count"])
            self.assertEqual(1.0, inspect["artifacts"]["dataset_dirs"][0]["metrics"]["accuracy"])

            (output_dir / "summary.json").unlink()
            report = json.loads(
                self._run_cli(
                    "report",
                    "--run-dir",
                    output_dir.as_posix(),
                ).stdout
            )
            self.assertEqual(1, report["dataset_count"])
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "report.html").exists())
            report_html = (output_dir / "report.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("mock-jsonl", report_html)
            self.assertIn("Accuracy", report_html)

            export = json.loads(
                self._run_cli(
                    "export",
                    "--run-dir",
                    output_dir.as_posix(),
                    "--format",
                    "csv",
                    "--output",
                    export_path.as_posix(),
                ).stdout
            )
            self.assertEqual("csv", export["format"])
            self.assertTrue(export_path.exists())

            xlsx_export = json.loads(
                self._run_cli(
                    "export",
                    "--run-dir",
                    output_dir.as_posix(),
                    "--format",
                    "xlsx",
                    "--output",
                    xlsx_path.as_posix(),
                ).stdout
            )
            self.assertEqual("xlsx", xlsx_export["format"])
            self.assertTrue(xlsx_path.exists())
            with ZipFile(xlsx_path) as workbook:
                self.assertIn("xl/workbook.xml", workbook.namelist())
                worksheet = workbook.read("xl/worksheets/sheet1.xml").decode(
                    "utf-8"
                )
            self.assertIn("mock_jsonl", worksheet)
            self.assertIn("auroc", worksheet)
            self.assertIn("auprc", worksheet)

    def test_validate_rejects_zero_sample_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "empty.jsonl"
            dataset_path.write_text("", encoding="utf-8")
            config_path = root / "empty-config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "version: 1",
                        "run_name: empty-jsonl",
                        "model:",
                        "  adapter: mock",
                        "datasets:",
                        "  - name: empty_dataset",
                        "    adapter: local_jsonl",
                        f"    path: {dataset_path.as_posix()}",
                        "output:",
                        f"  run_dir: {(root / 'run').as_posix()}",
                    )
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "guard_eval_harness",
                    "validate",
                    "--config",
                    config_path.as_posix(),
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
                env=self.env,
            )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "dataset empty_dataset loaded zero samples",
            result.stderr,
        )

    def test_csv_export_includes_safety_metrics(self) -> None:
        config_path = self.root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            csv_path = Path(tmpdir) / "metrics.csv"

            self._run_cli(
                "run",
                "--config",
                config_path.as_posix(),
                "--output-dir",
                output_dir.as_posix(),
            )
            self._run_cli(
                "export",
                "--run-dir",
                output_dir.as_posix(),
                "--format",
                "csv",
                "--output",
                csv_path.as_posix(),
            )

            content = csv_path.read_text(encoding="utf-8")
            header = content.splitlines()[0]
            for field in ("fpr", "fnr", "tp", "tn", "fp", "fn"):
                self.assertIn(
                    field, header, f"CSV header missing {field}"
                )

    def test_html_report_includes_fpr_fnr(self) -> None:
        config_path = self.root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"

            self._run_cli(
                "run",
                "--config",
                config_path.as_posix(),
                "--output-dir",
                output_dir.as_posix(),
            )

            report_html = (output_dir / "report.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("FPR", report_html)
            self.assertIn("FNR", report_html)

    def test_compare_reports_accuracy_delta(self) -> None:
        config_path = self.root / "examples" / "run-mock-jsonl.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_a = tmp / "config-a.yaml"
            config_b = tmp / "config-b.yaml"
            run_a = tmp / "run-a"
            run_b = tmp / "run-b"

            base_config = config_path.read_text(encoding="utf-8")
            dataset_path = (
                self.root / "examples" / "datasets" / "mock_samples.jsonl"
            ).as_posix()
            config_a.write_text(
                base_config.replace(
                    "run_name: mock-jsonl", "run_name: mock-jsonl-a"
                ).replace(
                    "path: datasets/mock_samples.jsonl",
                    f"path: {dataset_path}",
                ),
                encoding="utf-8",
            )
            config_b.write_text(
                base_config.replace(
                    "run_name: mock-jsonl", "run_name: mock-jsonl-b"
                ).replace("threshold: 0.5", "threshold: 0.95").replace(
                    "path: datasets/mock_samples.jsonl",
                    f"path: {dataset_path}",
                ),
                encoding="utf-8",
            )

            self._run_cli(
                "run",
                "--config",
                config_a.as_posix(),
                "--output-dir",
                run_a.as_posix(),
            )
            self._run_cli(
                "run",
                "--config",
                config_b.as_posix(),
                "--output-dir",
                run_b.as_posix(),
            )

            comparison = json.loads(
                self._run_cli(
                    "compare",
                    "--run-a",
                    run_a.as_posix(),
                    "--run-b",
                    run_b.as_posix(),
                ).stdout
            )

            self.assertEqual("mock-jsonl-a", comparison["run_a"])
            self.assertEqual("mock-jsonl-b", comparison["run_b"])
            self.assertEqual(-0.5, comparison["datasets"][0]["accuracy_delta"])
            self.assertEqual(-1.0, comparison["datasets"][0]["recall_delta"])
            self.assertEqual(0, comparison["datasets"][0]["count_delta"])
            self.assertIsNone(comparison["datasets"][0]["precision_delta"])
            self.assertIsNone(comparison["datasets"][0]["f1_delta"])


if __name__ == "__main__":
    unittest.main()
