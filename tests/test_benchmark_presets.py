"""Tests for package-owned benchmark presets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guard_eval_harness.benchmarks import (
    DATASET_SCOPE_CORE12,
    build_run_payload,
    get_preset,
    list_presets,
    resolve_model_config,
    run_preset,
    summarize_results,
)
from guard_eval_harness.config import load_config_from_path
from guard_eval_harness.execution import RunResult


class BenchmarkPresetTest(unittest.TestCase):
    """Validate preset definitions and execution helpers."""

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_list_presets_exposes_21x31(self) -> None:
        self.assertIn("21x31", list_presets())

    def test_21x31_preset_has_expected_dataset_counts(self) -> None:
        preset = get_preset("21x31")

        self.assertEqual(31, len(preset.datasets))
        self.assertEqual(
            12,
            sum(1 for dataset in preset.datasets if dataset.is_core12),
        )

    def test_resolve_model_config_for_api_model(self) -> None:
        config = resolve_model_config("GPT-4.1-mini", device=-1)

        self.assertEqual("openai_compatible", config["adapter"])
        self.assertEqual(
            "openai/gpt-4.1-mini",
            config["model_name"],
        )
        self.assertEqual(
            "https://openrouter.ai/api",
            config["args"]["root_url"],
        )
        self.assertNotIn("device", config["args"])

    def test_build_run_payload_accepts_batch_size_greater_than_one(
        self,
    ) -> None:
        dataset = get_preset("21x31").datasets[0]

        payload = build_run_payload(
            preset_name="21x31",
            model_name="GPT-4.1-mini",
            dataset=dataset,
            device=-1,
            batch_size=16,
            output_root="out/batch-test",
        )

        self.assertEqual(16, payload["execution"]["batch_size"])

    def test_build_run_payload_uses_preset_output_layout(self) -> None:
        dataset = get_preset("21x31").datasets[0]

        payload = build_run_payload(
            preset_name="21x31",
            model_name="GPT-4.1-mini",
            dataset=dataset,
            device=-1,
            batch_size=1,
            output_root="out/custom-21x31",
            concurrency_override=6,
        )

        self.assertEqual(
            "out/custom-21x31/gpt-4-1-mini/advbench_behaviors",
            payload["output"]["run_dir"],
        )
        self.assertEqual(
            6,
            payload["model"]["args"]["concurrency"],
        )
        self.assertEqual("21x31", payload["metadata"]["preset"])

    def test_run_preset_respects_core12_scope_and_skip_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            existing_dir = output_root / "gpt-4-1-mini" / "advbench_behaviors"
            existing_dir.mkdir(parents=True, exist_ok=True)
            (existing_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "datasets": [
                            {"metrics": {"count": 1, "accuracy": 0.9}}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(config) -> RunResult:
                run_dir = Path(config.output.run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "datasets": [
                                {"metrics": {"count": 2, "accuracy": 0.5}}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return RunResult(
                    run_dir=run_dir.as_posix(),
                    manifest_path=(run_dir / "manifest.json").as_posix(),
                    summary_path=summary_path.as_posix(),
                )

            with patch(
                "guard_eval_harness.benchmarks.presets.run_benchmark",
                side_effect=fake_run,
            ) as mock_run:
                results = run_preset(
                    preset_name="21x31",
                    model_name="GPT-4.1-mini",
                    device=-1,
                    batch_size=1,
                    dataset_scope=DATASET_SCOPE_CORE12,
                    output_root=output_root,
                    base_dir=self.root,
                )

        self.assertEqual(12, len(results))
        self.assertEqual("skipped", results[0].status)
        self.assertEqual("ok", results[1].status)
        self.assertEqual(11, mock_run.call_count)

        summary = summarize_results(results)
        self.assertEqual(1, summary["counts"]["skipped"])
        self.assertEqual(11, summary["counts"]["ok"])
        self.assertEqual(0, summary["counts"]["fail"])

    def test_run_preset_skip_existing_resolves_tilde_and_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home_dir = Path(tmpdir) / "home"
            home_dir.mkdir(parents=True, exist_ok=True)
            output_root = "~/${PRESET_RUN_ROOT}"
            existing_dir = (
                home_dir / "preset-runs" / "gpt-4-1-mini" / "advbench_behaviors"
            )
            existing_dir.mkdir(parents=True, exist_ok=True)
            (existing_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "datasets": [
                            {"metrics": {"count": 1, "accuracy": 0.9}}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(config) -> RunResult:
                run_dir = Path(config.output.run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "datasets": [
                                {"metrics": {"count": 2, "accuracy": 0.5}}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return RunResult(
                    run_dir=run_dir.as_posix(),
                    manifest_path=(run_dir / "manifest.json").as_posix(),
                    summary_path=summary_path.as_posix(),
                )

            with patch.dict(
                "os.environ",
                {
                    "HOME": home_dir.as_posix(),
                    "PRESET_RUN_ROOT": "preset-runs",
                },
                clear=False,
            ):
                with patch(
                    "guard_eval_harness.benchmarks.presets.run_benchmark",
                    side_effect=fake_run,
                ) as mock_run:
                    results = run_preset(
                        preset_name="21x31",
                        model_name="GPT-4.1-mini",
                        device=-1,
                        batch_size=1,
                        dataset_scope=DATASET_SCOPE_CORE12,
                        output_root=output_root,
                        base_dir=self.root,
                    )

        self.assertEqual(12, len(results))
        self.assertEqual("skipped", results[0].status)
        self.assertEqual(existing_dir.as_posix(), results[0].run_dir)
        self.assertEqual(11, mock_run.call_count)

    def test_local_and_api_example_configs_load(self) -> None:
        hf_config = load_config_from_path(
            self.root / "examples" / "run-hf-mock-jsonl.yaml"
        )
        openai_config = load_config_from_path(
            self.root / "examples" / "run-openai-mock-jsonl.yaml"
        )

        self.assertEqual("hf", hf_config.model.adapter)
        self.assertEqual("unitary/toxic-bert", hf_config.model.model_name)
        self.assertEqual("openai_compatible", openai_config.model.adapter)
        self.assertEqual(
            "OPENAI_API_KEY", openai_config.model.args["api_key_env"]
        )


if __name__ == "__main__":
    unittest.main()
