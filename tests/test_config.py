"""Tests for resolved config loading."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from guard_eval_harness.config import load_config, load_config_from_path


class ConfigLoadingTest(unittest.TestCase):
    """Validate config resolution and overrides."""

    def test_load_config_resolves_models(self) -> None:
        config = load_config(
            {
                "run_name": "demo",
                "model": {"adapter": "mock"},
                "datasets": [{"name": "demo", "adapter": "local_jsonl"}],
                "output": {"run_dir": "runs/demo"},
            },
            base_dir="/tmp/project",
        )
        self.assertEqual(config.output.run_dir, "/tmp/project/runs/demo")
        self.assertEqual(config.datasets[0].adapter, "local_jsonl")

    def test_rejects_duplicate_dataset_names(self) -> None:
        with self.assertRaisesRegex(Exception, "duplicate dataset name"):
            load_config(
                {
                    "run_name": "dup-test",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {"name": "same", "adapter": "local_jsonl"},
                        {"name": "same", "adapter": "local_jsonl"},
                    ],
                    "output": {"run_dir": "/tmp/dup-test"},
                },
                base_dir="/tmp",
            )

    def test_rejects_dataset_names_that_collide_on_disk(self) -> None:
        with self.assertRaisesRegex(
            Exception, "same artifact directory"
        ):
            load_config(
                {
                    "run_name": "collision-test",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {"name": "foo/bar", "adapter": "local_jsonl"},
                        {"name": "foo__bar", "adapter": "local_jsonl"},
                    ],
                    "output": {"run_dir": "/tmp/collision-test"},
                },
                base_dir="/tmp",
            )

    def test_load_config_from_path_expands_env_and_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("", encoding="utf-8")
            os.environ["GEH_DATASET_PATH"] = dataset_path.as_posix()

            config_path = root / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    run_name: env-demo
                    model:
                      adapter: mock
                    datasets:
                      - name: env_dataset
                        adapter: local_jsonl
                        path: ${GEH_DATASET_PATH}
                    output:
                      run_dir: out/default
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config_from_path(
                config_path,
                output_dir=(root / "runs" / "override").as_posix(),
                threshold=0.7,
                limit=2,
            )

            self.assertEqual(config.threshold, 0.7)
            self.assertEqual(config.execution.limit, 2)
            self.assertEqual(
                config.output.run_dir,
                (root / "runs" / "override").as_posix(),
            )
            self.assertEqual(config.datasets[0].path, dataset_path.as_posix())

    def test_load_config_accepts_auto_batch_resume_and_backoff(self) -> None:
        config = load_config(
            {
                "run_name": "auto-batch",
                "model": {"adapter": "mock"},
                "datasets": [{"name": "demo", "adapter": "local_jsonl"}],
                "output": {"run_dir": "runs/demo"},
                "execution": {
                    "batch_size": "auto",
                    "resume": True,
                    "retry_backoff": 2.5,
                },
            },
            base_dir="/tmp/project",
        )

        self.assertEqual(config.execution.batch_size, "auto")
        self.assertTrue(config.execution.resume)
        self.assertEqual(config.execution.retry_backoff, 2.5)

    def test_load_config_accepts_numeric_string_batch_size(self) -> None:
        config = load_config(
            {
                "run_name": "string-batch-size",
                "model": {"adapter": "mock"},
                "datasets": [{"name": "demo", "adapter": "local_jsonl"}],
                "output": {"run_dir": "runs/demo"},
                "execution": {"batch_size": "4"},
            },
            base_dir="/tmp/project",
        )

        self.assertEqual(config.execution.batch_size, 4)

    def test_rejects_invalid_auto_batch_syntax(self) -> None:
        with self.assertRaisesRegex(Exception, "batch_size"):
            load_config(
                {
                    "run_name": "bad-auto-batch",
                    "model": {"adapter": "mock"},
                    "datasets": [
                        {"name": "demo", "adapter": "local_jsonl"}
                    ],
                    "output": {"run_dir": "/tmp/demo"},
                    "execution": {"batch_size": "auto:4"},
                },
                base_dir="/tmp",
            )


if __name__ == "__main__":
    unittest.main()
