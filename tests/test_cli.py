"""CLI smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliSmokeTest(unittest.TestCase):
    """Validate the minimal CLI shell wiring."""

    def test_list_backends(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = (root / "src").as_posix()

        completed = subprocess.run(
            [sys.executable, "-m", "guard_eval_harness", "list", "backends"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        payload = json.loads(completed.stdout)
        self.assertIn("mock", payload["backends"])

    def test_run_smoke_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "examples" / "run-mock-jsonl.yaml"
        env = os.environ.copy()
        env["PYTHONPATH"] = (root / "src").as_posix()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "guard_eval_harness",
                    "run",
                    "--config",
                    config_path.as_posix(),
                    "--output-dir",
                    output_dir.as_posix(),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["run_dir"], output_dir.as_posix())
            self.assertTrue((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
