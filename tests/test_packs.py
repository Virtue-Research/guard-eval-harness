"""Tests for named benchmark packs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from guard_eval_harness.benchmarks.packs import (
    PackManifest,
    build_pack_run_payload,
    get_pack,
    list_pack_versions,
    list_packs,
)
from guard_eval_harness.config import load_config


class PackRegistryTest(unittest.TestCase):
    """Validate pack definitions and lookup."""

    def test_list_packs_returns_canonical_names(
        self,
    ) -> None:
        names = list_packs()
        self.assertEqual(
            sorted(names),
            [
                "core",
                "hate_harassment",
                "jailbreak",
                "prompt_injection",
                "toxicity",
            ],
        )

    def test_get_pack_by_name(self) -> None:
        pack = get_pack("core")
        self.assertIsInstance(pack, PackManifest)
        self.assertEqual("core", pack.name)
        self.assertEqual("v1", pack.version)

    def test_get_pack_by_qualified_alias(self) -> None:
        pack = get_pack("jailbreak-v1")
        self.assertEqual("jailbreak", pack.name)
        self.assertEqual("v1", pack.version)

    def test_get_pack_unknown_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_pack("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_list_pack_versions(self) -> None:
        versions = list_pack_versions("core")
        self.assertEqual(["v1"], versions)

    def test_list_pack_versions_unknown_raises(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            list_pack_versions("nonexistent")


class PackManifestTest(unittest.TestCase):
    """Validate PackManifest contract."""

    def test_qualified_name(self) -> None:
        pack = get_pack("toxicity")
        self.assertEqual("toxicity-v1", pack.qualified_name)

    def test_dataset_names(self) -> None:
        pack = get_pack("core")
        names = pack.dataset_names()
        self.assertEqual(12, len(names))
        self.assertIn("advbench_behaviors", names)
        self.assertIn("xstest", names)

    def test_as_manifest_dict_shape(self) -> None:
        pack = get_pack("core")
        manifest = pack.as_manifest_dict()
        self.assertEqual("core", manifest["name"])
        self.assertEqual("v1", manifest["version"])
        self.assertEqual(12, manifest["dataset_count"])
        self.assertIsInstance(manifest["datasets"], list)
        self.assertIsInstance(
            manifest["description"], str,
        )


class PackDatasetCountsTest(unittest.TestCase):
    """Validate each pack has expected dataset count."""

    def test_core_has_12_datasets(self) -> None:
        self.assertEqual(
            12, len(get_pack("core").datasets),
        )

    def test_jailbreak_has_10_datasets(self) -> None:
        self.assertEqual(
            10, len(get_pack("jailbreak").datasets),
        )

    def test_toxicity_has_7_datasets(self) -> None:
        self.assertEqual(
            7, len(get_pack("toxicity").datasets),
        )

    def test_hate_harassment_has_11_datasets(
        self,
    ) -> None:
        self.assertEqual(
            11,
            len(get_pack("hate_harassment").datasets),
        )

    def test_prompt_injection_has_6_datasets(
        self,
    ) -> None:
        self.assertEqual(
            6,
            len(get_pack("prompt_injection").datasets),
        )

class PackNoDuplicatesTest(unittest.TestCase):
    """Each pack must have unique dataset names."""

    def test_no_duplicate_datasets_in_any_pack(
        self,
    ) -> None:
        for name in list_packs():
            pack = get_pack(name)
            ds_names = pack.dataset_names()
            self.assertEqual(
                len(ds_names),
                len(set(ds_names)),
                f"Pack {name!r} has duplicate datasets",
            )


class BuildPackRunPayloadTest(unittest.TestCase):
    """Validate run payload generation from packs."""

    def test_payload_loads_as_valid_config(self) -> None:
        pack = get_pack("core")
        payload = build_pack_run_payload(
            pack=pack,
            model_adapter="mock",
            model_name=None,
            model_args={"strategy": "label_echo"},
            threshold=0.5,
            batch_size=8,
        )
        config = load_config(payload)
        self.assertEqual(
            "pack-core-mock", config.run_name,
        )
        self.assertEqual("mock", config.model.adapter)
        self.assertEqual(12, len(config.datasets))
        self.assertEqual(8, config.execution.batch_size)
        self.assertEqual("core", config.metadata["pack"])
        self.assertEqual(
            "v1", config.metadata["pack_version"],
        )

    def test_payload_output_dir_uses_pack_slug(
        self,
    ) -> None:
        pack = get_pack("jailbreak")
        payload = build_pack_run_payload(
            pack=pack,
            model_adapter="hf",
            model_name="meta-llama/Llama-Guard-3-8B",
            output_root="out/custom",
        )
        self.assertIn(
            "jailbreak-v1",
            payload["output"]["run_dir"],
        )
        self.assertTrue(
            payload["output"]["run_dir"].startswith(
                "out/custom/"
            )
        )

    def test_payload_respects_threshold_override(
        self,
    ) -> None:
        pack = get_pack("toxicity")
        payload = build_pack_run_payload(
            pack=pack,
            model_adapter="mock",
            threshold=0.7,
        )
        self.assertEqual(0.7, payload["threshold"])

    def test_payload_with_model_args(self) -> None:
        pack = get_pack("core")
        payload = build_pack_run_payload(
            pack=pack,
            model_adapter="openai_compatible",
            model_name="gpt-4o-mini",
            model_args={
                "root_url": "https://api.openai.com",
                "api_key_env": "OPENAI_API_KEY",
            },
        )
        config = load_config(payload)
        self.assertEqual(
            "https://api.openai.com",
            config.model.args["root_url"],
        )


class PackConfigResolutionTest(unittest.TestCase):
    """Test --pack flag resolves to valid configs."""

    def test_resolve_pack_config_in_process(
        self,
    ) -> None:
        """Exercise _resolve_pack_config end-to-end
        without downloading datasets."""
        from guard_eval_harness.cli.main import (
            HarnessCLI,
        )

        cli = HarnessCLI()
        args = cli.parse_args(
            [
                "run",
                "--pack",
                "core",
                "--model",
                "mock",
                "--model-args",
                '{"strategy": "label_echo"}',
                "--limit",
                "2",
                "--output-dir",
                "/tmp/pack-test",
            ]
        )
        config = cli._resolve_pack_config(args)
        self.assertEqual(
            "pack-core-mock", config.run_name,
        )
        self.assertEqual("mock", config.model.adapter)
        self.assertEqual(12, len(config.datasets))
        self.assertEqual(2, config.execution.limit)
        self.assertEqual(
            "label_echo",
            config.model.args["strategy"],
        )
        self.assertEqual(
            "core", config.metadata["pack"],
        )

    def test_resolve_pack_with_model_name(
        self,
    ) -> None:
        from guard_eval_harness.cli.main import (
            HarnessCLI,
        )

        cli = HarnessCLI()
        args = cli.parse_args(
            [
                "run",
                "--pack",
                "jailbreak",
                "--model",
                "hf",
                "--model-name",
                "meta-llama/Llama-Guard-3-8B",
                "--batch-size",
                "16",
            ]
        )
        config = cli._resolve_pack_config(args)
        self.assertEqual("hf", config.model.adapter)
        self.assertEqual(
            "meta-llama/Llama-Guard-3-8B",
            config.model.model_name,
        )
        self.assertEqual(
            16, config.execution.batch_size,
        )
        self.assertEqual(
            10, len(config.datasets),
        )


class CliPackSmokeTest(unittest.TestCase):
    """CLI integration tests for --pack flag."""

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = (
            self.root / "src"
        ).as_posix()

    def test_list_packs_cli(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "guard_eval_harness",
                "list",
                "packs",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
            env=self.env,
        )
        payload = json.loads(completed.stdout)
        names = [p["name"] for p in payload["packs"]]
        self.assertIn("core", names)
        self.assertIn("jailbreak", names)
        self.assertIn("toxicity", names)
        self.assertIn("hate_harassment", names)
        self.assertIn("prompt_injection", names)

    def test_run_pack_cli_resolves_config(
        self,
    ) -> None:
        """Verify --pack + --model resolve to a valid
        config via subprocess without downloading HF
        datasets (which would fail in CI)."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from guard_eval_harness.cli.main "
                    "import HarnessCLI; "
                    "cli = HarnessCLI(); "
                    "args = cli.parse_args(["
                    "'run', '--pack', 'core', "
                    "'--model', 'mock', "
                    "'--model-args', "
                    "'{\"strategy\": \"label_echo\"}', "
                    "'--limit', '2', "
                    "'--output-dir', '/tmp/test']); "
                    "cfg = cli._resolve_pack_config("
                    "args); "
                    "print('pack=' + str(args.pack)); "
                    "print('model=' + str("
                    "args.model_adapter)); "
                    "print('datasets=' + str("
                    "len(cfg.datasets)))"
                ),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
            env=self.env,
        )
        self.assertIn("pack=core", completed.stdout)
        self.assertIn(
            "model=mock", completed.stdout,
        )
        self.assertIn(
            "datasets=12", completed.stdout,
        )

    def test_pack_without_model_fails(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "guard_eval_harness",
                "run",
                "--pack",
                "core",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertNotEqual(0, completed.returncode)

    def test_config_and_pack_mutually_exclusive(
        self,
    ) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "guard_eval_harness",
                "run",
                "--config",
                "examples/run-mock-jsonl.yaml",
                "--pack",
                "core",
                "--model",
                "mock",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertNotEqual(0, completed.returncode)


if __name__ == "__main__":
    unittest.main()
