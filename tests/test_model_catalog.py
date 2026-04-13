"""Tests for the model catalog."""

from __future__ import annotations

import json
import unittest

from guard_eval_harness.models.catalog import (
    MODEL_CATALOG,
    ModelProfile,
    list_catalog,
    resolve_catalog,
)


class CatalogLookupTest(unittest.TestCase):
    """Validate catalog resolution."""

    def test_resolve_known_slug(self) -> None:
        entry = resolve_catalog("llama-guard-3-8b")
        self.assertIsNotNone(entry)
        self.assertEqual("hf", entry.adapter)
        self.assertEqual(
            "meta-llama/Llama-Guard-3-8B",
            entry.model_name,
        )
        self.assertEqual(
            "text-generation",
            entry.args["task"],
        )
        self.assertEqual(
            "llama_guard",
            entry.args["chat_template_profile"],
        )

    def test_resolve_vllm_backend(self) -> None:
        entry = resolve_catalog(
            "llama-guard-3-8b", backend="vllm",
        )
        self.assertIsNotNone(entry)
        self.assertEqual("vllm", entry.adapter)
        self.assertEqual(
            "meta-llama/Llama-Guard-3-8B",
            entry.model_name,
        )

    def test_resolve_unknown_returns_none(self) -> None:
        self.assertIsNone(resolve_catalog("not-a-model"))

    def test_resolve_raw_adapter_returns_none(self) -> None:
        self.assertIsNone(resolve_catalog("hf"))
        self.assertIsNone(resolve_catalog("mock"))

    def test_all_entries_have_adapter_and_model_name(self) -> None:
        for slug, profile in MODEL_CATALOG.items():
            with self.subTest(slug=slug):
                self.assertIsInstance(profile, ModelProfile)
                self.assertIn(
                    profile.adapter,
                    {"hf", "openai_compatible"},
                )
                self.assertIsNotNone(profile.model_name)

    def test_all_hf_generative_entries_set_task(self) -> None:
        for slug, profile in MODEL_CATALOG.items():
            if profile.adapter != "hf":
                continue
            with self.subTest(slug=slug):
                self.assertIn("task", profile.args)

    def test_list_catalog_returns_all(self) -> None:
        listing = list_catalog()
        self.assertEqual(len(MODEL_CATALOG), len(listing))
        slugs = {entry["slug"] for entry in listing}
        self.assertEqual(set(MODEL_CATALOG), slugs)


class CLICatalogResolutionTest(unittest.TestCase):
    """Validate CLI resolves catalog slugs correctly."""

    def test_catalog_hit_overrides_adapter(self) -> None:
        from guard_eval_harness.cli.main import HarnessCLI

        cli = HarnessCLI()
        args = cli.parse_args([
            "run", "--dataset", "xstest",
            "--model", "llama-guard-3-8b",
            "--limit", "1",
            "--output-dir", "/tmp/test",
        ])
        adapter, model_name, model_args = (
            cli._resolve_model(args)
        )
        self.assertEqual("hf", adapter)
        self.assertEqual(
            "meta-llama/Llama-Guard-3-8B",
            model_name,
        )
        self.assertEqual(
            "text-generation",
            model_args["task"],
        )

    def test_catalog_miss_uses_raw_adapter(self) -> None:
        from guard_eval_harness.cli.main import HarnessCLI

        cli = HarnessCLI()
        args = cli.parse_args([
            "run", "--dataset", "xstest",
            "--model", "mock",
            "--limit", "1",
            "--output-dir", "/tmp/test",
        ])
        adapter, model_name, model_args = (
            cli._resolve_model(args)
        )
        self.assertEqual("mock", adapter)
        self.assertIsNone(model_name)
        self.assertEqual({}, model_args)

    def test_user_args_override_catalog(self) -> None:
        from guard_eval_harness.cli.main import HarnessCLI

        cli = HarnessCLI()
        args = cli.parse_args([
            "run", "--dataset", "xstest",
            "--model", "llama-guard-3-8b",
            "--model-args", '{"device": 2}',
            "--limit", "1",
            "--output-dir", "/tmp/test",
        ])
        adapter, model_name, model_args = (
            cli._resolve_model(args)
        )
        self.assertEqual("hf", adapter)
        self.assertEqual(2, model_args["device"])
        self.assertEqual(
            "text-generation",
            model_args["task"],
        )

    def test_list_models_json(self) -> None:
        from guard_eval_harness.cli.main import HarnessCLI
        import io
        import sys

        cli = HarnessCLI()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            cli.execute(["list", "models"])
        finally:
            sys.stdout = old_stdout
        output = json.loads(buf.getvalue())
        self.assertIn("models", output)
        self.assertGreater(len(output["models"]), 20)


if __name__ == "__main__":
    unittest.main()
