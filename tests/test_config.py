"""Tests for the config loader and validators."""

import unittest

from pydantic import ValidationError

from guard_eval_harness.config import load_config
from guard_eval_harness.config import (
    InlinePolicy,
    ResolvedRunConfig,
)
from guard_eval_harness.guards import guard_registry
from guard_eval_harness.guards.base import Guard


class _FakeFixedGuard(Guard):
    """Stand-in guard that does not accept a custom policy or format."""

    name = "fake_fixed_for_tests"
    accepts_policy = False
    accepts_output_format = False

    def build_messages(self, sample, *, policy=None, output_format=None):
        return []

    def parse(self, output):
        return None


class ConfigV2LoaderTest(unittest.TestCase):
    """Validate the YAML → ResolvedRunConfig path."""

    @classmethod
    def setUpClass(cls) -> None:
        if "fake_fixed_for_tests" not in guard_registry:
            guard_registry.register(
                "fake_fixed_for_tests",
                target=_FakeFixedGuard,
            )

    def _base_payload(self, **overrides):
        payload = {
            "version": 2,
            "run_name": "test-run",
            "threshold": 0.5,
            "model": {
                "guard": "llm",
                "output_format": "safe_unsafe_first_line",
                "backend": {"kind": "mock"},
            },
            "datasets": [
                {
                    "name": "d1",
                    "adapter": "local_jsonl",
                    "policy": "general_safety",
                },
            ],
            "output": {"run_dir": "out/test", "resume": True},
        }
        for key, value in overrides.items():
            payload[key] = value
        return payload

    def test_loads_minimal_config(self) -> None:
        cfg = load_config(self._base_payload())
        self.assertIsInstance(cfg, ResolvedRunConfig)
        self.assertEqual(cfg.model.guard, "llm")
        self.assertEqual(cfg.model.backend.kind, "mock")
        self.assertTrue(cfg.output.resume)
        self.assertEqual(len(cfg.datasets), 1)
        self.assertEqual(cfg.datasets[0].policy, "general_safety")

    def test_inline_policy_object_on_dataset(self) -> None:
        payload = self._base_payload()
        payload["datasets"][0]["policy"] = {
            "name": "inline_p",
            "text": "inline policy text",
            "categories": ["a", "b"],
        }
        cfg = load_config(payload)
        self.assertIsInstance(cfg.datasets[0].policy, InlinePolicy)
        self.assertEqual(cfg.datasets[0].policy.name, "inline_p")

    def test_adapter_defaults_to_name(self) -> None:
        payload = self._base_payload()
        payload["datasets"] = [
            {"name": "xstest"},   # adapter omitted → "xstest"
            {"name": "alias-quick", "adapter": "local_jsonl", "limit": 5},
        ]
        cfg = load_config(payload)
        self.assertEqual(cfg.datasets[0].adapter, "xstest")
        self.assertEqual(cfg.datasets[1].adapter, "local_jsonl")

    def test_rejects_model_level_policy(self) -> None:
        """model.policy is no longer supported; must error clearly."""
        payload = self._base_payload()
        payload["model"]["policy"] = "general_safety"
        with self.assertRaisesRegex(
            ValueError, "model.policy is no longer supported"
        ):
            load_config(payload)

    def test_rejects_output_format_on_fixed_guard(self) -> None:
        payload = self._base_payload()
        payload["model"]["guard"] = "fake_fixed_for_tests"
        payload["model"]["output_format"] = "safe_unsafe_first_line"
        with self.assertRaisesRegex(
            ValueError, "does not accept a custom\\s+output_format"
        ):
            load_config(payload)

    def test_rejects_multiple_subset_selectors(self) -> None:
        payload = self._base_payload()
        payload["datasets"][0].update(
            limit=5,
            sample_indices=[0, 1],
        )
        with self.assertRaisesRegex(
            ValidationError, "at most one of\\s+limit/sample_ids/sample_indices"
        ):
            load_config(payload)

    def test_rejects_resume_with_overwrite(self) -> None:
        payload = self._base_payload()
        payload["output"]["overwrite"] = True
        with self.assertRaisesRegex(
            ValidationError, "mutually exclusive"
        ):
            load_config(payload)

    def test_rejects_duplicate_dataset_names(self) -> None:
        payload = self._base_payload()
        payload["datasets"] = [
            {"name": "x", "adapter": "local_jsonl"},
            {"name": "x", "adapter": "local_jsonl"},
        ]
        with self.assertRaisesRegex(
            ValidationError, "duplicate dataset name"
        ):
            load_config(payload)

    def test_sample_indices_must_be_non_negative(self) -> None:
        payload = self._base_payload()
        payload["datasets"][0]["sample_indices"] = [-1, 0]
        with self.assertRaisesRegex(
            ValidationError, "sample_indices must be non-negative"
        ):
            load_config(payload)


if __name__ == "__main__":
    unittest.main()
