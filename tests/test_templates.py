"""Tests for score resolution and path extraction."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from guard_eval_harness.models.templates import (
    env_value,
    extract_path,
    render_template,
    resolve_score,
    sample_context,
)
from guard_eval_harness.schemas import Message, NormalizedSample, PredictSample


class ExtractPathTest(unittest.TestCase):
    """Validate dotted path extraction from nested responses."""

    def test_extracts_nested_mapping(self) -> None:
        payload = {"a": {"b": {"c": 42}}}
        self.assertEqual(extract_path(payload, "a.b.c"), 42)

    def test_extracts_list_index(self) -> None:
        payload = {"items": ["x", "y", "z"]}
        self.assertEqual(extract_path(payload, "items.1"), "y")

    def test_returns_value_when_path_is_none(self) -> None:
        self.assertEqual(extract_path({"k": 1}, None), {"k": 1})

    def test_raises_key_error_on_missing_key(self) -> None:
        with self.assertRaises(KeyError):
            extract_path({"a": 1}, "b")

    def test_raises_index_error_on_bad_index(self) -> None:
        with self.assertRaises(IndexError):
            extract_path({"items": []}, "items.0")


class ResolveScoreTest(unittest.TestCase):
    """Validate score resolution with fallback chain."""

    def test_uses_score_path_when_valid(self) -> None:
        score = resolve_score(
            {"result": {"unsafe": 0.7}},
            score_path="result.unsafe",
        )
        self.assertAlmostEqual(score, 0.7)

    def test_raises_when_score_path_missing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_score(
                {"score": 0.8},
                score_path="missing.path",
            )
        self.assertIn("missing.path", str(ctx.exception))

    def test_raises_when_score_path_index_oob(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_score(
                {"score": 0.3},
                score_path="items.0",
            )
        self.assertIn("items.0", str(ctx.exception))

    def test_raises_when_no_score_found(self) -> None:
        with self.assertRaises(ValueError):
            resolve_score({"unrelated": "data"})


class EnvValueTest(unittest.TestCase):
    """Validate environment-backed config overrides."""

    def test_empty_env_var_overrides_default(self) -> None:
        with patch.dict(os.environ, {"GEH_EMPTY": ""}, clear=False):
            self.assertEqual(env_value("GEH_EMPTY", "fallback"), "")

    def test_missing_env_var_uses_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_value("GEH_EMPTY", "fallback"), "fallback")


class PolicySubstitutionTest(unittest.TestCase):
    """Validate per-sample policy substitution in template rendering."""

    def _sample(self, policy: object) -> PredictSample:
        return NormalizedSample(
            id="test-1",
            dataset="test",
            split="test",
            messages=[Message(role="user", content="hello")],
            label={"unsafe": False},
            metadata={"policy": policy},
        ).to_predict_sample(predict_metadata_fields=("policy",))

    def test_string_policy_injected_into_template(self) -> None:
        context = sample_context(self._sample("Refuse harmful content."))
        rendered = render_template(
            "Policy: {policy}\nText: {messages_text}", context
        )
        self.assertIn("Refuse harmful content.", rendered)
        self.assertIn("user: hello", rendered)

    def test_dict_policy_serialized_as_json(self) -> None:
        context = sample_context(
            self._sample({"name": "P1", "rules": ["no violence"]})
        )
        rendered = render_template("{policy}", context)
        self.assertIn('"name": "P1"', rendered)
        self.assertIn("no violence", rendered)

    def test_missing_policy_renders_empty_string(self) -> None:
        sample = NormalizedSample(
            id="test-2",
            dataset="test",
            split="test",
            messages=[Message(role="user", content="hi")],
            label={"unsafe": False},
            metadata={"policy": "hidden"},
        ).to_predict_sample()
        context = sample_context(sample)
        self.assertEqual(render_template("[{policy}]", context), "[]")

    def test_context_uses_only_predict_visible_metadata(self) -> None:
        sample = NormalizedSample(
            id="test-3",
            dataset="test",
            split="test",
            messages=[Message(role="user", content="hi")],
            label={"unsafe": True},
            metadata={
                "policy": "hidden",
                "target_role": "assistant",
                "category": "visible",
            },
        ).to_predict_sample(predict_metadata_fields=("category",))

        context = sample_context(sample)

        self.assertNotIn("label", context)
        self.assertNotIn("unsafe", context)
        self.assertEqual(context["raw_metadata"], {"category": "visible"})
        self.assertEqual(context["metadata_json"], '{"category": "visible"}')
        self.assertEqual(context["metadata_category"], "visible")
        self.assertEqual(context["metadata_target_role"], "User")
        self.assertNotIn("metadata_policy", context)


if __name__ == "__main__":
    unittest.main()
