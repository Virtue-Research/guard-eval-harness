"""Tests for the guard profile registry + loader + config integration."""

import unittest

from guard_eval_harness.config import load_config
from guard_eval_harness.guards.profiles import (
    deep_merge,
    list_profiles,
    load_profile,
)


class ProfileRegistryTest(unittest.TestCase):
    """Bundled profile lookup + listing."""

    def test_lists_all_bundled_profiles(self) -> None:
        slugs = list_profiles()
        # Stage 1
        self.assertIn("llama-guard-3-8b", slugs)
        self.assertIn("llama-guard-3-1b", slugs)
        self.assertIn("shieldgemma-9b", slugs)
        self.assertIn("md-judge", slugs)
        self.assertIn("wildguard", slugs)
        self.assertIn("qwen3guard-0.6b", slugs)
        self.assertIn("qwen3guard-4b", slugs)
        self.assertIn("gpt-4o-mini", slugs)
        # Stage 2
        self.assertIn("prompt-guard-86m", slugs)
        self.assertIn("llama-prompt-guard-22m", slugs)
        self.assertIn("granite-guardian-3.1-8b", slugs)
        self.assertIn("granite-guardian-3.2-5b", slugs)
        # Filler API/vLLM profiles
        self.assertIn("gpt-5.4-mini", slugs)
        self.assertIn("gemma4-31b-it", slugs)

    def test_load_known_profile(self) -> None:
        profile = load_profile("llama-guard-3-8b")
        self.assertEqual(profile["guard"], "llama_guard")
        self.assertEqual(profile["backend"]["kind"], "hf_generate")
        self.assertEqual(
            profile["backend"]["name"], "meta-llama/Llama-Guard-3-8B"
        )

    def test_unknown_profile_raises_with_hint(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            load_profile("does-not-exist")
        message = str(ctx.exception)
        self.assertIn("does-not-exist", message)
        # Available list is shown
        self.assertIn("wildguard", message)


class DeepMergeTest(unittest.TestCase):
    """Deep merge semantics used to layer user overrides on profiles."""

    def test_scalar_override(self) -> None:
        merged = deep_merge({"a": 1, "b": 2}, {"a": 99})
        self.assertEqual(merged, {"a": 99, "b": 2})

    def test_nested_dict_merges_per_key(self) -> None:
        merged = deep_merge(
            {"x": {"p": 1, "q": 2}},
            {"x": {"q": 99, "r": 3}},
        )
        self.assertEqual(merged, {"x": {"p": 1, "q": 99, "r": 3}})

    def test_list_is_replaced_wholesale(self) -> None:
        merged = deep_merge(
            {"l": [1, 2, 3]}, {"l": [9]}
        )
        self.assertEqual(merged, {"l": [9]})

    def test_inputs_are_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"c": 2}})
        self.assertEqual(base, {"a": {"b": 1}})


def _payload_with_model(model: dict) -> dict:
    """Build a minimal v2 run config with the given model block."""
    return {
        "version": 2,
        "run_name": "p-test",
        "threshold": 0.5,
        "model": model,
        "datasets": [
            {
                "name": "d1",
                "adapter": "local_jsonl",
                "policy": "general_safety",
            },
        ],
        "output": {"run_dir": "out/p", "resume": True},
    }


class ProfileInConfigTest(unittest.TestCase):
    """Profile expansion inside the full config loader."""

    def test_profile_expands_to_full_model_block(self) -> None:
        cfg = load_config(
            _payload_with_model({"profile": "wildguard"})
        )
        self.assertEqual(cfg.model.guard, "wildguard")
        self.assertEqual(cfg.model.backend.kind, "hf_generate")
        self.assertEqual(
            cfg.model.backend.name, "allenai/wildguard"
        )
        # max_new_tokens lands in args
        self.assertEqual(
            cfg.model.backend.args["max_new_tokens"], 32
        )

    def test_user_overrides_win_over_profile(self) -> None:
        cfg = load_config(
            _payload_with_model(
                {
                    "profile": "llama-guard-3-8b",
                    "backend": {
                        "args": {
                            "device": "cuda",
                            "max_new_tokens": 64,
                        },
                    },
                },
            ),
        )
        # Inherited from profile
        self.assertEqual(cfg.model.guard, "llama_guard")
        self.assertEqual(
            cfg.model.backend.name, "meta-llama/Llama-Guard-3-8B"
        )
        # Overridden by user
        self.assertEqual(cfg.model.backend.args["device"], "cuda")
        self.assertEqual(
            cfg.model.backend.args["max_new_tokens"], 64
        )
        # Profile field not overridden survives
        self.assertEqual(
            cfg.model.backend.args["temperature"], 0.0
        )

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(KeyError):
            load_config(
                _payload_with_model({"profile": "no-such"})
            )

    def test_empty_profile_slug_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config(_payload_with_model({"profile": "   "}))

    def test_profile_must_be_string(self) -> None:
        with self.assertRaises(ValueError):
            load_config(_payload_with_model({"profile": 123}))

    def test_classifier_profile_expands_with_guard_args(self) -> None:
        """Stage-2 profiles ship guard_args (unsafe_labels) — verify it flows."""
        cfg = load_config(
            _payload_with_model({"profile": "prompt-guard-86m"})
        )
        self.assertEqual(cfg.model.guard, "prompt_guard")
        self.assertEqual(
            cfg.model.backend.kind, "hf_text_classifier"
        )
        self.assertEqual(
            cfg.model.guard_args["unsafe_labels"],
            ["INJECTION", "JAILBREAK"],
        )

    def test_granite_profile_uses_generate(self) -> None:
        cfg = load_config(
            _payload_with_model({"profile": "granite-guardian-3.2-5b"})
        )
        self.assertEqual(cfg.model.guard, "granite_guardian")
        self.assertEqual(cfg.model.backend.kind, "hf_generate")
        self.assertEqual(
            cfg.model.backend.args["max_new_tokens"], 2
        )


if __name__ == "__main__":
    unittest.main()
