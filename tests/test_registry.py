"""Tests for plugin-friendly registry behavior."""

from __future__ import annotations

import unittest

from guard_eval_harness.registry import Registry, ensure_builtin_registrations
from guard_eval_harness.registry import dataset_registry, model_registry


class RegistryTest(unittest.TestCase):
    """Validate registry behavior."""

    def test_lazy_materialization(self) -> None:
        registry = Registry("test")
        registry.register("join", target="os.path:join")
        value = registry.get("join")
        self.assertTrue(callable(value))

    def test_duplicate_alias_raises(self) -> None:
        registry = Registry("test")

        @registry.register("demo")
        class First:
            pass

        with self.assertRaises(ValueError):

            @registry.register("demo")
            class Second:
                pass

        self.assertIsNotNone(First)

    def test_builtin_discovery_avoids_central_registry_file(self) -> None:
        ensure_builtin_registrations()
        self.assertIn("advbench_behaviors", dataset_registry.keys())
        self.assertIn("cat_qa", dataset_registry.keys())
        self.assertIn("do_anything_now_questions", dataset_registry.keys())
        self.assertIn("i_cona", dataset_registry.keys())
        self.assertIn("malicious_instruct", dataset_registry.keys())
        self.assertIn("mitre", dataset_registry.keys())
        self.assertIn("harmful_q", dataset_registry.keys())
        self.assertIn("harmful_qa_questions", dataset_registry.keys())
        self.assertIn("harm_eval", dataset_registry.keys())
        self.assertIn("niche_hazard_qa", dataset_registry.keys())
        self.assertIn("tdc_red_teaming", dataset_registry.keys())
        self.assertIn("advbench_strings", dataset_registry.keys())
        self.assertIn("hatecheck", dataset_registry.keys())
        self.assertIn("safe_text", dataset_registry.keys())
        self.assertIn("simple_safety_tests", dataset_registry.keys())
        self.assertIn("tech_hazard_qa", dataset_registry.keys())
        self.assertIn("jbb_behaviors", dataset_registry.keys())
        self.assertIn("strong_reject_instructions", dataset_registry.keys())
        self.assertIn("beaver_tails_330k", dataset_registry.keys())
        self.assertIn("pku_safe_rlhf", dataset_registry.keys())
        self.assertIn("wildguardmix", dataset_registry.keys())
        self.assertIn("circleguardbench_public", dataset_registry.keys())
        self.assertIn("civil_comments", dataset_registry.keys())
        self.assertIn("real_toxicity_prompts", dataset_registry.keys())
        self.assertIn(
            "aegis_ai_content_safety_dataset_2", dataset_registry.keys()
        )
        self.assertIn("local_jsonl", dataset_registry.keys())
        self.assertIn("local_csv", dataset_registry.keys())
        self.assertIn("msts", dataset_registry.keys())
        self.assertIn("vlsbench", dataset_registry.keys())
        self.assertIn("jailbreakv_28k", dataset_registry.keys())
        self.assertIn("mm_safetybench", dataset_registry.keys())
        self.assertIn("local_image_jsonl", dataset_registry.keys())
        self.assertIn("local_image_dir", dataset_registry.keys())
        self.assertIn("mock", model_registry.keys())
        self.assertIn("hf", model_registry.keys())
        self.assertIn("hf_image_classifier", model_registry.keys())
        self.assertIn("hf_vlm_guard", model_registry.keys())
        self.assertIn("hf_shieldgemma2", model_registry.keys())
        self.assertIn("openai_moderation", model_registry.keys())


if __name__ == "__main__":
    unittest.main()
