"""Import-smoke tests for VLM adapters.

These exist to catch transformers-API regressions (e.g. classes
removed across major versions) without requiring model weights.
A plain ``import`` of the adapter module only catches outright
``ImportError``; the slightly richer checks here also verify that
every flow-resolved ``transformers.*`` class name still exists in
the installed transformers package.
"""

from __future__ import annotations

import importlib
import unittest

from guard_eval_harness.config.models import ResolvedModelConfig


class VLMAdapterImportTest(unittest.TestCase):
    """Verify VLM adapter modules import and resolve real classes."""

    def test_hf_vlm_guard_module_imports(self) -> None:
        module = importlib.import_module(
            "guard_eval_harness.models.hf_vlm_guard"
        )
        self.assertTrue(
            hasattr(module, "HuggingFaceVLMGuardAdapter"),
            "hf_vlm_guard.HuggingFaceVLMGuardAdapter not exported",
        )

    def test_hf_safeqwen_vlm_module_imports(self) -> None:
        module = importlib.import_module(
            "guard_eval_harness.models.hf_safeqwen_vlm"
        )
        self.assertTrue(
            hasattr(module, "SafeQwenVLMAdapter"),
            "hf_safeqwen_vlm.SafeQwenVLMAdapter not exported",
        )

    def test_hf_gemma4_vlm_module_imports(self) -> None:
        module = importlib.import_module(
            "guard_eval_harness.models.hf_gemma4_vlm"
        )
        self.assertTrue(
            hasattr(module, "HuggingFaceGemma4VLMAdapter"),
            "hf_gemma4_vlm.HuggingFaceGemma4VLMAdapter not exported",
        )


class VLMAdapterResolvedClassExistsTest(unittest.TestCase):
    """Verify each flow's model_class is present in the installed transformers.

    Regression guard: transformers 5.x removed ``AutoModelForVision2Seq``
    in favor of ``AutoModelForImageTextToText``. Any future Auto-class
    removal should be caught here before runtime model loading.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.transformers = importlib.import_module("transformers")
        except ImportError:
            raise unittest.SkipTest("transformers not installed")

    def _assert_class_available(self, class_name: str) -> None:
        cls_obj = getattr(self.transformers, class_name, None)
        version = getattr(self.transformers, "__version__", "unknown")
        self.assertIsNotNone(
            cls_obj,
            f"transformers=={version} does not expose {class_name!r}; "
            "check whether it was renamed or removed in this version.",
        )

    def test_hf_vlm_guard_flows_resolve_real_classes(self) -> None:
        from guard_eval_harness.models.hf_vlm_guard import (
            HuggingFaceVLMGuardAdapter,
        )

        # (flow_name, model_name_hint) pairs covering every flow
        # branch in HuggingFaceVLMGuardAdapter._model_class_name().
        cases = [
            ("llama_guard_4", "meta-llama/Llama-Guard-4-12B"),
            ("llama_guard_3_vision", "meta-llama/Llama-Guard-3-11B-Vision"),
            ("llavaguard", "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"),
            ("guardreasoner_vl", "yueliu1999/GuardReasoner-VL-7B"),
            # internvl_chat uses AutoModel which is universally available;
            # included for completeness.
            ("internvl_chat", "OpenGVLab/InternVL3-1B"),
        ]
        for flow_name, model_name in cases:
            with self.subTest(flow=flow_name):
                config = ResolvedModelConfig(
                    adapter="hf_vlm_guard",
                    model_name=model_name,
                    args={"flow": flow_name},
                )
                adapter = HuggingFaceVLMGuardAdapter.from_config(config)
                resolved = adapter._model_class_name()
                # The removed-in-5.x class must not be used directly.
                self.assertNotEqual(
                    resolved,
                    "AutoModelForVision2Seq",
                    f"flow {flow_name!r} still references the removed "
                    "AutoModelForVision2Seq",
                )
                self._assert_class_available(resolved)

    def test_safeqwen_resolves_image_text_to_text_class(self) -> None:
        # SafeQwen iterates ("AutoModelForImageTextToText",
        # "AutoModelForVision2Seq") in that order. On any supported
        # transformers (>=4.50) the first name must exist.
        self._assert_class_available("AutoModelForImageTextToText")

    def test_gemma4_resolves_image_text_to_text_class(self) -> None:
        self._assert_class_available("AutoModelForImageTextToText")


if __name__ == "__main__":
    unittest.main()
