"""Tests for the get_policy_raw helper."""

import unittest

from guard_eval_harness.policies import get_policy_raw


class GetPolicyRawTest(unittest.TestCase):
    """Structured policy lookup for guards that re-render policy JSON."""

    def test_returns_dict_for_bundled_upstream_dataset(self) -> None:
        raw = get_policy_raw("dataset_toxic_chat_upstream")
        self.assertIsNotNone(raw)
        self.assertIn("policies", raw)
        # ToxicChat upstream has 2 policies (Toxicity, Jailbreaking)
        self.assertEqual(len(raw["policies"]), 2)
        for p in raw["policies"]:
            self.assertIn("policy_name", p)
            self.assertIn("policy_description", p)
            self.assertIn("block_activities", p)

    def test_returns_dict_for_bundled_generated_dataset(self) -> None:
        raw = get_policy_raw("dataset_advbench_behaviors_generated")
        self.assertIsNotNone(raw)
        self.assertIn("policies", raw)

    def test_returns_virtue_general(self) -> None:
        raw = get_policy_raw("virtue_general")
        self.assertIsNotNone(raw)
        self.assertIn("policies", raw)

    def test_returns_none_for_inline_or_unbundled(self) -> None:
        self.assertIsNone(get_policy_raw("general_safety"))
        self.assertIsNone(get_policy_raw("mlcommons_v1"))
        self.assertIsNone(get_policy_raw("does_not_exist"))
        self.assertIsNone(get_policy_raw("dataset_no_such_dataset_upstream"))


if __name__ == "__main__":
    unittest.main()
