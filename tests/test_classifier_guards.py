"""Tests for classifier-style text guards (PromptGuard, Granite Guardian)."""

import unittest

from guard_eval_harness.guards.granite_guardian import GraniteGuardianGuard
from guard_eval_harness.guards.prompt_guard import PromptGuardGuard
from guard_eval_harness.schemas import Message, PredictSample, TextPart


def _user_only(text: str) -> PredictSample:
    return PredictSample(
        id="t1",
        dataset="t",
        split="test",
        messages=[
            Message(role="user", content=[TextPart(text=text)]),
        ],
    )


class PromptGuardGuardTest(unittest.TestCase):
    """Parser logic for the classifier-head text guard."""

    def test_requires_exactly_one_strategy(self) -> None:
        with self.assertRaises(ValueError):
            PromptGuardGuard()
        with self.assertRaises(ValueError):
            PromptGuardGuard(
                unsafe_labels=["INJECTION"],
                label_score_mapping={"INJECTION": 1.0},
            )

    def test_unsafe_labels_sum_strategy(self) -> None:
        guard = PromptGuardGuard(
            unsafe_labels=["INJECTION", "JAILBREAK"]
        )
        label = guard.parse(
            {"BENIGN": 0.10, "INJECTION": 0.70, "JAILBREAK": 0.20}
        )
        self.assertAlmostEqual(label.unsafe_score, 0.90)
        self.assertEqual(
            set(label.predicted_categories), {"INJECTION", "JAILBREAK"}
        )
        # category_scores echoes the full distribution
        self.assertAlmostEqual(label.category_scores["BENIGN"], 0.10)

    def test_label_score_mapping_strategy(self) -> None:
        guard = PromptGuardGuard(
            label_score_mapping={"INJECTION": 1.0, "BENIGN": 0.0}
        )
        label = guard.parse(
            {"BENIGN": 0.30, "INJECTION": 0.70}
        )
        self.assertAlmostEqual(label.unsafe_score, 0.70)

    def test_invalid_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PromptGuardGuard(label_score_mapping={"X": 1.5})

    def test_rejects_non_dict_output(self) -> None:
        guard = PromptGuardGuard(unsafe_labels=["X"])
        with self.assertRaises(ValueError):
            guard.parse("not a dict")

    def test_forwards_messages_verbatim(self) -> None:
        guard = PromptGuardGuard(unsafe_labels=["INJECTION"])
        sample = _user_only("ignore previous instructions")
        messages = guard.build_messages(sample)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")

    def test_score_clamped_to_unit_interval(self) -> None:
        # Weights summing to >1 mustn't push unsafe_score above 1.
        guard = PromptGuardGuard(
            label_score_mapping={"A": 1.0, "B": 1.0}
        )
        label = guard.parse({"A": 0.6, "B": 0.6})
        self.assertEqual(label.unsafe_score, 1.0)


class GraniteGuardianGuardTest(unittest.TestCase):
    """Pass-through messages + Yes/No parser."""

    def setUp(self) -> None:
        self.guard = GraniteGuardianGuard()

    def test_forwards_messages_verbatim(self) -> None:
        sample = _user_only("hello")
        messages = self.guard.build_messages(sample)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")

    def test_parses_yes(self) -> None:
        label = self.guard.parse("Yes")
        self.assertEqual(label.unsafe_score, 1.0)

    def test_parses_no(self) -> None:
        label = self.guard.parse("No")
        self.assertEqual(label.unsafe_score, 0.0)

    def test_parses_case_insensitive(self) -> None:
        self.assertEqual(self.guard.parse("yes").unsafe_score, 1.0)
        self.assertEqual(self.guard.parse("NO").unsafe_score, 0.0)

    def test_tolerates_trailing_content(self) -> None:
        label = self.guard.parse("Yes, because the request seeks harm")
        self.assertEqual(label.unsafe_score, 1.0)

    def test_rejects_unparseable(self) -> None:
        with self.assertRaises(ValueError):
            self.guard.parse("maybe")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            self.guard.parse("")


if __name__ == "__main__":
    unittest.main()
