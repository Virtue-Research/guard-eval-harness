"""Unit tests for the shared path-confinement helper."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.vibecoding.safe_path import safe_relpath


class SafeRelpathTest(unittest.TestCase):
    def test_valid_nested_path_is_confined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = safe_relpath(root, "pkg/sub/a.py")
            self.assertTrue(target.is_relative_to(root.resolve()))
            self.assertEqual(target.name, "a.py")

    def test_absolute_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_relpath(tmp, "/etc/passwd")

    def test_parent_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../escape", "a/../../escape", "../../etc/passwd"):
                with self.assertRaises(ValueError):
                    safe_relpath(tmp, bad)

    def test_symlink_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            # A symlink inside root that points outside it.
            link = root / "link"
            os.symlink(outside, link)
            with self.assertRaises(ValueError):
                safe_relpath(root, "link/evil.txt")

    def test_returns_absolute_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = safe_relpath(tmp, "x.txt")
            self.assertTrue(target.is_absolute())


if __name__ == "__main__":
    unittest.main()
