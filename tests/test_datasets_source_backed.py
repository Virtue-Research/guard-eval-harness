"""Tests for source-backed dataset helpers."""

from __future__ import annotations

import tempfile
import unittest
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard_eval_harness.config.models import ResolvedDatasetConfig
from guard_eval_harness.datasets.source_backed import (
    cached_download,
    load_csv_rows,
    load_hf_rows,
    load_json_payload,
    load_text_lines,
)
from guard_eval_harness.registry import dataset_registry, ensure_builtin_registrations


class SourceBackedHelperTest(unittest.TestCase):
    """Validate HF revision pinning and smoke-limit loading behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_builtin_registrations()

    def test_load_hf_rows_passes_revision_and_sliced_split(self) -> None:
        mock_load_dataset = unittest.mock.Mock()
        with patch.dict(
            "sys.modules",
            {"datasets": SimpleNamespace(load_dataset=mock_load_dataset)},
        ):
            mock_load_dataset.return_value.to_list.return_value = [{"id": "1"}]

            rows = load_hf_rows(
                "demo/repo",
                split="train",
                subset="demo",
                revision="abc123",
                limit=5,
                offset=10,
            )

        self.assertEqual(rows, [{"id": "1"}])
        mock_load_dataset.assert_called_once_with(
            "demo/repo",
            "demo",
            split="train[10:15]",
            revision="abc123",
        )

    def test_load_csv_rows_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "rows.csv"
            csv_path.write_text(
                "id,value\n1,a\n2,b\n3,c\n",
                encoding="utf-8",
            )

            with patch(
                "guard_eval_harness.datasets.source_backed.cached_download",
                return_value=csv_path,
            ):
                rows = load_csv_rows(
                    alias="demo",
                    url="https://example.test/rows.csv",
                    filename="rows.csv",
                    limit=2,
                )

        self.assertEqual(rows, [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}])

    def test_load_json_payload_reads_remote_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "rows.json"
            json_path.write_text('{"items": ["a", "b"]}', encoding="utf-8")

            with patch(
                "guard_eval_harness.datasets.source_backed.cached_download",
                return_value=json_path,
            ):
                payload = load_json_payload(
                    alias="demo",
                    url="https://example.test/rows.json",
                    filename="rows.json",
                )

        self.assertEqual(payload, {"items": ["a", "b"]})

    def test_load_text_lines_respects_limit_and_skips_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / "rows.txt"
            text_path.write_text("first\n\nsecond\nthird\n", encoding="utf-8")

            with patch(
                "guard_eval_harness.datasets.source_backed.cached_download",
                return_value=text_path,
            ):
                lines = load_text_lines(
                    alias="demo",
                    url="https://example.test/rows.txt",
                    filename="rows.txt",
                    limit=2,
                )

        self.assertEqual(lines, ["first", "second"])

    def test_cached_download_includes_url_hash_in_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            class _Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return b"payload"

            with patch(
                "guard_eval_harness.datasets.source_backed._cache_root",
                return_value=root,
            ), patch(
                "guard_eval_harness.datasets.source_backed.urllib_request.urlopen",
                return_value=_Response(),
            ):
                first = cached_download(
                    alias="demo",
                    url="https://example.test/a.csv",
                    filename="rows.csv",
                )
                second = cached_download(
                    alias="demo",
                    url="https://example.test/b.csv",
                    filename="rows.csv",
                )

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.parent.name, "demo")
        self.assertEqual(second.parent.parent.name, "demo")

    def test_new_hf_datasets_pin_immutable_revisions(self) -> None:
        from guard_eval_harness.datasets import aegis_ai_content_safety_dataset_2
        from guard_eval_harness.datasets import ai_vs_real
        from guard_eval_harness.datasets import circleguardbench_public
        from guard_eval_harness.datasets import civil_comments
        from guard_eval_harness.datasets import hateful_memes
        from guard_eval_harness.datasets import real_toxicity_prompts
        from guard_eval_harness.datasets import self_harm_image_dataset
        from guard_eval_harness.datasets import wildguardmix

        revision_pattern = re.compile(r"^[0-9a-f]{40}$")
        for module in (
            aegis_ai_content_safety_dataset_2,
            ai_vs_real,
            circleguardbench_public,
            civil_comments,
            hateful_memes,
            real_toxicity_prompts,
            self_harm_image_dataset,
            wildguardmix,
        ):
            revision = getattr(module, "_REVISION")
            self.assertRegex(revision, revision_pattern)

    def test_cached_download_passes_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            class _Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return b"payload"

            with patch(
                "guard_eval_harness.datasets.source_backed._cache_root",
                return_value=root,
            ), patch(
                "guard_eval_harness.datasets.source_backed"
                ".urllib_request.urlopen",
                return_value=_Response(),
            ) as mock_urlopen:
                cached_download(
                    alias="demo",
                    url="https://example.test/a.csv",
                    filename="rows.csv",
                )

            mock_urlopen.assert_called_once()
            _, kwargs = mock_urlopen.call_args
            self.assertIn("timeout", kwargs)
            self.assertGreater(kwargs["timeout"], 0)

    def test_beavertails_uses_incremental_fetch_for_smoke_limit(self) -> None:
        dataset_cls = dataset_registry.get("beaver_tails_330k")
        dataset = dataset_cls.from_config(
            ResolvedDatasetConfig(
                name="beaver_tails_330k",
                adapter="beaver_tails_330k",
                options={"limit": 2},
            )
        )
        duplicate = {
            "prompt": "Prompt A",
            "response": "Response A",
            "is_safe": True,
            "category": {"violence": False, "privacy": True},
        }
        unique = {
            "prompt": "Prompt B",
            "response": "Response B",
            "is_safe": False,
            "category": {"violence": True, "privacy": False},
        }
        with patch(
            "guard_eval_harness.datasets.beaver_tails_330k.load_hf_rows",
            side_effect=[
                [duplicate, duplicate],
                [unique],
            ],
        ) as mock_load_hf_rows:
            samples = dataset.load()

        self.assertEqual(len(samples), 2)
        self.assertEqual(mock_load_hf_rows.call_count, 2)
        first_call = mock_load_hf_rows.call_args_list[0].kwargs
        second_call = mock_load_hf_rows.call_args_list[1].kwargs
        self.assertEqual(first_call["limit"], 32)
        self.assertEqual(first_call["offset"], 0)
        self.assertEqual(second_call["offset"], 2)


if __name__ == "__main__":
    unittest.main()
