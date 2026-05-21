"""Source-backed Do Anything Now Questions dataset adapter."""

import csv
from zipfile import ZipFile

from guard_eval_harness.datasets.base import (
    DatasetAdapter,
    cached_download,
    load_csv_rows,
)
from guard_eval_harness.registry import dataset_registry


_BASE_URL = (
    "https://github.com/verazuo/jailbreak_llms/raw/"
    "09ab5ecb9632087119af25a04b88b0f496c989be/data/forbidden_question"
)
_FILENAME = "forbidden_question_set.csv"
_PROMPTED_FILENAME = "forbidden_question_set_with_prompts.csv.zip"
_PROMPTED_MEMBER = "forbidden_question_set_with_prompts.csv"
_DEFAULT_PROMPT_TYPE = "earliest_prompt"
_DEFAULT_COMMUNITY_NAME = "Basic"
_DEFAULT_RESPONSE_IDX = "0"


def _load_dan_wrapped_rows(limit: int | None = None) -> list[dict[str, str]]:
    """Load one canonical DAN-wrapped prompt for each forbidden question."""
    archive_path = cached_download(
        alias="do_anything_now_questions",
        url=f"{_BASE_URL}/{_PROMPTED_FILENAME}",
        filename=_PROMPTED_FILENAME,
    )
    rows: list[dict[str, str]] = []
    seen_questions: set[tuple[str, str]] = set()
    with ZipFile(archive_path) as archive:
        with archive.open(_PROMPTED_MEMBER) as raw_handle:
            lines = (
                line.decode("utf-8", errors="replace")
                for line in raw_handle
            )
            reader = csv.DictReader(lines)
            for row in reader:
                if row.get("prompt_type") != _DEFAULT_PROMPT_TYPE:
                    continue
                if row.get("community_name") != _DEFAULT_COMMUNITY_NAME:
                    continue
                if row.get("response_idx") != _DEFAULT_RESPONSE_IDX:
                    continue
                question_key = (
                    str(row.get("content_policy_id", "")),
                    str(row.get("q_id", "")),
                )
                if question_key in seen_questions:
                    continue
                seen_questions.add(question_key)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


@dataset_registry.register("do_anything_now_questions")
class DoAnythingNowQuestionsDataset(DatasetAdapter):
    """Load Do Anything Now questions wrapped with a DAN jailbreak prompt."""

    display_name = "Do Anything Now Questions"
    source_uri = "https://github.com/verazuo/jailbreak_llms"
    license_name = "MIT"
    languages = ("en",)
    categories = (
        "Illegal Activity",
        "Hate Speech",
        "Malware Generation",
        "Physical Harm",
        "Fraud",
        "Pornography",
        "Political Lobbying",
        "Privacy Violence",
        "Legal Opinion",
        "Financial Advice",
        "Health Consultation",
        "Government Decision",
    )
    metadata_fields_to_preserve = (
        "content_policy_id",
        "content_policy_name",
        "q_id",
        "question",
        "jailbreak_prompt",
        "prompt_type",
        "community_name",
    )
    label_mapping_note = (
        "DAN jailbreak prompt plus forbidden question maps to unsafe"
    )

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load the upstream Do Anything Now rows."""
        if str(self.config.options.get("mode", "dan_wrapped")) == "questions":
            return self._load_question_only_rows()

        return [
            {
                "id": (
                    "do-anything-now-dan-"
                    f"{row['content_policy_id']}-{row['q_id']}"
                ),
                "prompt": f"{row['prompt'].strip()}\n\n{row['question'].strip()}",
                "unsafe": True,
                "content_policy_id": row.get("content_policy_id"),
                "content_policy_name": row.get("content_policy_name"),
                "q_id": row.get("q_id"),
                "question": row.get("question"),
                "jailbreak_prompt": row.get("prompt"),
                "prompt_type": row.get("prompt_type"),
                "community_name": row.get("community_name"),
            }
            for row in _load_dan_wrapped_rows(self.execution_limit)
        ]

    def _load_question_only_rows(self) -> list[dict[str, object]]:
        """Load the legacy plain forbidden-question view."""
        rows = load_csv_rows(
            alias="do_anything_now_questions",
            url=f"{_BASE_URL}/{_FILENAME}",
            filename=_FILENAME,
            limit=self.execution_limit,
        )
        return [
            {
                "id": f"do-anything-now-{index}",
                "prompt": row["question"],
                "unsafe": True,
            }
            for index, row in enumerate(rows, start=1)
        ]
