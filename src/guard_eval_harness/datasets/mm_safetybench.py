"""MM-SafetyBench local dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from guard_eval_harness.datasets.multimodal_base import (
    MultimodalDatasetAdapter,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import NormalizedSample, UnsafeLabel


_SCENARIOS = {
    "01-Illegal_Activitiy": "Illegal Activity",
    "02-HateSpeech": "Hate Speech",
    "03-Malware_Generation": "Malware Generation",
    "04-Physical_Harm": "Physical Harm",
    "05-EconomicHarm": "Economic Harm",
    "06-Fraud": "Fraud",
    "07-Sex": "Sex",
    "08-Political_Lobbying": "Political Lobbying",
    "09-Privacy_Violence": "Privacy Violation / Violence",
    "10-Legal_Opinion": "Legal Opinion",
    "11-Financial_Advice": "Financial Advice",
    "12-Health_Consultation": "Health Consultation",
    "13-Gov_Decision": "Government Decision",
}
_ATTACK_TYPES = ("SD", "SD_TYPO", "TYPO")
_REPHRASED_QUESTION_BY_ATTACK_TYPE = {
    "SD": "Rephrased Question(SD)",
    "SD_TYPO": "Rephrased Question",
    "TYPO": "Rephrased Question",
}


def _clean_text(value: Any) -> str:
    """Normalize one optional text field."""
    if value is None:
        return ""
    return str(value).strip()


@dataset_registry.register("mm_safetybench")
class MMSafetyBenchDataset(MultimodalDatasetAdapter):
    """Load MM-SafetyBench from a local repo/data checkout."""

    def load(self) -> list[NormalizedSample]:
        """Load and normalize MM-SafetyBench from local files."""
        root = self._resolve_root()
        data_root = self._resolve_data_root(root)
        self._source_metadata = {
            "display_name": "MM-SafetyBench",
            "source_uri": "https://github.com/isXinLiu/MM-SafetyBench",
            "upstream_images_uri": (
                "https://drive.google.com/file/d/"
                "1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_/view?usp=sharing"
            ),
            "license": "CC-BY-NC-4.0",
            "languages": ("en",),
            "access_mode": "local_restricted",
            "categories": tuple(_SCENARIOS.values()),
        }
        tiny_ids = self._tiny_id_map(root)
        normalized: list[NormalizedSample] = []
        row_number = 1
        for scenario_key in self._selected_scenarios():
            questions = self._load_questions(data_root, scenario_key)
            allowed_ids = tiny_ids.get(scenario_key)
            for question_id, payload in questions.items():
                if (
                    allowed_ids is not None
                    and int(question_id) not in allowed_ids
                ):
                    continue
                for attack_type in self._selected_attack_types():
                    image_path = self._find_image(
                        data_root=data_root,
                        scenario_key=scenario_key,
                        attack_type=attack_type,
                        question_id=question_id,
                    )
                    image_ref = self.resolve_image(image_path)
                    prompt_text = self._question_text(
                        payload,
                        attack_type=attack_type,
                    )
                    category = _SCENARIOS[scenario_key]
                    normalized.append(
                        NormalizedSample(
                            id=self._make_sample_id(
                                {
                                    "scenario": scenario_key,
                                    "question_id": question_id,
                                    "attack_type": attack_type,
                                    "image_sha256": image_ref.sha256,
                                },
                                row_number,
                            ),
                            dataset=self.config.name,
                            split=self.config.split,
                            messages=[
                                self.build_multimodal_message(
                                    text=prompt_text,
                                    image_ref=image_ref,
                                )
                            ],
                            label=UnsafeLabel(unsafe=True),
                            category_labels=(category,),
                            metadata={
                                "scenario_id": scenario_key,
                                "scenario": category,
                                "question_id": int(question_id),
                                "attack_type": attack_type,
                                **payload,
                            },
                        )
                    )
                    row_number += 1
        return normalized

    def _resolve_root(self) -> Path:
        """Resolve the local MM-SafetyBench root directory."""
        if self.config.path is None:
            raise ValueError("mm_safetybench requires a local path")
        root = Path(self.config.path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"dataset source not found: {root}")
        return root

    def _resolve_data_root(self, root: Path) -> Path:
        """Resolve the directory containing processed questions and images."""
        if (root / "data" / "processed_questions").exists():
            return root / "data"
        if (root / "processed_questions").exists():
            return root
        raise FileNotFoundError(
            "mm_safetybench expects either "
            "'<root>/data/processed_questions' or '<root>/processed_questions'"
        )

    def _selected_scenarios(self) -> tuple[str, ...]:
        """Return the scenarios to include for this run."""
        configured = self.config.options.get("scenarios")
        if configured is None:
            return tuple(_SCENARIOS)
        if isinstance(configured, str):
            raw_items = [item.strip() for item in configured.split(",")]
        elif isinstance(configured, Sequence):
            raw_items = [str(item).strip() for item in configured]
        else:
            raise ValueError("scenarios must be a string or list")

        selected: list[str] = []
        for item in raw_items:
            if not item:
                continue
            if item in _SCENARIOS:
                selected.append(item)
                continue
            matched = [
                key for key, label in _SCENARIOS.items() if label == item
            ]
            if matched:
                selected.extend(matched)
                continue
            raise ValueError(f"unknown MM-SafetyBench scenario: {item}")
        return tuple(dict.fromkeys(selected))

    def _selected_attack_types(self) -> tuple[str, ...]:
        """Return the attack types to include for this run."""
        configured = self.config.options.get("attack_types")
        if configured is None:
            return _ATTACK_TYPES
        if isinstance(configured, str):
            raw_items = [item.strip().upper() for item in configured.split(",")]
        elif isinstance(configured, Sequence):
            raw_items = [str(item).strip().upper() for item in configured]
        else:
            raise ValueError("attack_types must be a string or list")

        selected = []
        for item in raw_items:
            if item not in _ATTACK_TYPES:
                raise ValueError(
                    f"unsupported MM-SafetyBench attack type: {item}"
                )
            selected.append(item)
        return tuple(dict.fromkeys(selected))

    def _tiny_id_map(self, root: Path) -> dict[str, set[int]]:
        """Return one optional scenario -> sampled ids mapping."""
        variant = (
            str(self.config.options.get("variant", "full")).strip().lower()
        )
        if variant != "tiny":
            return {}
        tiny_path_candidates = (
            root / "TinyVersion_ID_List.json",
            root.parent / "TinyVersion_ID_List.json",
        )
        tiny_path = next(
            (path for path in tiny_path_candidates if path.exists()),
            None,
        )
        if tiny_path is None:
            raise FileNotFoundError(
                "variant=tiny requested but TinyVersion_ID_List.json was not found"
            )
        payload = json.loads(tiny_path.read_text(encoding="utf-8"))
        result: dict[str, set[int]] = {}
        for item in payload:
            scenario = str(item["Scenario"])
            sampled_ids = {int(value) for value in item["Sampled_ID_List"]}
            result[scenario] = sampled_ids
        return result

    def _load_questions(
        self,
        data_root: Path,
        scenario_key: str,
    ) -> dict[str, dict[str, Any]]:
        """Load one scenario question file."""
        path = data_root / "processed_questions" / f"{scenario_key}.json"
        if not path.exists():
            raise FileNotFoundError(f"MM-SafetyBench file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"MM-SafetyBench question file must be an object: {path}"
            )
        return {
            str(question_id): dict(question_payload)
            for question_id, question_payload in payload.items()
        }

    def _find_image(
        self,
        *,
        data_root: Path,
        scenario_key: str,
        attack_type: str,
        question_id: str,
    ) -> Path:
        """Resolve one attack image path for one question."""
        base_path = data_root / "imgs" / scenario_key / attack_type
        for extension in (".jpg", ".png", ".jpeg"):
            candidate = base_path / f"{question_id}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"MM-SafetyBench image not found for scenario={scenario_key} "
            f"attack_type={attack_type} question_id={question_id}"
        )

    def _question_text(
        self,
        payload: Mapping[str, Any],
        *,
        attack_type: str,
    ) -> str:
        """Select the prompt text corresponding to one attack type."""
        field_name = _REPHRASED_QUESTION_BY_ATTACK_TYPE[attack_type]
        question_text = _clean_text(payload.get(field_name))
        if question_text:
            return question_text
        for fallback_field in (
            "Changed Question",
            "Question",
        ):
            fallback_text = _clean_text(payload.get(fallback_field))
            if fallback_text:
                return fallback_text
        raise ValueError(
            f"MM-SafetyBench question is missing prompt text for {attack_type}"
        )
