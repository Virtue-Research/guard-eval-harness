"""Local CWE-organized JSON dataset adapter for code
vulnerability detection.

Loads test data from a local directory structured as::

    path/
      c/
        CWE-125.json
        CWE-416.json
      python/
        CWE-79.json
      java/
        CWE-89.json

Each JSON file contains an array of samples with fields:
``code``, ``target`` (0/1), ``cwe`` (list[str]), ``idx``.

This matches the VulnLLM-R local dataset layout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from guard_eval_harness.datasets.base import DatasetAdapter
from guard_eval_harness.datasets.code_vuln.prompts import (
    build_code_vuln_prompt,
    resolve_system_prompt,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import (
    DatasetMetadata,
    NormalizedSample,
)


@dataset_registry.register("code_vuln_local")
class LocalCweJsonDataset(DatasetAdapter):
    """Load CWE-organized JSON files from a local directory."""

    source_suffixes = (".json",)

    def describe(
        self,
        samples: list[NormalizedSample],
    ) -> DatasetMetadata:
        """Declare ``code`` input modality."""
        meta = super().describe(samples)
        metric_eligibility = dict(meta.metric_eligibility)
        metric_eligibility["code_vuln"] = True
        return meta.model_copy(
            update={
                "input_modalities": ("code",),
                "metric_eligibility": metric_eligibility,
            }
        )

    def load(self) -> list[NormalizedSample]:
        """Load and normalize samples from local JSON files."""
        if not self.config.path:
            raise ValueError(
                "code_vuln_local requires an explicit 'path' "
                "pointing to a directory of CWE JSON files"
            )
        base_path = Path(self.config.path)
        if not base_path.is_dir():
            raise ValueError(
                f"code_vuln_local requires a directory path, got: {base_path}"
            )

        language = self.config.options.get("language")
        cwe_filter = self.config.options.get("cwe")
        use_cot = self.config.options.get("use_cot", True)
        use_policy = self.config.options.get("use_policy", True)
        use_cwe_constraint = self.config.options.get(
            "use_cwe_constraint", False
        )
        random_cwe = self.config.options.get("random_cwe", True)
        model_name = self.config.options.get("model_name")
        sys_prompt_key = self.config.options.get("system_prompt")
        system_prompt = resolve_system_prompt(
            model_name=model_name,
            system_prompt_key=sys_prompt_key,
        )

        if cwe_filter and isinstance(cwe_filter, str):
            cwe_filter = [cwe_filter]

        rows: list[dict[str, Any]] = []
        lang_dirs = (
            [base_path / language]
            if language
            else sorted(p for p in base_path.iterdir() if p.is_dir())
        )

        for lang_dir in lang_dirs:
            if not lang_dir.is_dir():
                continue
            lang_name = lang_dir.name

            for json_file in sorted(lang_dir.glob("CWE-*.json")):
                file_cwe = json_file.stem
                if cwe_filter and file_cwe not in cwe_filter:
                    continue

                raw = json.loads(json_file.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    continue

                for item in raw:
                    rows.append(
                        {
                            **item,
                            "_language": lang_name,
                            "_file_cwe": file_cwe,
                        }
                    )

        samples: list[NormalizedSample] = []
        for row_num, row in enumerate(rows):
            if (
                self.execution_limit is not None
                and row_num >= self.execution_limit
            ):
                break

            code = str(row.get("code", ""))
            cwe_ids = _extract_cwe_list(row)
            related = list(row.get("related_cwe") or [])
            prompt_cwes = cwe_ids + [c for c in related if c not in cwe_ids]
            if random_cwe:
                idx = str(row.get("idx", row_num))
                rng = random.Random(idx)
                rng.shuffle(prompt_cwes)

            prompt = build_code_vuln_prompt(
                code=code,
                cwe_list=prompt_cwes if use_policy else None,
                use_cot=use_cot,
                use_policy=use_policy,
                use_cwe_constraint=use_cwe_constraint,
            )

            target = row.get("target", 0)
            unsafe = target in {1, "1", True}
            lang = str(row.get("_language", ""))
            idx = str(row.get("idx", row_num))

            payload = {
                "code": code[:200],
                "cwe": cwe_ids,
                "idx": idx,
                "language": lang,
            }
            sample_id = self._make_sample_id(payload, row_num)

            messages: list[dict[str, str]] = []
            if system_prompt is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )
            messages.append({"role": "user", "content": prompt})

            samples.append(
                NormalizedSample(
                    id=sample_id,
                    dataset=self.config.name,
                    split=self.config.split,
                    messages=messages,
                    label={"unsafe": unsafe},
                    category_labels=tuple(cwe_ids),
                    metadata={
                        "language": lang,
                        "idx": idx,
                    },
                )
            )

        return samples


def _extract_cwe_list(row: dict[str, Any]) -> list[str]:
    """Extract CWE IDs from a dataset row."""
    raw = row.get("CWE_ID") or row.get("cwe") or []
    if isinstance(raw, str):
        return [raw]
    return [str(c) for c in raw]
