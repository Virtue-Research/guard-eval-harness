"""HF-backed VulnLLM-R code vulnerability dataset adapters.

Loads test data from the ``UCSB-SURFI/VulnLLM-R-Test-Data``
Hugging Face dataset, which provides function-level and
repo-level code vulnerability samples across C, Python, and
Java with CWE annotations.

Reference: arXiv:2512.07533
"""

from __future__ import annotations

import random
from typing import Any

from guard_eval_harness.datasets.code_vuln.prompts import (
    build_code_vuln_prompt,
    resolve_system_prompt,
)
from guard_eval_harness.datasets.source_backed import (
    SourceBackedDatasetAdapter,
    load_hf_rows,
)
from guard_eval_harness.registry import dataset_registry
from guard_eval_harness.schemas import (
    DatasetMetadata,
    NormalizedSample,
)


_HF_REPO = "UCSB-SURFI/VulnLLM-R-Test-Data"
_REVISION = "905053e59cf1c2e29b64a33095ef8984f3449190"


class _VulnLLMRBase(SourceBackedDatasetAdapter):
    """Shared logic for VulnLLM-R dataset splits."""

    display_name = "VulnLLM-R"
    source_uri = (
        "https://huggingface.co/datasets/UCSB-SURFI/VulnLLM-R-Test-Data"
    )
    license_name = "MIT"
    languages = ("c", "python", "java")
    categories = ()
    metadata_fields_to_preserve = (
        "language",
        "function_name",
        "original_file",
        "idx",
    )
    label_mapping_note = (
        "target=1 maps to unsafe (vulnerable), target=0 maps to safe (benign)"
    )
    metric_eligibility = {
        **SourceBackedDatasetAdapter.metric_eligibility,
        "code_vuln": True,
    }
    version = _REVISION

    def describe(
        self,
        samples: list[NormalizedSample],
    ) -> DatasetMetadata:
        """Declare ``code`` input modality."""
        meta = super().describe(samples)
        return meta.model_copy(update={"input_modalities": ("code",)})

    # Subclasses set this to the HF split name.
    _hf_split: str = "function_level"

    def load_source_rows(self) -> list[dict[str, object]]:
        """Load rows from the HF dataset."""
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

        rows = load_hf_rows(
            _HF_REPO,
            split=self._hf_split,
            revision=_REVISION,
            limit=self.execution_limit,
        )

        result: list[dict[str, object]] = []
        for row in rows:
            row_lang = str(row.get("language", "")).lower()
            if language and row_lang != str(language).lower():
                continue

            cwe_ids: list[str] = _extract_cwe_list(row)
            if cwe_filter:
                filter_set = (
                    {cwe_filter}
                    if isinstance(cwe_filter, str)
                    else set(cwe_filter)
                )
                if not any(c in filter_set for c in cwe_ids):
                    continue

            related_cwes: list[str] = list(row.get("RELATED_CWE") or [])
            prompt_cwes = cwe_ids + [
                c for c in related_cwes if c not in cwe_ids
            ]
            if random_cwe:
                # Seed per sample so the shuffle is
                # deterministic and reproducible across runs.
                idx = str(row.get("idx", ""))
                rng = random.Random(idx)
                rng.shuffle(prompt_cwes)

            code = str(row.get("code", ""))
            has_stack_trace = bool(row.get("stack_trace", False))
            prompt = build_code_vuln_prompt(
                code=code,
                cwe_list=prompt_cwes if use_policy else None,
                use_cot=use_cot,
                use_policy=use_policy,
                use_cwe_constraint=use_cwe_constraint,
                has_stack_trace=has_stack_trace,
            )

            target = row.get("target", 0)
            unsafe = target in {1, "1", True}

            messages: list[dict[str, str]] = []
            if system_prompt is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )
            messages.append({"role": "user", "content": prompt})

            raw_idx = str(row.get("idx", ""))

            result.append(
                {
                    "messages": messages,
                    "unsafe": unsafe,
                    "category_labels": tuple(cwe_ids),
                    "language": row_lang,
                    "function_name": row.get("function_name", ""),
                    "original_file": row.get("original_file", ""),
                    "idx": raw_idx,
                }
            )

        return result


def _extract_cwe_list(row: dict[str, Any]) -> list[str]:
    """Extract CWE IDs from a dataset row."""
    raw = row.get("CWE_ID") or row.get("cwe") or []
    if isinstance(raw, str):
        return [raw]
    return [str(c) for c in raw]


@dataset_registry.register("vulnllm_r_function_level")
class VulnLLMRFunctionLevel(_VulnLLMRBase):
    """VulnLLM-R function-level vulnerability dataset."""

    display_name = "VulnLLM-R Function Level"
    _hf_split = "function_level"
    supported_splits = ("function_level", "test")


@dataset_registry.register("vulnllm_r_repo_level")
class VulnLLMRRepoLevel(_VulnLLMRBase):
    """VulnLLM-R repo-level vulnerability dataset."""

    display_name = "VulnLLM-R Repo Level"
    _hf_split = "repo_level"
    supported_splits = ("repo_level", "test")


@dataset_registry.register("vulnllm_r_application")
class VulnLLMRApplication(_VulnLLMRBase):
    """VulnLLM-R application-level vulnerability dataset."""

    display_name = "VulnLLM-R Application Level"
    _hf_split = "application"
    supported_splits = ("application", "test")
