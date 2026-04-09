"""Named benchmark packs for one-command quickstart runs.

Each pack is a versioned, curated collection of datasets targeting a
specific safety evaluation concern.  Packs are the primary user-facing
grouping — ``geh run --pack core --model llama-guard-3-8b`` is the
30-second quickstart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guard_eval_harness.benchmarks.presets import PresetDataset


@dataclass(frozen=True, slots=True)
class PackManifest:
    """A versioned, named benchmark pack."""

    name: str
    version: str
    description: str
    datasets: tuple[PresetDataset, ...]
    expected_sample_counts: dict[str, int] = field(
        default_factory=dict,
    )
    recommended_thresholds: dict[str, float] = field(
        default_factory=dict,
    )

    @property
    def qualified_name(self) -> str:
        """Return ``name-version`` for artifact paths."""
        return f"{self.name}-{self.version}"

    def dataset_names(self) -> list[str]:
        """Return the ordered list of dataset names in this pack."""
        return [d.name for d in self.datasets]

    def as_manifest_dict(self) -> dict[str, Any]:
        """Render a JSON-friendly manifest payload."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dataset_count": len(self.datasets),
            "datasets": self.dataset_names(),
            "expected_sample_counts": (
                dict(self.expected_sample_counts)
                if self.expected_sample_counts
                else None
            ),
            "recommended_thresholds": (
                dict(self.recommended_thresholds)
                if self.recommended_thresholds
                else None
            ),
        }


# ------------------------------------------------------------------
# Pack definitions
# ------------------------------------------------------------------

_CORE_V1 = PackManifest(
    name="core",
    version="v1",
    description=(
        "Broad safety benchmark covering jailbreak, toxicity, hate "
        "speech, and harmful content.  Based on the validated core-12 "
        "dataset selection from the 21x31 reproduction wave."
    ),
    datasets=(
        PresetDataset(
            name="advbench_behaviors",
            adapter="advbench_behaviors",
        ),
        PresetDataset(
            name="beaver_tails_330k",
            adapter="beaver_tails_330k",
        ),
        PresetDataset(
            name="convabuse",
            adapter="convabuse",
        ),
        PresetDataset(
            name="do_not_answer",
            adapter="do_not_answer",
        ),
        PresetDataset(
            name="harmful_qa",
            adapter="harmful_qa",
        ),
        PresetDataset(
            name="harmbench_behaviors",
            adapter="harmbench_behaviors",
        ),
        PresetDataset(
            name="hatemoji_check",
            adapter="hatemoji_check",
        ),
        PresetDataset(
            name="jbb_behaviors",
            adapter="jbb_behaviors",
        ),
        PresetDataset(
            name="pku_safe_rlhf",
            adapter="pku_safe_rlhf",
        ),
        PresetDataset(
            name="strong_reject_instructions",
            adapter="strong_reject_instructions",
        ),
        PresetDataset(
            name="toxic_chat",
            adapter="toxic_chat",
        ),
        PresetDataset(
            name="xstest",
            adapter="xstest",
        ),
    ),
)

_JAILBREAK_V1 = PackManifest(
    name="jailbreak",
    version="v1",
    description=(
        "Adversarial jailbreak and prompt-attack benchmarks.  Tests a "
        "model's resistance to instruction-following exploits."
    ),
    datasets=(
        PresetDataset(
            name="advbench_behaviors",
            adapter="advbench_behaviors",
        ),
        PresetDataset(
            name="advbench_strings",
            adapter="advbench_strings",
        ),
        PresetDataset(
            name="do_anything_now_questions",
            adapter="do_anything_now_questions",
        ),
        PresetDataset(
            name="guardrailsai_jailbreak",
            adapter="guardrailsai_jailbreak",
            split="train",
        ),
        PresetDataset(
            name="harmbench_behaviors",
            adapter="harmbench_behaviors",
        ),
        PresetDataset(
            name="jailbreakbench",
            adapter="jailbreakbench",
            split="harmful",
        ),
        PresetDataset(
            name="jbb_behaviors",
            adapter="jbb_behaviors",
        ),
        PresetDataset(
            name="malicious_instruct",
            adapter="malicious_instruct",
        ),
        PresetDataset(
            name="strong_reject_instructions",
            adapter="strong_reject_instructions",
        ),
        PresetDataset(
            name="wildjailbreak",
            adapter="wildjailbreak",
            split="train",
        ),
    ),
)

_TOXICITY_V1 = PackManifest(
    name="toxicity",
    version="v1",
    description=(
        "Toxicity detection benchmarks spanning comments, prompts, "
        "and dialog.  Evaluates a model's ability to flag toxic "
        "language across domains."
    ),
    datasets=(
        PresetDataset(
            name="civil_comments",
            adapter="civil_comments",
            split="validation",
        ),
        PresetDataset(
            name="jigsaw_toxicity",
            adapter="jigsaw_toxicity",
        ),
        PresetDataset(
            name="olid",
            adapter="olid",
        ),
        PresetDataset(
            name="or_bench",
            adapter="or_bench",
            split="train",
        ),
        PresetDataset(
            name="real_toxicity_prompts",
            adapter="real_toxicity_prompts",
            split="train",
        ),
        PresetDataset(
            name="toxic_chat",
            adapter="toxic_chat",
        ),
        PresetDataset(
            name="toxigen",
            adapter="toxigen",
        ),
    ),
)

_HATE_HARASSMENT_V1 = PackManifest(
    name="hate_harassment",
    version="v1",
    description=(
        "Hate speech, harassment, and bias detection benchmarks.  "
        "Covers explicit hate, implicit bias, emoji-based hate, and "
        "conversational abuse."
    ),
    datasets=(
        PresetDataset(
            name="convabuse",
            adapter="convabuse",
        ),
        PresetDataset(
            name="dynahate",
            adapter="dynahate",
        ),
        PresetDataset(
            name="ethos",
            adapter="ethos",
            split="train",
        ),
        PresetDataset(
            name="hate_speech_offensive",
            adapter="hate_speech_offensive",
            split="train",
        ),
        PresetDataset(
            name="hatecheck",
            adapter="hatecheck",
        ),
        PresetDataset(
            name="hatemoji_check",
            adapter="hatemoji_check",
        ),
        PresetDataset(
            name="hatexplain",
            adapter="hatexplain",
        ),
        PresetDataset(
            name="implicit_hate",
            adapter="implicit_hate",
            split="train",
        ),
        PresetDataset(
            name="measuring_hate_speech",
            adapter="measuring_hate_speech",
            split="train",
        ),
        PresetDataset(
            name="social_bias_frames",
            adapter="social_bias_frames",
            split="train",
        ),
        PresetDataset(
            name="tweet_eval_hate",
            adapter="tweet_eval_hate",
        ),
    ),
)

_CODE_VULN_V1 = PackManifest(
    name="code_vuln",
    version="v1",
    description=(
        "Code vulnerability detection benchmarks from the "
        "VulnLLM-R evaluation suite.  Tests a model's ability "
        "to identify vulnerable code across C, Python, and "
        "Java with CWE-level categorization."
    ),
    datasets=(
        PresetDataset(
            name="vulnllm_r_function_level_c",
            adapter="vulnllm_r_function_level",
            split="function_level",
            options={"language": "c"},
        ),
        PresetDataset(
            name="vulnllm_r_function_level_python",
            adapter="vulnllm_r_function_level",
            split="function_level",
            options={"language": "python"},
        ),
        PresetDataset(
            name="vulnllm_r_function_level_java",
            adapter="vulnllm_r_function_level",
            split="function_level",
            options={"language": "java"},
        ),
        PresetDataset(
            name="vulnllm_r_repo_level",
            adapter="vulnllm_r_repo_level",
            split="repo_level",
            options={"language": "java"},
        ),
    ),
)

_PROMPT_INJECTION_V1 = PackManifest(
    name="prompt_injection",
    version="v1",
    description=(
        "Prompt injection and adversarial instruction benchmarks.  "
        "Tests a model's resistance to crafted inputs that attempt "
        "to override safety guidelines."
    ),
    datasets=(
        PresetDataset(
            name="advbench_behaviors",
            adapter="advbench_behaviors",
        ),
        PresetDataset(
            name="advbench_strings",
            adapter="advbench_strings",
        ),
        PresetDataset(
            name="hex_phi",
            adapter="hex_phi",
            split="Category_1_Illegal_Activity",
        ),
        PresetDataset(
            name="i_malicious_instructions",
            adapter="i_malicious_instructions",
        ),
        PresetDataset(
            name="mitre",
            adapter="mitre",
        ),
        PresetDataset(
            name="tdc_red_teaming",
            adapter="tdc_red_teaming",
        ),
    ),
)

_AUDIO_V1 = PackManifest(
    name="audio",
    version="v1",
    description=(
        "Native-audio safety benchmark for models that operate directly "
        "on audio instead of an ASR pipeline."
    ),
    datasets=(
        PresetDataset(
            name="nemotron_content_safety_audio",
            adapter="nemotron_content_safety_audio",
        ),
    ),
)

# ------------------------------------------------------------------
# Pack registry
# ------------------------------------------------------------------

_PACKS: dict[str, PackManifest] = {
    "audio": _AUDIO_V1,
    "audio-v1": _AUDIO_V1,
    "core": _CORE_V1,
    "core-v1": _CORE_V1,
    "jailbreak": _JAILBREAK_V1,
    "jailbreak-v1": _JAILBREAK_V1,
    "toxicity": _TOXICITY_V1,
    "toxicity-v1": _TOXICITY_V1,
    "hate_harassment": _HATE_HARASSMENT_V1,
    "hate_harassment-v1": _HATE_HARASSMENT_V1,
    "prompt_injection": _PROMPT_INJECTION_V1,
    "prompt_injection-v1": _PROMPT_INJECTION_V1,
    "code_vuln": _CODE_VULN_V1,
    "code_vuln-v1": _CODE_VULN_V1,
}


def list_packs() -> list[str]:
    """Return canonical pack names (without version aliases)."""
    return sorted(
        {pack.name for pack in _PACKS.values()}
    )


def list_pack_versions(name: str) -> list[str]:
    """Return available versions for a pack name."""
    versions = [
        p.version
        for p in _PACKS.values()
        if p.name == name
    ]
    if not versions:
        available = ", ".join(list_packs())
        raise ValueError(
            f"Unknown pack {name!r}. Available: {available}"
        )
    return sorted(set(versions))


def get_pack(name: str) -> PackManifest:
    """Return one benchmark pack by name or qualified alias."""
    try:
        return _PACKS[name]
    except KeyError as exc:
        available = ", ".join(list_packs())
        raise ValueError(
            f"Unknown pack {name!r}. Available: {available}"
        ) from exc


def build_pack_run_payload(
    *,
    pack: PackManifest,
    model_adapter: str,
    model_name: str | None = None,
    model_args: dict[str, Any] | None = None,
    threshold: float = 0.5,
    batch_size: int = 1,
    output_root: str = "out/packs",
) -> dict[str, Any]:
    """Build a full run config payload from a pack + model spec.

    This is the bridge between ``--pack`` / ``--model`` CLI flags
    and the ``load_config`` → ``run_benchmark`` pipeline.
    """
    from guard_eval_harness.benchmarks.presets import (
        _slugify,
    )

    model_slug = _slugify(model_name or model_adapter)
    run_dir = (
        f"{output_root}/{pack.qualified_name}/{model_slug}"
    )

    return {
        "version": 1,
        "run_name": f"pack-{pack.name}-{model_slug}",
        "threshold": threshold,
        "model": {
            "adapter": model_adapter,
            "model_name": model_name,
            "args": dict(model_args or {}),
        },
        "datasets": [
            d.as_config() for d in pack.datasets
        ],
        "execution": {
            "batch_size": batch_size,
            "concurrency": 1,
            "retries": 0,
        },
        "output": {
            "run_dir": run_dir,
            "overwrite": False,
        },
        "metadata": {
            "pack": pack.name,
            "pack_version": pack.version,
        },
    }
