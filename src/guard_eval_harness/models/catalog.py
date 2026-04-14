"""Model catalog — YAML-backed known-good profiles for safety models.

Each YAML file in the ``catalog/`` directory defines one model profile.
The loader scans the directory at import time and builds a flat lookup
dict keyed by the file stem (e.g. ``llama-guard-3-8b.yaml`` becomes
slug ``llama-guard-3-8b``).

Prompt templates can be referenced by name via ``prompt_template: $md_judge``
syntax.  The ``$`` prefix triggers a lookup in :mod:`models.templates`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from guard_eval_harness.models import templates as _tpl

_log = logging.getLogger(__name__)

_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"

_TEMPLATE_MAP: dict[str, str] = {
    "$md_judge": _tpl.MD_JUDGE_PROMPT,
    "$shieldgemma": _tpl.SHIELDGEMMA_PROMPT,
    "$api_judge": _tpl.API_JUDGE_PROMPT,
    "$wildguard": _tpl.WILDGUARD_PROMPT,
    "$mistral_plus": _tpl.MISTRAL_PLUS_PROMPT,
    "$llama_guard_taxonomy": _tpl.LLAMA_GUARD_TAXONOMY_PROMPT,
}

# Tasks that can run on vLLM (generative models only).
_VLLM_COMPATIBLE_TASKS = {"text-generation"}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A reusable model configuration profile."""

    name: str
    adapter: str
    model_name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    catalog_name: str | None = None


def _resolve_templates(obj: Any) -> Any:
    """Recursively replace ``$template_name`` strings."""
    if isinstance(obj, str) and obj in _TEMPLATE_MAP:
        return _TEMPLATE_MAP[obj]
    if isinstance(obj, dict):
        return {k: _resolve_templates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_templates(v) for v in obj]
    return obj


def _load_catalog() -> dict[str, ModelProfile]:
    """Scan the catalog directory and build the lookup dict.

    Raises ``RuntimeError`` if ``catalog/`` is missing or contains no
    loadable entries. That situation almost always means the package
    was built without the catalog/*.yaml data files, which would make
    ``geh list models`` and catalog-based ``--model`` resolution
    silently return zero entries. Failing loudly at import time
    surfaces the packaging bug instead of silently degrading the CLI.
    """
    catalog: dict[str, ModelProfile] = {}
    if not _CATALOG_DIR.is_dir():
        raise RuntimeError(
            "Model catalog directory is missing: "
            f"{_CATALOG_DIR}. This usually means the package was "
            "built without the catalog/*.yaml data files — check "
            "[tool.setuptools.package-data] in pyproject.toml."
        )
    for path in sorted(_CATALOG_DIR.glob("*.yaml")):
        slug = path.stem
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            args = _resolve_templates(raw.get("args", {}))
            catalog[slug] = ModelProfile(
                name=slug,
                adapter=raw["adapter"],
                model_name=raw.get("model_name"),
                args=args,
                catalog_name=slug,
            )
        except Exception:
            _log.warning("Failed to load catalog entry: %s", path, exc_info=True)
    if not catalog:
        raise RuntimeError(
            "Model catalog is empty: no *.yaml files were loaded from "
            f"{_CATALOG_DIR}. This usually means the package was built "
            "without the catalog/*.yaml data files — check "
            "[tool.setuptools.package-data] in pyproject.toml."
        )
    return catalog


MODEL_CATALOG: dict[str, ModelProfile] = _load_catalog()


def _to_vllm_args(hf_args: dict[str, Any]) -> dict[str, Any]:
    """Translate HF catalog args to vLLM-compatible args."""
    vllm_args: dict[str, Any] = {}
    # Copy shared args
    for key in (
        "apply_chat_template",
        "chat_template_profile",
        "prompt_template",
        "generated_text_line_index",
        "text_score_mapping",
    ):
        if key in hf_args:
            vllm_args[key] = hf_args[key]
    # Translate generation_kwargs to top-level vLLM args
    gen_kwargs = hf_args.get("generation_kwargs", {})
    max_new_tokens = gen_kwargs.get("max_new_tokens", 16)
    vllm_args["max_new_tokens"] = max_new_tokens
    vllm_args["temperature"] = 0
    return vllm_args


_BACKEND_OVERRIDE_SUPPORTED_SOURCES = {"hf"}


def resolve_catalog(
    slug: str,
    *,
    backend: str | None = None,
) -> ModelProfile | None:
    """Look up a model by catalog slug, optionally overriding the backend.

    Parameters
    ----------
    slug:
        Human-friendly model name (e.g. ``llama-guard-3-8b``).
    backend:
        Optional backend override (``hf`` or ``vllm``). Backend
        overrides are only valid for HF catalog entries; API-backed
        entries (anthropic, openai_compatible) cannot be rehosted on
        a local inference engine by swapping adapter names.

    Returns
    -------
    ModelProfile or None if the slug is not in the catalog.

    Raises
    ------
    ValueError
        If ``--backend`` is requested for an API-backed catalog entry,
        or if ``--backend vllm`` is requested for an HF model whose
        task is not supported by vLLM.
    """
    profile = MODEL_CATALOG.get(slug)
    if profile is None:
        return None

    if backend is None or backend == profile.adapter:
        return profile

    if profile.adapter not in _BACKEND_OVERRIDE_SUPPORTED_SOURCES:
        raise ValueError(
            f"Model '{slug}' uses adapter '{profile.adapter}' and "
            "cannot be rehosted via --backend. Remove --backend or "
            "pick a local (hf) catalog entry."
        )

    if backend == "vllm":
        task = profile.args.get("task", "text-classification")
        if task not in _VLLM_COMPATIBLE_TASKS:
            raise ValueError(
                f"Model '{slug}' uses task '{task}' which is not "
                f"supported by vLLM. Use --backend hf instead."
            )
        vllm_args = _to_vllm_args(profile.args)
        return ModelProfile(
            name=profile.name,
            adapter="vllm",
            model_name=profile.model_name,
            args=vllm_args,
            catalog_name=profile.catalog_name,
        )

    if backend == "hf":
        return ModelProfile(
            name=profile.name,
            adapter="hf",
            model_name=profile.model_name,
            args=dict(profile.args),
            catalog_name=profile.catalog_name,
        )

    raise ValueError(
        f"Unsupported --backend '{backend}'. Use 'hf' or 'vllm'."
    )


def list_catalog() -> list[dict[str, str]]:
    """Return catalog entries for CLI listing."""
    return [
        {
            "slug": slug,
            "adapter": profile.adapter,
            "model_name": profile.model_name or "",
        }
        for slug, profile in sorted(MODEL_CATALOG.items())
    ]
