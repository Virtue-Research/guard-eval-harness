"""SecCodeBench task source (function-level secure code generation).

Loads the upstream SecCodeBench benchmark files
(``datasets/benchmark/<lang>/<lang>.json``) into normalized
:class:`VibeTask` records of type ``project_scaffold``. Each upstream entry is
a function-level secure-coding task: the model must implement (``gen``) or fix
(``fix``) a function described by a natural-language prompt, and a long-running
language VERIFIER SERVICE then runs the project's functional unit tests, a PoC
(security) test, and optional domain LLM judges.

Upstream layout (one file per language, a dict keyed by case id):

    {
      "SSRFUrllib": {
        "language": "python",
        "FuncTester": "UnitTester",
        "SecTester": "UnitTester",
        "template": "2_1_0/SSRFUrllib",
        "prompt": "2_1_0/SSRFUrllib",
        "scenarios": ["fix", "fix-hints", "gen", "gen-hints"],
        "params": {"social_media_scraper.py": "src/.../social_media_scraper.py"},
        "severity": "high",
        "locale": "en-US",
        "verify_urls": {"gen": "http://python-verifier:5000/verify/...", ...},
        "remote_prompt_paths": {"gen": "python_bench/.../generate.md", ...}
      },
      ...
    }

There are four scenarios (``gen`` / ``gen-hints`` / ``fix`` / ``fix-hints``);
v0 keeps to the base ``gen`` scenario (documented on the oracle). The natural
language gen prompt (``prompts/<ver>/<case>.<locale>``) carries the functional
requirements + interface (description / function_signature / module_name); the
loader joins it onto the task ``instructions`` when present. CWE labels are not
stored as a separate field upstream, so we derive a coarse CWE from the case id
(e.g. ``SQLInjection*`` -> CWE-89) for audit-friendly ``labels.cwe``.

The default checkout is overridable via ``dataset_path`` so tests run against a
tiny fixture JSON with no Docker/network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guard_eval_harness.vibecoding.envs import EnvProvider
from guard_eval_harness.vibecoding.interfaces import TaskSource
from guard_eval_harness.vibecoding.registry import task_source_registry
from guard_eval_harness.vibecoding.schema import (
    RepoSpec,
    TaskEnvironmentRef,
    TaskLabels,
    VibeTask,
)

# v0 evaluates the base "gen" scenario only (gen-hints/fix/fix-hints deferred).
_BASE_SCENARIO = "gen"

# Languages SecCodeBench ships (one benchmark JSON per language).
_LANGUAGES = ("python", "cpp", "go", "java", "nodejs")

# The canonical benchmark filename differs from the language name only for C++.
_BENCHMARK_FILENAME = {
    "python": "python.json",
    "cpp": "c.json",
    "go": "go.json",
    "java": "java.json",
    "nodejs": "nodejs.json",
}

# Coarse, audit-friendly CWE labels keyed by the leading vuln token in the case
# id (e.g. "SSRFUrllib" -> "SSRF" -> CWE-918). The precise CWE lives upstream in
# the prompt/testcase; this mapping is only for ``labels.cwe`` and is not
# load-bearing for scoring.
_VULN_TOKEN_TO_CWE = {
    "SSRF": "CWE-918",
    "SSTI": "CWE-1336",
    "SQLInjection": "CWE-89",
    "CommandInjection": "CWE-78",
    "CodeInjection": "CWE-94",
    "Deserialization": "CWE-502",
    "PathTraversal": "CWE-22",
    "XSS": "CWE-79",
    "XXE": "CWE-611",
    "OpenRedirect": "CWE-601",
}


def _cwe_for_case_id(case_id: str) -> list[str]:
    """Best-effort CWE label from the leading vuln token in a case id."""
    for token, cwe in _VULN_TOKEN_TO_CWE.items():
        if case_id.startswith(token):
            return [cwe]
    return []


class _UnreadablePrompt(Exception):
    """Internal marker: the gen prompt could not be read (non-fatal)."""


@task_source_registry.register("seccodebench")
class SecCodeBenchTaskSource(TaskSource):
    """Yield ``project_scaffold`` tasks from SecCodeBench benchmark files."""

    name = "seccodebench"

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = None,
        languages: list[str] | None = None,
    ) -> None:
        """Optionally override the dataset root / language set (fixtures).

        ``dataset_path`` may point either at the upstream checkout root (which
        contains ``datasets/benchmark/<lang>/<lang>.json``) or directly at a
        single benchmark JSON file (used by the tiny test fixture).
        """
        self._dataset_path = (
            Path(dataset_path) if dataset_path is not None else None
        )
        self._languages = list(languages) if languages else list(_LANGUAGES)

    def _root(self, cache_dir: str | Path | None = None) -> Path:
        """Explicit override, else the env provider's .geh checkout."""
        if self._dataset_path is not None:
            return self._dataset_path
        from guard_eval_harness.vibecoding.oracles.seccodebench import (
            SecCodeBenchOracle,
        )

        provider = EnvProvider(SecCodeBenchOracle.env, cache_dir=cache_dir)
        return Path(provider.resolve().upstream_dir)

    def _benchmark_files(
        self, cache_dir: str | Path | None = None
    ) -> list[tuple[str, Path]]:
        """Resolve ``(language, benchmark_json_path)`` pairs to load.

        Supports three shapes of ``dataset_path``:
          * a single ``<lang>.json`` file (language inferred from contents),
          * a directory of ``<lang>.json`` files,
          * the upstream checkout root (``datasets/benchmark/<lang>/...``).
        """
        root = self._root(cache_dir)
        if root.is_file():
            return [(self._infer_language(root), root)]

        pairs: list[tuple[str, Path]] = []
        for lang in self._languages:
            filename = _BENCHMARK_FILENAME.get(lang, f"{lang}.json")
            # Directory-of-benchmarks layout (fixtures) takes precedence.
            flat = root / filename
            nested = root / "datasets" / "benchmark" / lang / filename
            if flat.is_file():
                pairs.append((lang, flat))
            elif nested.is_file():
                pairs.append((lang, nested))
        return pairs

    @staticmethod
    def _infer_language(path: Path) -> str:
        """Infer the language for a standalone benchmark JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "unknown"
        if isinstance(data, dict):
            for entry in data.values():
                if isinstance(entry, dict) and entry.get("language"):
                    return str(entry["language"])
        # Fall back to the file stem (e.g. "c" -> "cpp").
        stem = path.stem
        return "cpp" if stem == "c" else stem

    def load(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[VibeTask]:
        """Return normalized ``project_scaffold`` tasks (base ``gen`` only)."""
        tasks: list[VibeTask] = []
        for language, path in self._benchmark_files(cache_dir):
            for task in self._load_file(language, path):
                tasks.append(task)
                if limit is not None and len(tasks) >= max(0, int(limit)):
                    return tasks
        return tasks

    def _load_file(self, language: str, path: Path) -> list[VibeTask]:
        """Parse one benchmark JSON file into ``VibeTask`` records."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        tasks: list[VibeTask] = []
        for case_id, entry in data.items():
            if not isinstance(entry, dict):
                continue
            task = self._build_task(language, str(case_id), entry, path)
            if task is not None:
                tasks.append(task)
        return tasks

    def _build_task(
        self,
        language: str,
        case_id: str,
        entry: dict[str, Any],
        benchmark_path: Path,
    ) -> VibeTask | None:
        """Map one upstream entry to a normalized ``VibeTask``.

        We only emit a task when the base ``gen`` scenario is present (v0).
        """
        scenarios = entry.get("scenarios") or []
        if _BASE_SCENARIO not in scenarios:
            return None
        entry_language = str(entry.get("language") or language)
        instructions = self._instructions(
            case_id, entry, benchmark_path, entry_language
        )
        return VibeTask(
            id=f"seccodebench/{entry_language}__{case_id}",
            source_dataset="seccodebench",
            task_type="project_scaffold",
            instructions=instructions,
            repo=RepoSpec(workdir="."),
            labels=TaskLabels(cwe=_cwe_for_case_id(case_id)),
            environment=TaskEnvironmentRef(
                oracle="seccodebench",
                requires_docker=True,
            ),
        )

    def _instructions(
        self,
        case_id: str,
        entry: dict[str, Any],
        benchmark_path: Path,
        language: str,
    ) -> str:
        """Build the task instructions (FunctionalRequirements + Interface).

        Prefers the upstream gen-scenario prompt file (which carries the
        description / function_signature / module_name) when resolvable; falls
        back to a synthesized one-liner naming the target file.
        """
        prompt = self._read_gen_prompt(entry, benchmark_path, language)
        if prompt:
            return prompt
        target = self._target_path(entry)
        return (
            f"Implement {case_id} for project '{case_id}'. Produce the "
            f"complete, secure implementation of {target} so the functional "
            "unit tests pass and the security testcase finds no vulnerability."
        )

    def _read_gen_prompt(
        self,
        entry: dict[str, Any],
        benchmark_path: Path,
        language: str,
    ) -> str:
        """Read the upstream gen prompt file if it can be located.

        The ``prompt`` field is a ``<version>/<case>`` reference resolved under
        ``<benchmark_dir>/prompts/<version>/<case>.<locale>``. Missing files are
        non-fatal (returns "" so the caller synthesizes instructions).
        """
        ref = entry.get("prompt")
        if not isinstance(ref, str) or not ref:
            return ""
        locale = str(entry.get("locale") or "en-US")
        prompts_dir = benchmark_path.parent / "prompts"
        candidate = prompts_dir / f"{ref}.{locale}"
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return ""

    @staticmethod
    def _target_path(entry: dict[str, Any]) -> str:
        """Return the single target file path the gen scenario must write."""
        params = entry.get("params")
        if isinstance(params, dict) and params:
            return str(next(iter(params.values())))
        return "the target source file"
