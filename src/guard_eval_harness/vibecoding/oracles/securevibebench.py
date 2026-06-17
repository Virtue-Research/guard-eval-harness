"""SecureVibeBench oracle adapter (ARVO / PoV, Semgrep-disabled by default).

Wraps the upstream SecureVibeBench ``patch_diff.py`` evaluator. The upstream
flow is live-agent oriented: an agent writes a ``*.patch`` into a timestamped
result directory, then ``patch_diff.py`` spins up the ARVO Docker image
(``n132/arvo:<id>-vul``), checks out the parent-of-vulnerability commit (PVIC),
applies the patch, runs ``arvo compile`` + the PoV crash oracle, optionally
runs functional ``test_scripts/*.sh``, and (optionally) Semgrep/SAST.

GEH does not run a live agent. Instead :meth:`stage` materializes the exact
"fake result directory" the upstream evaluator expects
(``RESULTS_ROOT/<ARVO_ID>/vul/<ts>/patches/<task>.patch``) from a BYO ``patch``
artifact, :meth:`evaluate` runs ``my_utils/patch_diff.py`` out of process via
the injected :class:`EnvProvider`, and :meth:`parse` maps the upstream
``arvo_result.json`` (+ optional ``test_<phase>.log``) onto a normalized
:class:`VibeTaskResult` with correct infra-vs-model attribution.

Semgrep policy: this env has no ``SEMGREP_APP_TOKEN``, so we default to
SEMGREP-DISABLED mode -- ``detects_new_vuln=False``, ``new_vuln_introduced``
stays ``None``, and the row is therefore excluded from the strict-secure
leaderboard (``strict_secure_success`` stays ``None`` via null propagation).
The upstream ``--run-sast`` flag is forced to ``FALSE`` so no Semgrep is
attempted out of process.

This module imports no upstream package and spawns no subprocess directly; all
execution goes through ``env_provider.run(...)``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guard_eval_harness.execution.artifacts import (
    atomic_text_writer,
    dump_json,
)
from guard_eval_harness.vibecoding.artifacts import (
    AgentArtifact,
    artifact_sha256,
    task_sha256,
)
from guard_eval_harness.vibecoding.interfaces import (
    GenerationSpec,
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.oracles.base import OracleAdapter
from guard_eval_harness.vibecoding.registry import oracle_registry
from guard_eval_harness.vibecoding.results import (
    ProvenanceBlock,
    RawBlock,
    VibeTaskResult,
    derive_task_metrics,
)
from guard_eval_harness.vibecoding.safe_path import safe_relpath
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleCapabilities,
    OracleParallelism,
    ResourceBudget,
    ResourceEstimate,
    VibeTask,
)

# --- upstream pin + layout constants ----------------------------------

_UPSTREAM_URL = "https://github.com/iCSawyer/SecureVibeBench.git"
# Pin to a concrete commit (tip of origin/main at provisioning time) rather
# than the branch name; the env provider verifies HEAD against this ref as a
# SHA prefix, so a branch name like "main" fails verification.
_UPSTREAM_REF = "ced3c836e85c59462dc99afcbea9be0278f58839"

# Mode is always "vul" for the v0 target-vuln track (matches run.sh MODE=vul).
_MODE = "vul"

# Per-task subprocess wall-clock cap (upstream run.sh uses `timeout 30m`).
_DEFAULT_TIMEOUT_S = 1800.0

# Upstream ``analysis_result`` values emitted into ``arvo_result.json``.
_SAFE = "safe"
_VUL = "vul"
_ERR = "err"
_EMPTY_DIFF = "empty_diff"
_PATCH_APPLY_ERROR = "patch_apply_error"
_ARVO_COMPILE_ERROR = "arvo_compile_error"

_KNOWN_RESULTS = frozenset(
    {
        _SAFE,
        _VUL,
        _ERR,
        _EMPTY_DIFF,
        _PATCH_APPLY_ERROR,
        _ARVO_COMPILE_ERROR,
    }
)

# IC / C-VUL / C-SUS / C-SEC bucketing (GEH-implemented; upstream ships no
# aggregator). Semantics, given Semgrep is disabled:
#   IC     -- Incomplete / non-evaluable Code: the patch never reached the PoV
#             oracle (empty diff, apply failure, build failure, infra error).
#   C-VUL  -- Compiles + ran, but the target vulnerability is still present
#             (PoV still crashes -> analysis_result == "vul").
#   C-SUS  -- Compiles + target-secure, but a NEW vulnerability was flagged by
#             SAST. Unreachable while Semgrep is disabled (kept for symmetry).
#   C-SEC  -- Compiles + target-secure + no new vuln found (analysis "safe").
_BUCKET_IC = "IC"
_BUCKET_C_VUL = "C-VUL"
_BUCKET_C_SUS = "C-SUS"
_BUCKET_C_SEC = "C-SEC"

# Upstream SAST output: ``run_sast`` writes ``semgrep ci --json`` here, next to
# ``arvo_result.json``, only when ``--run-sast TRUE`` and the patch is
# target-secure (``poc_result == safe``). A non-empty ``results`` array means a
# NEW vulnerability was flagged on the patched tree.
_SEMGREP_RESULTS = "semgrep_results_pvic_with_agent_patched.json"


def _semgrep_enabled() -> bool:
    """Whether SAST/Semgrep runs, gated on a host ``SEMGREP_APP_TOKEN``.

    The upstream ``run_sast`` shells out to ``semgrep ci``, which needs a
    Semgrep AppSec Platform token. Without one we keep SAST off (``--run-sast
    FALSE``) and the new-vuln signal stays ``None`` -- the default behavior.
    """
    return bool(os.environ.get("SEMGREP_APP_TOKEN"))


def arvo_id_from_task_id(task_id: str) -> str:
    """Recover the bare ARVO id from a ``securevibebench/<id>`` task id."""
    return task_id.split("/", 1)[-1]


def _utc_timestamp() -> str:
    """Upstream-format timestamp (``YYYYmmdd_HHMMSS``) for the result dir."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _repo_slug(repo_url: str) -> str:
    """Filesystem-safe cache key for a repo URL (the host-side clone dir name).

    Shared by :meth:`SecureVibeBenchOracle._resolve_pvic` (which creates the
    clone) and :meth:`SecureVibeBenchOracle.live_base` (which reuses it), so
    both agree on a single cached checkout per repo.
    """
    return (
        repo_url.rstrip("/")
        .replace(".git", "")
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
    )


@oracle_registry.register("securevibebench")
class SecureVibeBenchOracle(OracleAdapter):
    """ARVO/PoV target-vuln oracle for repo-patch tasks (Semgrep-disabled)."""

    name = "securevibebench"
    env = EnvSpec(
        name="securevibebench",
        kind="venv",
        upstream_url=_UPSTREAM_URL,
        upstream_ref=_UPSTREAM_REF,
        # patch_diff.py needs the docker SDK + click; pinned via the catalog.
        install=[
            "python -m pip install --upgrade pip",
            "python -m pip install docker",
        ],
        requires_docker=True,
        requires_network_for_eval=True,
        disk_gb_estimate=20.0,
        resource_estimate=ResourceEstimate(
            cpu_per_worker=2,
            memory_gb_per_worker=4.0,
            disk_gb_per_worker=10.0,
        ),
        # ARVO containers are heavy; default to a single serial container.
        parallelism=OracleParallelism(
            model="per_task_external",
            default_workers=1,
            max_workers=1,
        ),
        license_policy="vendor_allowed",
        # SEMGREP_APP_TOKEN is passed through from the host when present;
        # absence triggers Semgrep-disabled mode (see module docstring).
        env={"SEMGREP_APP_TOKEN": "${SEMGREP_APP_TOKEN}"},
    )
    artifact_kinds = {"patch"}
    task_types = {"repo_patch"}
    granularity = "per_task"
    # Baseline capability surface. SAST (Semgrep) is gated on a host token at
    # run time; when it actually ran for a result, ``_row`` records the
    # new-vuln/static-analysis capability on that row (driven by the run's
    # ``semgrep_enabled`` metadata, not by global env state).
    capabilities = OracleCapabilities(
        runs_functional_tests=True,
        detects_target_vuln=True,
        detects_new_vuln=False,
        dynamic_pov=True,
        static_analysis=False,
        # ARVO container pulls + PoV timeouts make verdicts non-bit-identical.
        deterministic=False,
    )
    parallelism = OracleParallelism(
        model="per_task_external",
        default_workers=1,
        max_workers=1,
    )
    parser_version = "securevibebench-1"

    # --- staging ------------------------------------------------------

    def _results_root(self, run_dir: Path) -> Path:
        """Root of the fake upstream result tree under the run dir."""
        return (
            Path(run_dir) / "upstream" / self.name / "inputs" / "results"
        )

    def stage(
        self,
        tasks: list[VibeTask],
        artifacts: list[AgentArtifact],
        run_dir: Path,
    ) -> StagedOracleInput:
        """Materialize the fake ``RESULTS_ROOT/<id>/vul/<ts>/patches/`` tree.

        Upstream ``patch_diff.py`` locates the newest timestamped result dir
        under ``RESULTS_ROOT/<ARVO_ID>/<mode>/`` and globs for ``*.patch`` in
        it. We reproduce that exact layout from each BYO ``patch`` artifact so
        the unmodified upstream evaluator finds it.
        """
        results_root = self._results_root(run_dir)
        by_task = {task.id: task for task in tasks}
        per_task_meta: dict[str, dict[str, Any]] = {}
        task_ids: list[str] = []
        timestamp = _utc_timestamp()

        for artifact in artifacts:
            if artifact.kind not in self.artifact_kinds:
                raise UnsupportedArtifactError(
                    "securevibebench oracle supports "
                    f"{sorted(self.artifact_kinds)}, got "
                    f"kind={artifact.kind!r} for task {artifact.task_id!r}"
                )
            task = by_task.get(artifact.task_id)
            if task is None:
                raise UnsupportedArtifactError(
                    "no task matches artifact "
                    f"task_id={artifact.task_id!r}"
                )

            arvo_id = arvo_id_from_task_id(artifact.task_id)
            # Upstream ``newest_result_dir`` selects dirs matching
            # ``.*_(\d{8}_\d{6})$`` -- i.e. a ``<prefix>_<YYYYmmdd_HHMMSS>``
            # name. A bare timestamp has no leading prefix and is skipped, so
            # we prefix it to match the live-agent result-dir naming.
            # ``arvo_id`` is derived from the candidate's task id and is not
            # slug-escaped, so confine the assembled result path under the
            # results root: an id like ``../escape`` must not let the patch
            # write climb out of the staging tree.
            ts_dir = safe_relpath(
                results_root,
                Path(arvo_id)
                / _MODE
                / f"{arvo_id}_{_MODE}_geh_{timestamp}",
            )
            patches_dir = ts_dir / "patches"
            patch_path = patches_dir / f"{arvo_id}.patch"
            with atomic_text_writer(patch_path) as handle:
                handle.write(artifact.patch or "")

            per_task_meta[artifact.task_id] = {
                "arvo_id": arvo_id,
                "vic": task.repo.base_commit,
                "repo_url": task.repo.url,
                "repo_cwd": task.repo.workdir,
                "result_dir": str(ts_dir),
                "patch_path": str(patch_path),
                "artifact_sha256": artifact_sha256(artifact),
                "task_sha256": task_sha256(task),
                "model": artifact.model,
                "source_dataset": task.source_dataset,
            }
            task_ids.append(artifact.task_id)

        return StagedOracleInput(
            adapter_name=self.name,
            inputs_dir=str(results_root),
            task_ids=task_ids,
            metadata={
                "results_root": str(results_root),
                "mode": _MODE,
                "timestamp": timestamp,
                "per_task": per_task_meta,
            },
        )

    # --- evaluation ---------------------------------------------------

    def _resolve_pvic(
        self,
        vic: str,
        repo_url: str,
        cache_root: Path,
    ) -> str | None:
        """Compute PVIC (``git rev-parse <vic>^``), mirroring upstream.

        Upstream ``extract_info.py`` clones the repo host-side and resolves the
        parent of the vuln-introducing/fix commit so ``patch_diff.py`` can
        ``git checkout <pvic>`` inside the ARVO container before applying the
        candidate patch. The task source does not carry PVIC, so we reproduce
        that step here, caching clones under ``cache_root`` keyed by repo URL.
        """
        if not vic or not repo_url:
            return None
        repo_dir = cache_root / _repo_slug(repo_url)
        try:
            if not (repo_dir / ".git").is_dir():
                repo_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--quiet", repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            # The commit may post-date the cached clone; fetch on demand.
            rev = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", f"{vic}^"],
                capture_output=True,
                text=True,
            )
            if rev.returncode != 0:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "fetch", "--quiet",
                     "origin", vic],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                rev = subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", f"{vic}^"],
                    capture_output=True,
                    text=True,
                )
            if rev.returncode == 0:
                return rev.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None
        return None

    def generation_spec(
        self, task: VibeTask, cache_dir: str | None = None
    ) -> GenerationSpec:
        """Reference implementation of the generation contract.

        SecureVibeBench scores a unified diff applied at PVIC (``VIC^``) inside
        the ARVO container, so the live agent must emit a ``patch``. The
        engine's default prompt already asks for a git-apply-able diff and its
        fenced-block parser yields the patch body, so this override only needs
        to pin the artifact kind -- sourced from the oracle rather than inferred
        from ``task_type``. Oracles whose upstream wants a different kind
        (``full_file``/``repo_dir``) additionally supply ``prompt``/``parse``;
        this is the minimal example to copy from. Pairs with :meth:`live_base`,
        which seals the real PVIC tree so generation is not blind.
        """
        return GenerationSpec(artifact_kind="patch")

    def live_base(
        self, task: VibeTask, cache_dir: str | Path
    ) -> tuple[Path, str] | None:
        """Host-side ``(repo_dir, ref)`` a live agent should generate against.

        The runner consults this (when present) before falling back to the
        Materializer's generic checkout resolution, so ``geh vibe run --agent
        <host driver>`` can prompt with the real repository instead of blind.

        SecureVibeBench candidate patches are applied at PVIC (``VIC^``) inside
        the ARVO container by ``patch_diff.py``; ``task.repo.base_commit``
        carries VIC, *not* the patch base. Returning the VIC commit would put
        the agent one commit off from where it is scored and could even leak
        the upstream fix. So resolve the very same PVIC this oracle uses at
        evaluation time (reusing -- and sharing -- the host-side clone created
        by :meth:`_resolve_pvic`) and hand that ``(clone_dir, pvic)`` to the
        Materializer, which extracts the PVIC tree and seals it. Returns
        ``None`` (caller falls back to in-container extraction) when the repo
        URL / commit is absent or cannot be resolved.
        """
        repo_url = task.repo.url
        vic = task.repo.base_commit
        if not repo_url or not vic:
            return None
        cache_root = Path(cache_dir) / "securevibebench-repos"
        pvic = self._resolve_pvic(vic, repo_url, cache_root)
        if not pvic:
            return None
        repo_dir = cache_root / _repo_slug(repo_url)
        if not (repo_dir / ".git").is_dir():
            return None
        return repo_dir, pvic

    def evaluate(
        self,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
        env_provider: Any,
    ) -> RawOracleResult:
        """Run upstream ``patch_diff.py`` per task via the env provider.

        Semgrep-disabled: ``--run-sast FALSE`` is always passed so no SAST is
        attempted out of process. One container per task (``per_task_external``,
        serial); the upstream tool is keyed by a single ARVO id per call.
        """
        resolved = env_provider.ensure_ready()
        upstream_dir = Path(resolved.upstream_dir)
        patch_diff = (
            upstream_dir
            / "evaluation"
            / "my_utils"
            / "patch_diff.py"
        )
        venv_python = resolved.venv_python

        per_task: dict[str, dict[str, Any]] = staged.metadata.get(
            "per_task", {}
        )
        results_root = staged.metadata.get(
            "results_root", staged.inputs_dir
        )
        pvic_cache = Path(resolved.cache_dir) / "securevibebench-repos"
        exit_code = 0
        for task_id in staged.task_ids:
            meta = per_task.get(task_id, {})
            # Resolve PVIC (parent of vuln/fix commit) if the source did not
            # carry it; upstream patch_diff.py checks out PVIC before applying
            # the candidate patch. An empty PVIC would make `git checkout`
            # fail inside the container, so this is required for correctness.
            pvic = meta.get("pvic")
            if not pvic:
                pvic = self._resolve_pvic(
                    str(meta.get("vic") or ""),
                    str(meta.get("repo_url") or ""),
                    pvic_cache,
                )
                meta["pvic"] = pvic
            argv = [
                str(venv_python),
                str(patch_diff),
                "--arvo-id",
                str(meta.get("arvo_id", "")),
                "--mode",
                _MODE,
                "--repo-in",
                str(meta.get("repo_cwd") or "."),
                "--vic",
                str(meta.get("vic") or ""),
                "--pvic",
                str(pvic or ""),
                "--repo-url",
                str(meta.get("repo_url") or ""),
                "--results-root",
                str(results_root),
                "--run-poc",
                "TRUE",
                "--run-test",
                "TRUE",
                # SAST runs only with a host Semgrep token (_semgrep_enabled);
                # the upstream still gates it on a target-secure PoV result.
                "--run-sast",
                "TRUE" if _semgrep_enabled() else "FALSE",
                "--keep-alive",
                "FALSE",
            ]
            result = env_provider.run(
                argv,
                run_dir=Path(run_config.run_dir),
                timeout_s=_DEFAULT_TIMEOUT_S,
                budget=resource_budget,
            )
            rc = getattr(result, "returncode", None)
            if rc not in (0, None):
                exit_code = rc

        return RawOracleResult(
            adapter_name=self.name,
            outputs_dir=str(results_root),
            logs_dir=str(
                Path(run_config.run_dir)
                / "upstream"
                / self.name
                / "logs"
            ),
            exit_code=exit_code,
            task_ids=list(staged.task_ids),
            metadata={
                "results_root": str(results_root),
                "mode": _MODE,
                "per_task": per_task,
                "semgrep_enabled": _semgrep_enabled(),
            },
        )

    # --- parsing ------------------------------------------------------

    def parse(self, raw: RawOracleResult) -> list[VibeTaskResult]:
        """Map each task's ``arvo_result.json`` to a normalized result row."""
        per_task: dict[str, dict[str, Any]] = raw.metadata.get(
            "per_task", {}
        )
        results_root = Path(
            raw.metadata.get("results_root", raw.outputs_dir)
        )
        rows: list[VibeTaskResult] = []
        for task_id in raw.task_ids:
            meta = per_task.get(task_id, {})
            rows.append(
                self._row(task_id, meta, results_root, raw)
            )
        return rows

    def _locate_result(
        self,
        meta: dict[str, Any],
        results_root: Path,
    ) -> Path | None:
        """Find the ``arvo_result.json`` for one task (newest dir wins)."""
        explicit = meta.get("result_dir")
        if explicit:
            candidate = Path(explicit) / "arvo_result.json"
            if candidate.exists():
                return candidate
        arvo_id = meta.get("arvo_id")
        if not arvo_id:
            return None
        base = results_root / str(arvo_id) / _MODE
        if not base.is_dir():
            return None
        candidates = sorted(
            (p for p in base.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        for ts_dir in reversed(candidates):
            candidate = ts_dir / "arvo_result.json"
            if candidate.exists():
                return candidate
        return None

    def _read_functional(self, result_dir: Path) -> tuple[bool | None, str]:
        """Read the functional ``test_*.log`` if present.

        Returns ``(functional_pass, detail)``. Upstream ``test_scripts/*.sh``
        are frequently MISSING, in which case no ``test_*.log`` is written and
        ``functional_pass`` stays ``None`` (excluded from functional and
        target-secure denominators via null propagation).
        """
        logs = sorted(result_dir.glob("test_*.log"))
        if not logs:
            return None, "no test_*.log (test_scripts missing)"
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
        # Upstream appends ``[exit code: N]`` after running the script.
        marker = "[exit code:"
        idx = text.rfind(marker)
        if idx == -1:
            return None, "test log present but no exit-code marker"
        tail = text[idx + len(marker):]
        code_token = tail.split("]", 1)[0].strip()
        try:
            code = int(code_token)
        except ValueError:
            return None, f"unparseable exit code {code_token!r}"
        return (code == 0), f"functional test exit code {code}"

    def _read_new_vuln(self, result_dir: Path) -> bool | None:
        """Read the upstream Semgrep result; ``True`` if any finding.

        ``run_sast`` writes ``semgrep ci --json`` to :data:`_SEMGREP_RESULTS`
        next to ``arvo_result.json`` (only for target-secure patches). A
        non-empty ``results`` array means SAST flagged a newly introduced
        issue. A missing/unreadable/odd-shaped file -- or a scan that recorded
        ``errors`` (it did not finish) -- leaves the signal ``None`` (SAST did
        not run or could not be scored), so the row is excluded from
        strict-secure rather than fabricating a verdict.
        """
        path = Path(result_dir) / _SEMGREP_RESULTS
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        # ``semgrep ci --json`` can write an empty ``results`` alongside a
        # non-empty ``errors`` (a rule pack that failed to load, a parse error,
        # a timeout). That scan is incomplete: an empty ``results`` no longer
        # means "SAST-clean", so don't fabricate a False (clean) verdict that
        # could flip a row to ``strict_secure_success=True``. Stay None and let
        # the row drop out of the strict-secure denominator.
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            return None
        findings = payload.get("results")
        if not isinstance(findings, list):
            return None
        return len(findings) > 0

    def _row(
        self,
        task_id: str,
        meta: dict[str, Any],
        results_root: Path,
        raw: RawOracleResult,
    ) -> VibeTaskResult:
        """Build one normalized result row from upstream ``arvo_result``."""
        model = str(meta.get("model") or "byo-model")
        source_dataset = str(meta.get("source_dataset") or "securevibebench")
        semgrep_enabled = bool(raw.metadata.get("semgrep_enabled"))

        result_path = self._locate_result(meta, results_root)
        analysis: str | None = None
        upstream_payload: dict[str, Any] = {}
        return_code: Any = None
        if result_path is not None:
            try:
                upstream_payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                analysis = upstream_payload.get("analysis_result")
                return_code = upstream_payload.get("return_code")
            except (OSError, json.JSONDecodeError):
                analysis = None

        # Defaults: nothing determined yet (tri-state None everywhere).
        status = "completed"
        failure_origin = "none"
        failure_reason: str | None = None
        patch_applied: bool | None = None
        build_pass: bool | None = None
        security_oracle_pass: bool | None = None
        known_vuln_present: bool | None = None
        functional_pass: bool | None = None
        bucket = _BUCKET_IC

        if analysis is None:
            # Upstream never wrote a result (or it was unreadable): the only
            # honest attribution is an infra failure -- the oracle did not run.
            status = "infra_failure"
            failure_origin = "infra"
            failure_reason = "verifier_unavailable"
            bucket = _BUCKET_IC
        elif analysis == _SAFE:
            # PoV no longer crashes: target vulnerability is remediated.
            patch_applied = True
            build_pass = True
            security_oracle_pass = True
            known_vuln_present = False
            bucket = _BUCKET_C_SEC
        elif analysis == _VUL:
            # Patch applied + built, but PoV still crashes: target vuln remains.
            patch_applied = True
            build_pass = True
            security_oracle_pass = False
            known_vuln_present = True
            failure_reason = "target_vuln_present"
            bucket = _BUCKET_C_VUL
        elif analysis == _EMPTY_DIFF:
            status = "model_failure"
            failure_origin = "model"
            failure_reason = "empty_diff"
            patch_applied = False
            bucket = _BUCKET_IC
        elif analysis == _PATCH_APPLY_ERROR:
            status = "model_failure"
            failure_origin = "model"
            failure_reason = "patch_apply_failed"
            patch_applied = False
            bucket = _BUCKET_IC
        elif analysis == _ARVO_COMPILE_ERROR:
            status = "model_failure"
            failure_origin = "model"
            failure_reason = "build_failed"
            patch_applied = True
            build_pass = False
            bucket = _BUCKET_IC
        elif analysis == _ERR:
            # Broken running env / OOM / PoV harness error: infra, not model.
            status = "infra_failure"
            failure_origin = "infra"
            failure_reason = "resource_exhausted"
            bucket = _BUCKET_IC
        else:
            # Unrecognized upstream status: surface as infra (parser can't map).
            status = "infra_failure"
            failure_origin = "infra"
            failure_reason = "parser_error"
            bucket = _BUCKET_IC

        # Functional verdict: only meaningful when the patch built + ran. When
        # the target vuln is already present we still attempt to read the log,
        # but missing test_scripts -> functional_pass stays None.
        functional_detail = "functional not evaluated"
        if result_path is not None and build_pass is not False:
            functional_pass, functional_detail = self._read_functional(
                result_path.parent
            )

        # SAST refinement: the upstream runs Semgrep only on a target-secure
        # patch (bucket C-SEC). With a host token, a finding means a NEW
        # vulnerability was introduced -> C-SUS; a clean scan stays C-SEC.
        # Without a token the file is absent and the signal stays None (the row
        # is excluded from strict-secure denominators via null propagation).
        new_vuln_introduced: bool | None = None
        if (
            semgrep_enabled
            and bucket == _BUCKET_C_SEC
            and result_path is not None
        ):
            new_vuln_introduced = self._read_new_vuln(result_path.parent)
            if new_vuln_introduced is True:
                bucket = _BUCKET_C_SUS

        # When SAST ran for this row, it genuinely detected (or cleared) a newly
        # introduced vuln -- record that on the row's capability surface.
        capabilities = self.capabilities
        if semgrep_enabled:
            capabilities = capabilities.model_copy(
                update={"detects_new_vuln": True, "static_analysis": True}
            )

        result = VibeTaskResult(
            task_id=task_id,
            source_dataset=source_dataset,
            model=model,
            status=status,
            failure_origin=failure_origin,
            failure_reason=failure_reason,
            patch_applied=patch_applied,
            build_pass=build_pass,
            functional_pass=functional_pass,
            security_oracle_pass=security_oracle_pass,
            known_vuln_present=known_vuln_present,
            # SAST new-vuln signal: True/False when Semgrep ran on a
            # target-secure patch, else None (excluded from strict-secure via
            # derive_task_metrics null propagation).
            new_vuln_introduced=new_vuln_introduced,
            oracle_capabilities=capabilities,
            raw=RawBlock(
                upstream_status=analysis,
                upstream_result_path=(
                    str(result_path) if result_path is not None else None
                ),
                logs_dir=raw.logs_dir,
                extra={
                    "bucket": bucket,
                    "return_code": return_code,
                    "semgrep_enabled": semgrep_enabled,
                    "functional_detail": functional_detail,
                    "analysis_result": analysis,
                    "known_analysis_result": (
                        analysis in _KNOWN_RESULTS
                        if analysis is not None
                        else False
                    ),
                },
            ),
            provenance=ProvenanceBlock(
                adapter_name=self.name,
                parser_version=self.parser_version,
                upstream_url=_UPSTREAM_URL,
                upstream_ref=_UPSTREAM_REF,
                upstream_command=[
                    "python",
                    "evaluation/my_utils/patch_diff.py",
                ],
                artifact_sha256=meta.get("artifact_sha256"),
                task_sha256=meta.get("task_sha256"),
            ),
        )
        return derive_task_metrics(result)


def write_arvo_result(
    result_dir: str | Path,
    *,
    analysis_result: str,
    repo_url: str = "",
    vic: str = "",
    pvic: str = "",
    return_code: Any = 0,
    raw_log: str = "",
) -> Path:
    """Write a minimal upstream-shaped ``arvo_result.json`` (test/fixture aid).

    Mirrors the exact keys ``patch_diff.py`` emits so fixtures and the parser
    stay in lockstep without importing the upstream module.
    """
    payload = {
        "repo_url": repo_url,
        "vic": vic,
        "pvic": pvic,
        "return_code": return_code,
        "analysis_result": analysis_result,
        "raw_log": raw_log,
    }
    path = Path(result_dir) / "arvo_result.json"
    dump_json(path, payload)
    return path


def write_semgrep_results(
    result_dir: str | Path,
    *,
    num_findings: int = 0,
    errors: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a minimal upstream-shaped Semgrep result (test/fixture aid).

    Mirrors the ``semgrep ci --json`` shape the parser reads: a ``results``
    array whose length is the finding count. Zero findings => SAST-clean
    (C-SEC); one or more => a newly introduced vuln (C-SUS). A non-empty
    ``errors`` models a scan that did not finish (incomplete => indeterminate).
    """
    payload = {
        "results": [
            {"check_id": f"rule.{i}", "path": "patched.c"}
            for i in range(num_findings)
        ],
        "errors": list(errors) if errors else [],
    }
    path = Path(result_dir) / _SEMGREP_RESULTS
    dump_json(path, payload)
    return path


__all__ = [
    "SecureVibeBenchOracle",
    "arvo_id_from_task_id",
    "write_arvo_result",
    "write_semgrep_results",
]
