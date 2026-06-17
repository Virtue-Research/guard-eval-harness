"""VibeRunner: the integration join point for the vibecoding subsystem.

The runner wires together the Stage-A contracts and the Stage-B helpers:

    ensure_vibe_registrations
        -> TaskSource.load
        -> resolve artifacts (patch_eval / BYO from a predictions list, or
           live_agent: drive a registered agent driver over each task, framing
           generation with the oracle's GenerationSpec)
        -> ArtifactAdapter.validate (incompatible kind -> ``unsupported`` row)
        -> Materializer.prepare (only when the oracle needs ``repo_dir``)
        -> stage / evaluate / parse
             * batch_internal / granularity=batch: ONE call over the whole
               compatible list (never fan out per task -- avoids
               double-scheduling Docker)
             * per_task_external / serial: iterate per task
        -> cache check (skip on hit; cache only completed / model_failure)
        -> derive_task_metrics each row
        -> merge unsupported / infra rows
        -> compute_vibe_metrics
        -> append_result / write_results

Infrastructure failures raised by ``evaluate`` never crash the run: they are
mapped to ``infra_failure`` / ``failure_origin=infra`` rows (one per expected
task) so they stay out of the model-score denominators.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import Field

from guard_eval_harness.execution.artifacts import sha256_payload
from guard_eval_harness.vibecoding.artifacts import (
    TELEMETRY_METADATA_KEY,
    AgentArtifact,
    artifact_scoring_sha256,
    artifact_sha256,
    task_sha256,
)
from guard_eval_harness.vibecoding.cache import (
    CacheKey,
    OracleResultCache,
    resolve_cache_dir,
)
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
    UnsupportedArtifactError,
)
from guard_eval_harness.vibecoding.materialize import (
    MaterializeError,
    Materializer,
)
from guard_eval_harness.vibecoding.metrics import compute_vibe_metrics
from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    oracle_registry,
    task_source_registry,
)
from guard_eval_harness.vibecoding.results import (
    ProvenanceBlock,
    RawBlock,
    VibeTaskResult,
    derive_task_metrics,
)
from guard_eval_harness.vibecoding.run_store import (
    ensure_vibe_run_layout,
    write_agent_artifact,
    write_manifest,
    write_results,
    write_tasks,
)
from guard_eval_harness.vibecoding.sandbox.anti_cheat import (
    DEFAULT_POLICY_ID,
)
from guard_eval_harness.vibecoding.schema import (
    OracleCapabilities,
    ResourceBudget,
    VibeModel,
    VibeTask,
)

if TYPE_CHECKING:
    from guard_eval_harness.vibecoding.agents.base import AgentResult

_log = logging.getLogger(__name__)

# Anti-cheat is wired in Stage E; until then the runner records a stable
# placeholder policy id so cache keys and provenance stay deterministic. The
# token is single-sourced from sandbox.anti_cheat.DEFAULT_POLICY_ID so the
# detectors' policy identity and the runner's cache keys can never drift.
_DEFAULT_ANTI_CHEAT_POLICY = DEFAULT_POLICY_ID

# Statuses whose rows are safe to cache (environment-independent). Infra and
# unsupported rows must always be retried, so they are never cached.
_CACHEABLE_STATUSES = frozenset({"completed", "model_failure"})

# Cap on how many missing task ids are spelled out in the reconciliation
# warning before it summarizes the remainder as ``(+K more)``.
_MISSING_ID_LOG_CAP = 20


class VibeRunResult(VibeModel):
    """Outcome of one :meth:`VibeRunner.run`.

    Carries the loaded tasks, the normalized result rows (scored + merged
    unsupported / infra rows), and the capability-scoped metrics summary, plus
    pointers to the on-disk run dir.
    """

    run_id: str = Field(min_length=1)
    run_dir: str
    source: str = Field(min_length=1)
    oracle: str = Field(min_length=1)
    tasks: list[VibeTask] = Field(default_factory=list)
    results: list[VibeTaskResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class _RunInputs:
    """Resolved (task, artifact) pairs split into compatible/unsupported.

    ``pairs`` are the (task, artifact) the oracle can actually score;
    ``unsupported`` holds pre-built ``unsupported`` result rows for artifacts
    whose kind the oracle rejects (and for predictions with no matching task).
    """

    __slots__ = ("pairs", "unsupported")

    def __init__(self) -> None:
        self.pairs: list[tuple[VibeTask, AgentArtifact]] = []
        self.unsupported: list[VibeTaskResult] = []


def _coerce_artifact(
    entry: AgentArtifact | dict[str, Any],
) -> AgentArtifact:
    """Build an :class:`AgentArtifact` from a prediction entry."""
    if isinstance(entry, AgentArtifact):
        return entry
    return AgentArtifact.model_validate(entry)


def _fold_agent_telemetry(result: AgentResult) -> AgentArtifact:
    """Carry an :class:`AgentResult`'s telemetry into its artifact metadata.

    Only ``result.artifact`` flows through the rest of the live pipeline, so
    the sibling bookkeeping fields -- token counts, ``cost_usd``, the resolved
    model, and any driver ``metadata`` (e.g. ``error``/``raw_text``) -- would
    otherwise be dropped right after generation, leaving live runs with no
    cost or provenance signal. Fold them into ``artifact.metadata`` so they
    survive into the persisted ``artifact.json`` (and thus budgets/reports).

    Per-generation telemetry (tokens/cost/resolved model) is nested under the
    reserved ``TELEMETRY_METADATA_KEY`` so the scoring cache key, which hashes
    the artifact via :func:`artifact_scoring_sha256`, deliberately excludes it:
    two runs that produce the same candidate must score-identically regardless
    of usage/cost. Driver ``metadata`` (``error``/``raw_text``/...) is merged
    at the top level -- it is part of the candidate's identity, exactly as it
    already was for empty-result artifacts -- and last so a driver keeps the
    final say on any key it sets.
    """
    art = result.artifact
    telemetry = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "agent_model": result.model,
    }
    return art.model_copy(
        update={
            "metadata": {
                **art.metadata,
                **result.metadata,
                TELEMETRY_METADATA_KEY: telemetry,
            }
        }
    )


def _unsupported_row(
    task: VibeTask | None,
    artifact: AgentArtifact,
    *,
    detail: str,
) -> VibeTaskResult:
    """A normalized ``unsupported`` row for an incompatible artifact."""
    source = task.source_dataset if task is not None else artifact.task_id
    return VibeTaskResult(
        task_id=artifact.task_id,
        source_dataset=source,
        model=artifact.model,
        status="unsupported",
        failure_origin="adapter",
        failure_reason="unsupported_artifact",
        raw=RawBlock(upstream_status="unsupported", extra={"detail": detail}),
    )


def _infra_row(
    task: VibeTask,
    artifact: AgentArtifact,
    *,
    detail: str,
    trial_index: int,
    random_seed: int | None,
) -> VibeTaskResult:
    """A normalized ``infra_failure`` row attributed to infra."""
    return VibeTaskResult(
        task_id=task.id,
        source_dataset=task.source_dataset,
        model=artifact.model,
        status="infra_failure",
        failure_origin="infra",
        failure_reason="resource_exhausted",
        trial_index=trial_index,
        random_seed=random_seed,
        raw=RawBlock(
            upstream_status="infra_failure", extra={"detail": detail}
        ),
        provenance=ProvenanceBlock(
            artifact_sha256=artifact_sha256(artifact),
            task_sha256=task_sha256(task),
        ),
    )


def _unscored_row(
    task: VibeTask,
    artifact: AgentArtifact,
) -> VibeTaskResult:
    """A row for a candidate the adapter produced no distinct result for.

    Happens when several candidate artifacts share a ``task_id`` in one batch
    but the upstream evaluator keys by instance id and so collapses them to a
    single result. The surplus candidate is preserved (never silently
    dropped) but excluded from denominators; multi-run/trial scoring should
    use one run per trial (distinct ``run_id``/seed).
    """
    return VibeTaskResult(
        task_id=task.id,
        source_dataset=task.source_dataset,
        model=artifact.model,
        status="unsupported",
        failure_origin="adapter",
        failure_reason="unsupported_artifact",
        raw=RawBlock(
            upstream_status="no_distinct_result",
            extra={
                "detail": (
                    "duplicate candidate for this task_id in a single batch; "
                    "the upstream evaluator returned no distinct result. Score "
                    "trials in separate runs (distinct run_id/seed)."
                )
            },
        ),
        provenance=ProvenanceBlock(
            artifact_sha256=artifact_sha256(artifact),
            task_sha256=task_sha256(task),
        ),
    )


def _no_submission_row(
    task: VibeTask,
    oracle: Any,
    *,
    model: str,
) -> VibeTaskResult:
    """An in-denominator non-submission failure for a task with no prediction.

    A loaded task whose ``id`` matched no candidate artifact produced no scored
    row at all. Every upstream benchmark counts a missing submission as an
    in-denominator FAIL (e.g. SusVibes' ``no_patch`` rows are part of its
    186-instance ``correct_ratio`` denominator; SecRepoBench / SecCodeBench
    likewise score a non-submission as not-correct & not-secure), so we mirror
    that here rather than letting the task silently vanish from every
    denominator. ``status='model_failure'`` is an in-denominator terminal
    status (see ``metrics.in_denominator`` / ``DENOMINATOR_STATUSES``), and the
    functional/security gates are a definite ``False`` (not ``None``): nothing
    was submitted, so the candidate is definitely not-correct and not-secure.
    ``known_vuln_present`` / ``new_vuln_introduced`` stay ``None`` because with
    no applied patch there is nothing to assess. The row is run through
    :func:`derive_task_metrics` so ``target_secure_success`` /
    ``strict_secure_success`` are set by the same Kleene path as every other
    row (a ``False`` gate short-circuits to a definite ``False``).
    """
    result = VibeTaskResult(
        task_id=task.id,
        source_dataset=task.source_dataset,
        model=model or "none",
        status="model_failure",
        failure_origin="model",
        failure_reason="no_submission",
        patch_applied=False,
        build_pass=False,
        functional_pass=False,
        security_oracle_pass=False,
        known_vuln_present=None,
        new_vuln_introduced=None,
        oracle_capabilities=getattr(
            oracle, "capabilities", OracleCapabilities()
        ),
        raw=RawBlock(
            upstream_status="no_submission",
            extra={
                "detail": (
                    "loaded task had no matching prediction; scored as an "
                    "in-denominator non-submission fail"
                )
            },
        ),
        provenance=ProvenanceBlock(task_sha256=task_sha256(task)),
    )
    return derive_task_metrics(result)


class VibeRunner:
    """Orchestrates a single patch-evaluation / BYO vibecoding run."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._cache_dir = resolve_cache_dir(cache_dir)

    def run(
        self,
        source_name: str,
        oracle_name: str,
        *,
        predictions: Sequence[AgentArtifact | dict[str, Any]] | None = None,
        artifacts: Sequence[AgentArtifact] | None = None,
        split: str | None = None,
        limit: int | None = None,
        resource_budget: ResourceBudget | None = None,
        run_config: OracleRunConfig | None = None,
        env_provider: Any = None,
        run_dir: str | Path | None = None,
        no_cache: bool = False,
        allow_empty: bool = False,
        mode: str = "patch_eval",
        agent: str | None = None,
        agent_model: str | None = None,
    ) -> VibeRunResult:
        """Evaluate predictions for ``source_name`` via ``oracle_name``.

        Two modes share the same scoring path:

        - ``mode='patch_eval'`` (default): score the BYO ``predictions`` (or
          already-typed ``artifacts``).
        - ``mode='live_agent'``: drive the registered ``agent`` driver over
          each loaded task to produce the artifacts, then score them. The
          generated artifacts are the only difference; classification,
          materialization, oracle stage/evaluate/parse, caching, and metrics
          are identical to ``patch_eval``.

        A source that loads zero tasks fails loudly (a missing upstream
        checkout should not masquerade as a no-op run) unless
        ``allow_empty=True``.
        """
        if mode not in ("patch_eval", "live_agent"):
            raise ValueError(
                f"unknown run mode {mode!r} (expected 'patch_eval' or "
                "'live_agent')"
            )

        ensure_vibe_registrations()
        source = task_source_registry.get(source_name)()
        oracle = oracle_registry.get(oracle_name)()

        # Thread the run's cache dir into task loading so a source that
        # resolves its checkout under .geh uses the same dir as the oracle's
        # env provider (matters for `geh vibe eval --cache-dir <dir>`).
        tasks = source.load(
            split=split, limit=limit, cache_dir=self._cache_dir
        )
        if not tasks and not allow_empty:
            raise ValueError(
                f"source {source_name!r} loaded 0 tasks (split={split!r}, "
                f"limit={limit!r}). The upstream checkout is likely missing "
                f"or empty -- run `geh vibe acquire --dataset {source_name}` "
                f"to clone + build it, or pass allow_empty=True for an "
                f"explicit empty run."
            )
        if run_config is None:
            run_config = OracleRunConfig(
                run_id="vibe-run",
                run_dir=str(run_dir) if run_dir is not None else "",
            )
        resolved_run_dir = self._resolve_run_dir(run_dir, run_config)
        run_config = run_config.model_copy(
            update={"run_dir": str(resolved_run_dir), "no_cache": no_cache}
        )
        ensure_vibe_run_layout(resolved_run_dir)

        # Resolve the artifacts to score. live_agent drives a registered agent
        # over each task (handing it a sealed base checkout when one exists);
        # patch_eval scores the BYO list. Both feed the same pipeline below.
        if mode == "live_agent":
            artifact_list = self._generate_live(
                tasks,
                oracle=oracle,
                agent=agent,
                agent_model=agent_model,
                run_dir=resolved_run_dir,
            )
        else:
            artifact_list = self._resolve_artifacts(predictions, artifacts)

        budget = resource_budget or self._default_budget(oracle)

        inputs = self._classify_artifacts(oracle, tasks, artifact_list)
        for _task, artifact in inputs.pairs:
            write_agent_artifact(resolved_run_dir, artifact)

        content_hashes = self._materialize_if_needed(
            oracle, inputs, resolved_run_dir
        )

        scored_rows = self._evaluate(
            oracle,
            inputs.pairs,
            run_config=run_config,
            budget=budget,
            env_provider=env_provider,
            no_cache=no_cache,
            content_hashes=content_hashes,
        )

        results = scored_rows + inputs.unsupported

        # Coverage reconciliation: a LOADED task whose id matched no candidate
        # artifact produces NO row above, yet compute_vibe_metrics keys its
        # denominators on the RESULTS list (n_total = len(results)); the
        # ``tasks`` arg only feeds the by_cwe/by_dataset breakdowns. So a task
        # absent from the predictions would silently shrink every denominator.
        # Every upstream benchmark counts a non-submission as an in-denominator
        # FAIL, so we synthesize one placeholder per missing task here. This is
        # a NO-OP when coverage is already complete (live_agent always emits an
        # artifact per task; a full predictions file leaves zero missing).
        #
        # Deliberate subset scoring is expressed via --limit / --split, which
        # changes what ``source.load`` returns; reconciliation is against the
        # LOADED ``tasks``, so a deliberate subset is unaffected (the subset is
        # exactly the loaded set).
        results = self._reconcile_coverage(results, tasks, oracle)

        write_tasks(resolved_run_dir, tasks)
        write_results(resolved_run_dir, results)
        metrics = compute_vibe_metrics(results, tasks)
        write_manifest(
            resolved_run_dir,
            {
                "run_id": run_config.run_id,
                "source": source_name,
                "oracle": oracle_name,
                "mode": mode,
                "n_tasks": len(tasks),
                "n_results": len(results),
                "resource_budget": budget.model_dump(mode="json"),
            },
        )

        return VibeRunResult(
            run_id=run_config.run_id,
            run_dir=str(resolved_run_dir),
            source=source_name,
            oracle=oracle_name,
            tasks=list(tasks),
            results=results,
            metrics=metrics,
        )

    # --- coverage reconciliation -------------------------------------

    def _reconcile_coverage(
        self,
        results: list[VibeTaskResult],
        tasks: Sequence[VibeTask],
        oracle: Any,
    ) -> list[VibeTaskResult]:
        """Append an in-denominator fail for every loaded task with no row.

        A loaded task whose ``id`` appears nowhere in ``results`` had no
        prediction (or its candidate was dropped before producing any row), so
        it would silently vanish from ``compute_vibe_metrics``' result-keyed
        denominators. Synthesize one ``model_failure`` / ``no_submission``
        placeholder per missing task -- the upstream-faithful "a missing
        submission is a scored FAIL" rule -- so the denominators stay anchored
        to the loaded task set. A no-op when coverage is already complete.
        """
        covered = {row.task_id for row in results}
        missing = [task for task in tasks if task.id not in covered]
        if not missing:
            return results

        # Best-effort model label: reuse whatever model the other rows carry so
        # a single-model run keeps one coherent label; fall back to ``none``.
        model = next(
            (
                row.model
                for row in results
                if row.model and row.model != "none"
            ),
            "none",
        )
        placeholders = [
            _no_submission_row(task, oracle, model=model) for task in missing
        ]

        missing_ids = [task.id for task in missing]
        shown = missing_ids[:_MISSING_ID_LOG_CAP]
        extra = len(missing_ids) - len(shown)
        id_summary = list(shown)
        if extra > 0:
            id_summary.append(f"(+{extra} more)")
        _log.warning(
            "vibe eval: %d of %d loaded task(s) had no prediction; scored as "
            "in-denominator non-submission fails: %s",
            len(missing_ids),
            len(tasks),
            id_summary,
        )
        return results + placeholders

    # --- artifact resolution -----------------------------------------

    def _resolve_artifacts(
        self,
        predictions: Sequence[AgentArtifact | dict[str, Any]] | None,
        artifacts: Sequence[AgentArtifact] | None,
    ) -> list[AgentArtifact]:
        """Build the BYO artifact list from ``predictions``/``artifacts``."""
        if artifacts is not None:
            return list(artifacts)
        if predictions is not None:
            return [_coerce_artifact(entry) for entry in predictions]
        raise ValueError(
            "patch_eval mode requires `predictions` or `artifacts`"
        )

    def _generate_live(
        self,
        tasks: Sequence[VibeTask],
        *,
        oracle: Any,
        agent: str | None,
        agent_model: str | None,
        run_dir: Path,
    ) -> list[AgentArtifact]:
        """Drive a registered agent over each task -> a BYO-equivalent list.

        Each task is handed a **sealed base checkout** as ``workdir`` when one
        can be restored, so a generic driver (e.g. ``claude``) sees the
        repository contents rather than generating blind from the instructions
        alone; ``git`` history and local solution artifacts are stripped first
        so the agent cannot read the upstream fix.

        The base is resolved oracle-first: when the oracle exposes a
        ``live_base(task, cache_dir)`` hook (e.g. SecureVibeBench, whose tasks
        carry the real ``repo.url`` and whose scoring checks out PVIC = ``VIC^``
        from a host-side clone), that ``(repo_dir, ref)`` is sealed so the agent
        generates against *exactly* the tree its patch is later scored on.
        Otherwise the Materializer's generic resolution applies. ``workdir`` is
        ``None`` only when neither yields a checkout (the per-task repo lives in
        a dataset Docker image and is the in-container driver's job to extract).

        The driver produces one artifact per task; a driver that fails on a
        single task degrades to an empty-payload artifact (its own contract)
        so one bad generation never aborts the batch. Per-task telemetry on
        the returned :class:`AgentResult` (token counts, ``cost_usd``, the
        resolved model, driver ``metadata``) is folded into the artifact's
        metadata by :func:`_fold_agent_telemetry` so it persists into
        ``artifact.json`` rather than being discarded with the result wrapper.
        The resulting list flows through the identical classify -> materialize
        -> score path as a BYO ``patch_eval`` run, so live and offline runs
        are scored the same way.
        """
        if not agent:
            raise ValueError(
                "live_agent mode requires an --agent driver name"
            )
        from guard_eval_harness.vibecoding.agents.base import (
            get_agent_driver,
        )

        driver = get_agent_driver(agent)
        materializer = Materializer(self._cache_dir, run_dir)
        live_base = getattr(oracle, "live_base", None)
        # The oracle frames generation (artifact kind + optional prompt/parse)
        # so a dataset-agnostic driver emits exactly what this oracle scores.
        generation_spec = getattr(oracle, "generation_spec", None)
        artifacts: list[AgentArtifact] = []
        for task in tasks:
            resolved_base = (
                live_base(task, self._cache_dir)
                if callable(live_base)
                else None
            )
            if resolved_base is not None:
                repo_dir, ref = resolved_base
                workdir = materializer.restore_base(
                    task, source=repo_dir, ref=ref
                )
            else:
                workdir = materializer.restore_base(task)
            spec = (
                generation_spec(task, cache_dir=self._cache_dir)
                if callable(generation_spec)
                else None
            )
            result = driver.generate(
                task, workdir=workdir, model=agent_model, gen_spec=spec
            )
            artifacts.append(_fold_agent_telemetry(result))
        return artifacts

    def _classify_artifacts(
        self,
        oracle: Any,
        tasks: Sequence[VibeTask],
        artifacts: Sequence[AgentArtifact],
    ) -> _RunInputs:
        """Split artifacts into oracle-compatible pairs vs unsupported rows.

        An artifact is unsupported when its ``kind`` is not in the oracle's
        ``artifact_kinds`` or when no loaded task matches its ``task_id``.
        ``ArtifactAdapter.validate`` (when the oracle exposes one) gets the
        final say on shape compatibility.
        """
        by_id = {task.id: task for task in tasks}
        validator = self._artifact_validator(oracle)
        inputs = _RunInputs()
        staged_task_ids: set[str] = set()
        for artifact in artifacts:
            task = by_id.get(artifact.task_id)
            if artifact.kind not in oracle.artifact_kinds:
                inputs.unsupported.append(
                    _unsupported_row(
                        task,
                        artifact,
                        detail=(
                            f"oracle {oracle.name!r} supports "
                            f"{sorted(oracle.artifact_kinds)}, got "
                            f"kind={artifact.kind!r}"
                        ),
                    )
                )
                continue
            if task is None:
                inputs.unsupported.append(
                    _unsupported_row(
                        None,
                        artifact,
                        detail=(
                            "no loaded task matches "
                            f"task_id={artifact.task_id!r}"
                        ),
                    )
                )
                continue
            if validator is not None:
                try:
                    validator.validate(artifact)
                except UnsupportedArtifactError as exc:
                    inputs.unsupported.append(
                        _unsupported_row(task, artifact, detail=str(exc))
                    )
                    continue
            if artifact.task_id in staged_task_ids:
                # These benchmark upstreams evaluate one candidate per task
                # per run (instance/ARVO-keyed), so a second candidate for the
                # same task in one batch cannot be scored distinctly and would
                # otherwise overwrite the first in the upstream staging dir.
                # Preserve it as a clearly-labeled row; run trials / pass@k /
                # model comparisons as separate runs (distinct run_id).
                inputs.unsupported.append(_unscored_row(task, artifact))
                continue
            staged_task_ids.add(artifact.task_id)
            inputs.pairs.append((task, artifact))
        return inputs

    @staticmethod
    def _artifact_validator(oracle: Any) -> Any:
        """Return the oracle's artifact validator if it exposes one."""
        validate = getattr(oracle, "validate", None)
        return oracle if callable(validate) else None

    # --- materialization ---------------------------------------------

    @staticmethod
    def _needs_worktree(oracle: Any) -> bool:
        """Whether this oracle scores a materialized ``repo_dir``."""
        return "repo_dir" in set(oracle.artifact_kinds)

    def _materialize_if_needed(
        self,
        oracle: Any,
        inputs: _RunInputs,
        run_dir: Path,
    ) -> dict[str, str]:
        """Build worktrees for repo_dir oracles; demote failures.

        Returns a ``task_id -> worktree_sha256`` map for the worktrees built
        this run. For ``repo_dir`` artifacts the cache key must reflect the
        *contents* the oracle scores rather than the candidate's reusable
        worktree path string, so the runner keys the cache on this
        materialized tree digest (see ``_cache_key_builder``).

        A materialization failure is a candidate-artifact problem, so the
        pair is dropped from scoring and replaced with an ``unsupported``
        row rather than crashing the whole run.
        """
        if not self._needs_worktree(oracle):
            return {}
        materializer = Materializer(self._cache_dir, run_dir)
        kept: list[tuple[VibeTask, AgentArtifact]] = []
        content_hashes: dict[str, str] = {}
        for task, artifact in inputs.pairs:
            try:
                materialized = materializer.prepare(
                    task, artifact, need_worktree=True
                )
            except MaterializeError as exc:
                inputs.unsupported.append(
                    _unsupported_row(
                        task,
                        artifact,
                        detail=f"materialize failed: {exc}",
                    )
                )
                continue
            if materialized is not None:
                content_hashes[task.id] = materialized.worktree_sha256
            kept.append((task, artifact))
        inputs.pairs = kept
        return content_hashes

    # --- evaluation ---------------------------------------------------

    def _is_batch(self, oracle: Any) -> bool:
        """Whether stage/evaluate/parse run ONCE over the whole list."""
        granularity = getattr(oracle, "granularity", "per_task")
        model = getattr(getattr(oracle, "parallelism", None), "model", "")
        return granularity == "batch" or model == "batch_internal"

    def _evaluate(
        self,
        oracle: Any,
        pairs: Sequence[tuple[VibeTask, AgentArtifact]],
        *,
        run_config: OracleRunConfig,
        budget: ResourceBudget,
        env_provider: Any,
        no_cache: bool,
        content_hashes: dict[str, str] | None = None,
    ) -> list[VibeTaskResult]:
        """Stage/evaluate/parse compatible pairs and return scored rows."""
        if not pairs:
            return []

        cache = OracleResultCache(self._cache_dir / "cache" / "vibecoding")
        key_for = self._cache_key_builder(
            oracle, run_config, content_hashes or {}
        )

        if self._is_batch(oracle):
            return self._evaluate_batch(
                oracle,
                pairs,
                run_config=run_config,
                budget=budget,
                env_provider=env_provider,
                cache=cache,
                key_for=key_for,
                no_cache=no_cache,
            )
        return self._evaluate_serial(
            oracle,
            pairs,
            run_config=run_config,
            budget=budget,
            env_provider=env_provider,
            cache=cache,
            key_for=key_for,
            no_cache=no_cache,
        )

    def _evaluate_batch(
        self,
        oracle: Any,
        pairs: Sequence[tuple[VibeTask, AgentArtifact]],
        *,
        run_config: OracleRunConfig,
        budget: ResourceBudget,
        env_provider: Any,
        cache: OracleResultCache,
        key_for: Any,
        no_cache: bool,
    ) -> list[VibeTaskResult]:
        """Single stage/evaluate/parse over the whole compatible list.

        Cache hits short-circuit per task; on a full hit the oracle's
        ``evaluate`` is never invoked. Misses are staged + evaluated once as
        a batch (never fanned out per task).
        """
        # Index-aligned so duplicate task_ids never collapse: each pair keeps
        # its own slot whether it was a cache hit or freshly scored.
        rows: list[VibeTaskResult | None] = [None] * len(pairs)
        pending: list[tuple[VibeTask, AgentArtifact]] = []
        pending_idx: list[int] = []
        for index, (task, artifact) in enumerate(pairs):
            hit = self._cache_hit(
                cache, key_for(task, artifact), no_cache
            )
            if hit is not None:
                rows[index] = hit
            else:
                pending.append((task, artifact))
                pending_idx.append(index)

        if pending:
            scored = self._run_oracle(
                oracle,
                pending,
                run_config=run_config,
                budget=budget,
                env_provider=env_provider,
            )
            self._store(scored, pending, cache, key_for, no_cache)
            for slot, row in zip(pending_idx, scored):
                rows[slot] = row

        return [row for row in rows if row is not None]

    def _evaluate_serial(
        self,
        oracle: Any,
        pairs: Sequence[tuple[VibeTask, AgentArtifact]],
        *,
        run_config: OracleRunConfig,
        budget: ResourceBudget,
        env_provider: Any,
        cache: OracleResultCache,
        key_for: Any,
        no_cache: bool,
    ) -> list[VibeTaskResult]:
        """Per-task stage/evaluate/parse for non-batch oracles."""
        rows: list[VibeTaskResult] = []
        for task, artifact in pairs:
            key = key_for(task, artifact)
            hit = self._cache_hit(cache, key, no_cache)
            if hit is not None:
                rows.append(hit)
                continue
            scored = self._run_oracle(
                oracle,
                [(task, artifact)],
                run_config=run_config,
                budget=budget,
                env_provider=env_provider,
            )
            self._store(scored, [(task, artifact)], cache, key_for, no_cache)
            rows.extend(scored)
        return rows

    def _run_oracle(
        self,
        oracle: Any,
        pairs: Sequence[tuple[VibeTask, AgentArtifact]],
        *,
        run_config: OracleRunConfig,
        budget: ResourceBudget,
        env_provider: Any,
    ) -> list[VibeTaskResult]:
        """Stage + evaluate + parse one batch; return a pairs-aligned list.

        The result is one row per input pair, in order, so multiple candidate
        artifacts sharing a ``task_id`` are all preserved (never collapsed).
        Any exception out of ``evaluate``/``parse`` is treated as infra and
        yields one ``infra_failure`` row per pair so the run never crashes and
        infra never pollutes the model denominator.
        """
        tasks = [task for task, _ in pairs]
        artifacts = [artifact for _, artifact in pairs]
        run_dir = Path(run_config.run_dir)
        try:
            staged: StagedOracleInput = oracle.stage(
                tasks, artifacts, run_dir
            )
        except UnsupportedArtifactError as exc:
            # A staging-time shape rejection is a candidate-artifact problem,
            # not an environment failure: label the batch ``unsupported``
            # (adapter origin) so it never pollutes the model/infra
            # denominators. Adapters that expose ``validate()`` scope this to
            # the offending artifact in ``_classify_artifacts`` already; this
            # is the backstop for adapters that only detect the shape problem
            # once inside ``stage()``.
            return [
                _unsupported_row(task, artifact, detail=str(exc))
                for task, artifact in pairs
            ]
        try:
            raw: RawOracleResult = oracle.evaluate(
                staged, run_config, budget, env_provider
            )
            parsed = oracle.parse(raw)
        except Exception as exc:  # noqa: BLE001 - infra isolation boundary
            return [
                _infra_row(
                    task,
                    artifact,
                    detail=f"{type(exc).__name__}: {exc}",
                    trial_index=run_config.trial_index,
                    random_seed=run_config.random_seed,
                )
                for task, artifact in pairs
            ]
        # Map parsed rows back to pairs in order, consuming one parsed row per
        # matching task_id so duplicate candidates each keep their own row.
        by_task: dict[str, deque[VibeTaskResult]] = defaultdict(deque)
        for row in parsed:
            by_task[row.task_id].append(derive_task_metrics(row))
        out: list[VibeTaskResult] = []
        for task, artifact in pairs:
            queue = by_task.get(task.id)
            if queue:
                out.append(queue.popleft())
            else:
                out.append(_unscored_row(task, artifact))
        return out

    @staticmethod
    def _cache_hit(
        cache: OracleResultCache,
        key: CacheKey,
        no_cache: bool,
    ) -> VibeTaskResult | None:
        """Cache lookup that refuses environment-dependent rows.

        Rows cached BEFORE the infra-origin write guard existed (e.g. a
        SecCodeBench outage scored as completed/False under the unchanged
        adapter version) must not replay against a healthy environment:
        a hit whose ``failure_origin`` is ``infra`` is treated as a miss so
        the task re-scores, mirroring the write-side policy in
        :meth:`_store`.
        """
        if no_cache:
            return None
        hit = cache.get(key)
        if hit is not None and hit.failure_origin == "infra":
            return None
        return hit

    def _store(
        self,
        scored: Sequence[VibeTaskResult],
        pairs: Sequence[tuple[VibeTask, AgentArtifact]],
        cache: OracleResultCache,
        key_for: Any,
        no_cache: bool,
    ) -> None:
        """Cache only environment-independent rows (completed/model_failure).

        ``scored`` is aligned to ``pairs``. Infra and unsupported rows are
        intentionally never cached so they are retried on the next run. The
        same applies to a row whose status is cacheable but whose
        ``failure_origin`` is ``infra`` (e.g. SecCodeBench's upstream-parity
        scored fail for an unverifiable submission): its verdict depends on
        the run's environment, so replaying it from the cache against a
        healthy verifier would freeze a transient outage into a permanent 0.
        """
        if no_cache:
            return
        for (task, artifact), row in zip(pairs, scored):
            if row is None or row.status not in _CACHEABLE_STATUSES:
                continue
            if row.failure_origin == "infra":
                continue
            cache.put(key_for(task, artifact), row)

    # --- cache key derivation ----------------------------------------

    def _cache_key_builder(
        self,
        oracle: Any,
        run_config: OracleRunConfig,
        content_hashes: dict[str, str] | None = None,
    ) -> Any:
        """Return a ``(task, artifact) -> CacheKey`` closure for ``oracle``.

        The config / capabilities / anti-cheat hashes are stable for the
        whole run, so they are computed once here and closed over.
        ``content_hashes`` maps ``task_id`` to a materialized worktree digest
        for ``repo_dir`` artifacts (empty for non-worktree oracles).
        """
        content_hashes = content_hashes or {}
        adapter_version = str(getattr(oracle, "parser_version", "0"))
        upstream_ref = getattr(oracle.env, "upstream_ref", None)
        caps_hash = sha256_payload(
            oracle.capabilities.model_dump(mode="json")
        )
        config_hash = sha256_payload(
            {
                "trial_index": run_config.trial_index,
                "random_seed": run_config.random_seed,
                "extra": run_config.extra,
            }
        )
        policy_hash = sha256_payload(
            {"anti_cheat_policy_id": _DEFAULT_ANTI_CHEAT_POLICY}
        )

        def build(task: VibeTask, artifact: AgentArtifact) -> CacheKey:
            # For repo_dir oracles the candidate identity that matters is the
            # materialized tree CONTENT, not the artifact's worktree path
            # string (a BYO run can regenerate different code at the same
            # path). Key on the materialized digest when present so changed
            # contents miss the cache instead of returning a stale verdict.
            # Otherwise key on the artifact's *scoreable* hash, which excludes
            # the live runner's folded telemetry so per-generation token/cost
            # never perturbs the cache key (it is provenance, not scoring
            # input); BYO artifacts hash identically to before.
            content_hash = content_hashes.get(task.id)
            artifact_identity = (
                content_hash
                if content_hash is not None
                else artifact_scoring_sha256(artifact)
            )
            return CacheKey(
                task_id=task.id,
                artifact_sha256=artifact_identity,
                adapter_name=oracle.name,
                adapter_version=adapter_version,
                upstream_ref=upstream_ref,
                oracle_config_hash=config_hash,
                oracle_capabilities_hash=caps_hash,
                trial_index=run_config.trial_index,
                random_seed=run_config.random_seed,
                anti_cheat_policy_hash=policy_hash,
            )

        return build

    # --- misc helpers -------------------------------------------------

    @staticmethod
    def _default_budget(oracle: Any) -> ResourceBudget:
        """A conservative budget honoring the oracle's worker bound."""
        parallelism = getattr(oracle, "parallelism", None)
        workers = int(getattr(parallelism, "default_workers", 1) or 1)
        return ResourceBudget(max_workers=max(1, workers))

    @staticmethod
    def _resolve_run_dir(
        run_dir: str | Path | None,
        run_config: OracleRunConfig,
    ) -> Path:
        """Resolve the run directory from arg / config / default layout."""
        if run_dir is not None:
            return Path(run_dir)
        if run_config.run_dir:
            return Path(run_config.run_dir)
        return Path("runs") / "vibecoding" / run_config.run_id


__all__ = ["VibeRunner", "VibeRunResult"]
