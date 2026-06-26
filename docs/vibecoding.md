# VibeCoding Safety Bench

A repository-level **secure-coding** benchmark family. Each task asks a model to
write, complete, or patch real-world code; an out-of-process **oracle** then
builds the result in a container and scores two things:

- **functional correctness** — does the code build and pass the task's tests?
- **security** — is the target vulnerability absent (and, where supported, no
  new vulnerability introduced)?

It is exposed under the `geh vibe` command group and is independent of the
classification benchmarks documented elsewhere. Because scoring runs candidate
code, **Docker is required** for every dataset below.

!!! note "Single-shot, not agentic"
    This distribution ships only **single-shot** model drivers (one prompt →
    one completion → score). There is no multi-step / tool-using agent driver
    here; agentic and bring-your-own runs are supported via
    [pre-generated predictions](#bring-your-own-predictions).

## Install

```bash
pip install -e ".[vibecoding]"
```

The subsystem is pure-Python (stdlib + pydantic); it shells out to each upstream
benchmark rather than importing it.

## Quickstart

```bash
# 0. (optional) put the cache + upstream checkouts somewhere with disk
export GEH_CACHE_DIR=/scratch/$USER/.geh

# 1. clone + build a dataset's upstream environment (one-time)
geh vibe acquire --dataset baxbench

# 2. drive a model end-to-end, then score (functional + security)
export OPENAI_API_KEY=sk-...
geh vibe run --dataset baxbench --agent llm --model gpt-5.5 --limit 5

# 3. (re)build summary.json + report.md for the run dir it printed
geh vibe report --run-dir runs/vibecoding/<run-id>
```

`geh vibe datasets` lists every registered dataset with its artifact kind and
capabilities as JSON; `geh vibe doctor --dataset <d>` probes Docker, the upstream
checkout, the venv, disk, and secrets before you commit to a run.

## How it works

Every dataset wires together a **task source** (loads tasks), an **agent driver**
(produces a candidate — code, completion, or patch), and an **oracle** (stages
the candidate into the upstream harness, runs it in Docker, and parses verdicts):

```
source.load → agent (live model | BYO) → oracle.stage → oracle.evaluate (Docker) → oracle.parse → metrics
```

Two ways to produce candidates:

- **Live** (`geh vibe run --agent ...`) — a model is driven for each task.
- **BYO** (`geh vibe eval --predictions file.jsonl`) — you supply candidates you
  generated yourself (any agent framework). This is the only path for `ase`, and
  is available for every dataset.

## Datasets

| Dataset | What it checks | Task type | Live (`run`)? | Heavy deps |
| --- | --- | --- | :---: | --- |
| `baxbench` | Full backend-app scaffolds: functional tests + security exploits | `project_scaffold` | yes | Docker; 392 tasks (28 scenarios × 14 envs) |
| `seccodebench` | Function-level secure code (5 languages) via verifier services | `project_scaffold` | yes | Docker + verifier services; LLM judges (Java) |
| `secrepobench` | Secure completion of a masked region in real C/C++ repos | `repo_completion` | yes | Docker (ARVO); **external checkout you supply** |
| `securevibebench` | Repo **patch** scored against the target vuln at PVIC | `repo_patch` | yes | Docker (ARVO); 105 tasks (HF `iCSawyer/SecureVibeBench`) |
| `susvibes` | SWE-bench-style patch: fix the issue **and** remove the CWE/CVE | `repo_patch` | yes | Docker; heaviest (~400 GB images) |
| `ase` | Whole-repo edits scored by A.S.E / AICGSecEval | `repo_dir` | no — BYO/agentic only | Docker; needs materialized worktrees |

"Live" means `geh vibe run` can drive a model end-to-end. `ase` scores a
`repo_dir` artifact, which a single-shot driver cannot produce — score it with
`geh vibe eval --predictions` instead.

A few per-dataset specifics worth knowing:

- **`seccodebench` is a separate capability tier.** Its security verdict can
  involve LLM judges (Java path, majority vote), so its rows are deliberately
  **segregated from the deterministic target-secure leaderboard** — treat its
  numbers on their own, not pooled with the other datasets.
- **`secrepobench` is `external_only`.** No redistributable upstream license, so
  you must supply the checkout yourself (`SecRepoBench`); the catalog URL is a
  placeholder. It also needs ~200 GB for ARVO images.
- **`securevibebench`** loads its 105 tasks from the Hugging Face dataset
  `iCSawyer/SecureVibeBench` and is the only oracle that seeds a **live** agent
  from the real pre-fix tree (PVIC) rather than blind. Semgrep/SAST "new-vuln"
  detection is **off by default** (set `SEMGREP_APP_TOKEN` to enable); without it
  the `strict_secure` track is not scored.
- **`baxbench`** needs `geh vibe acquire` to materialize per-scenario descriptors;
  without acquisition it falls back to a 6-env representative slice.

## Models and providers

A live run needs an **agent** (`--agent`) and usually a **model** (`--model`).
The model id is resolved as **explicit `--model` > `GEH_VIBE_MODEL` > the agent's
default**.

### Routing

`--agent llm` is a **router**: it picks the provider from the model name.

| Model id (lowercased) | Routes to | API key |
| --- | --- | --- |
| starts with `claude` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-*`, `gpt4*`, `gpt5*`, `chatgpt*`, `codex*`, or `o<N>` (e.g. `o3`, `o4-mini`) | OpenAI | `OPENAI_API_KEY` |
| anything else | OpenRouter | `OPENROUTER_API_KEY` |

Or pin a provider directly with a fixed-provider agent alias:

| `--agent` | Provider | Default model |
| --- | --- | --- |
| `anthropic`, `claude` | Anthropic | `claude-opus-4-8` |
| `openai`, `gpt` | OpenAI | `gpt-5.5` |
| `codex` | OpenAI | `gpt-5.1-codex` |
| `deepseek` | OpenRouter | `deepseek/deepseek-v4-flash` |
| `gemini` | OpenRouter | `google/gemini-2.5-pro` |
| `qwen` | OpenRouter | `qwen/qwen3.7-max` |
| `glm` | OpenRouter | `z-ai/glm-5.2` |
| `openrouter` | OpenRouter | `deepseek/deepseek-v4-flash` |

`--agent` is **required** (there is no default). A missing API key fails fast
with a clear error rather than degrading to empty outputs.

### OpenRouter vendor-namespacing

OpenRouter needs `vendor/model` ids. A bare third-party name is normalized
automatically from its leading alphabetic run:

| bare prefix | becomes vendor |
| --- | --- |
| `gemini` | `google` |
| `deepseek` | `deepseek` |
| `qwen` | `qwen` |
| `glm` | `z-ai` |
| `llama` | `meta-llama` |
| `mistral`, `mixtral` | `mistralai` |
| `grok` | `x-ai` |

So `--agent llm --model gemini-2.5-pro` is sent as `google/gemini-2.5-pro`, and
`qwen3.7-max` → `qwen/qwen3.7-max` (the leading-run rule keeps dotted/dashed
versions intact). An id that already contains `/` is passed through unchanged; an
unknown bare name errors rather than silently 404-ing.

### Reasoning effort

Effort is provider-specific — set the knob that matches your provider:

- **Anthropic** → `GEH_VIBE_THINK_EFFORT` (e.g. `high`). Sends adaptive extended
  thinking; thinking tokens are billed as output and share the token budget.
- **OpenAI / OpenRouter** → `GEH_VIBE_REASONING_EFFORT` (e.g. `high`). Direct
  OpenAI receives `reasoning_effort`; OpenRouter receives `reasoning: {effort}`
  (per-model support varies).

```bash
# Anthropic at high effort
GEH_VIBE_THINK_EFFORT=high geh vibe run --dataset securevibebench --agent claude --model claude-opus-4-8

# OpenAI / OpenRouter at high effort
GEH_VIBE_REASONING_EFFORT=high geh vibe run --dataset baxbench --agent llm --model gpt-5.5
```

!!! tip "Raise the token budget for high effort"
    Thinking / reasoning tokens share `max_tokens`. On high effort with full-file
    generation the default budget can starve the answer — raise
    `GEH_VIBE_MAX_TOKENS` (e.g. `32000`) to avoid truncation.

## Environment variables

| Variable | Effect | Default |
| --- | --- | --- |
| `GEH_CACHE_DIR` | Cache + upstream-checkout root. Precedence: `--cache-dir` > this > `<repo>/.geh`. | `<repo>/.geh` |
| `GEH_VIBE_MODEL` | Default model id when `--model` is omitted. | unset |
| `GEH_VIBE_MAX_TOKENS` | Output-token budget (`max_tokens` / `max_completion_tokens`). | `32000` |
| `GEH_VIBE_HTTP_TIMEOUT` | Per-request httpx **read** timeout, seconds (fires only when no bytes arrive in the window). | `180` |
| `GEH_VIBE_HARD_TIMEOUT` | Hard wall-clock cap per request, seconds. Guards against servers that dribble keepalive bytes and defeat the read timeout; on expiry the request is abandoned and retried, then degrades to an in-denominator failure so the run advances. `0` disables. | `0` |
| `GEH_VIBE_PROMPT_CACHE` | Anthropic prompt caching (one ephemeral breakpoint). `0` disables. | `1` |
| `GEH_VIBE_THINK_EFFORT` | Anthropic adaptive-thinking effort (e.g. `high`). | unset |
| `GEH_VIBE_REASONING_EFFORT` | OpenAI / OpenRouter reasoning effort (e.g. `high`). | unset |
| `GEH_VIBE_SHARD` | Strided task sharding `"<idx>/<num>"` for parallel runs (use a distinct `--run-id` per shard). | unset |

!!! note
    `GEH_VIBE_MAX_TOKENS`, `GEH_VIBE_HTTP_TIMEOUT`, and `GEH_VIBE_HARD_TIMEOUT`
    are read once at import — set them in the launch environment, not mid-process.

## Commands

```bash
geh vibe datasets                                      # list sources/oracles + capabilities (JSON)
geh vibe acquire  --dataset <d> [--force]              # clone + build upstream env (run this first)
geh vibe doctor   --dataset <d> [--skip-docker]        # probe Docker/checkout/venv/disk/secrets
geh vibe run      --dataset <d> --agent <a> [...]      # live model generation + scoring
geh vibe eval     --dataset <d> --predictions f.jsonl  # score BYO predictions (no live model)
geh vibe report   --run-dir <dir>                      # rebuild summary.json + report.md
```

`geh vibe run` flags: `--dataset` (required), `--agent` (required), `--model`,
`--limit N`, `--run-id`, `--run-dir`, `--cache-dir`, `--no-cache`, `--allow-empty`.
`--concurrency` and `--trials` must be `1` (other values are rejected). Omit
`--run-id` / `--run-dir` to auto-name `runs/vibecoding/vibe-<dataset>-<unixtime>`.

`acquire` is a prerequisite for the upstream-backed datasets — without it a
`run` / `eval` loads zero tasks and stops.

## Bring-your-own predictions

`geh vibe eval --predictions <file>` scores a JSONL file where each line is one
`AgentArtifact`. This is the path for any dataset (and the only path for `ase`).
Each record:

| Field | Required | Notes |
| --- | --- | --- |
| `task_id` | yes | The task id from `geh vibe datasets` / the dataset. |
| `model` | yes | Free-form label recorded in the report. |
| `kind` | yes | One of `patch`, `full_file`, `completion`, `repo_dir`. |
| payload | yes | The field matching `kind` (e.g. `patch` / `files` / `completion`). |
| `metadata` | no | Arbitrary JSON carried into the result. |

The artifact `kind` must match what the dataset's oracle accepts (see the
task-type column above): patch datasets take `patch`, full-file / scaffold
datasets take `full_file`, `secrepobench` takes `completion` (or `full_file`), and
`ase` takes `repo_dir`.

## Metrics and scoring

Scores are **capability-scoped** and tri-state (`True` / `False` / `None`). The
two headline metrics:

- **`target_secure_success`** = functional correctness **AND** security-oracle
  pass.
- **`strict_secure_success`** = the above **AND** no new vulnerability introduced
  (only meaningful where new-vuln detection is enabled, e.g. `securevibebench`
  with Semgrep on).

These combine with **three-valued (Kleene) AND**: any definite `False` makes the
result `False` (even if another gate is unknown); a result is only `None`
(indeterminate) when nothing failed but a required gate is unknown.

**Denominator.** A row counts toward a rate when its status is `completed`,
`model_failure`, or `cheating_detected`. Crucially, a model that produced a
non-building, non-applying, or empty candidate (`build_failed`,
`patch_apply_failed`, `empty_diff`) — or no submission at all — is a **scored
failure in the denominator**, not an exclusion. Only `infra_failure` and
`unsupported` (environment / adapter problems, not the model's fault) are
**excluded** from rates (and still reported as counts).

**`excluded_null`.** Within the denominator, a row whose verdict is genuinely
indeterminate (`None`) is dropped from *that metric only* and reported as
`excluded_null`. A definite `False` is always a scored failure, never an
exclusion.

Each metric cell serializes as `{"rate": <float|null>, "n_scored": <int>,
"excluded_null": <int>}` (`rate` is `null` when nothing was scored). `summary.json`
(written next to the run) also carries `totals`, per-track `leaderboard` sections,
`auxiliary_rates` (functional-only, oracle-security, and the functional→secure
gap), `breakdowns` (per-CWE / per-dataset / per-task-type), `failures`, and a
`quality_gate` that fails if the excluded fraction exceeds 20%.

**Reading a number.** `target_secure_success = 0.42` with `n_scored=100`,
`excluded_null=8`, `excluded_infra=5` means: of 100 rows with a definite verdict,
42% were both functionally correct and passed the security oracle; 8 rows were
indeterminate (dropped from this metric only); 5 were infra failures excluded
from all metrics; and build / apply / empty / missing-submission failures are
already counted as failures within that 100.

### Leaderboard tracks

Tracks are scored over their own datasets, pooled, never mixing dataset families:

| Track | Metric | Datasets |
| --- | --- | --- |
| `vibecoding_safety_repo_patch_v0` | `target_secure_success` | `susvibes`, `securevibebench` |
| `vibecoding_safety_repo_completion_v0` | `target_secure_success` | `secrepobench` |
| `strict_secure` (secondary) | `strict_secure_success` | `securevibebench` |

Use `geh vibe report --run-dir <dir>` to (re)generate `summary.json` and a
human-readable `report.md` from a finished run.
