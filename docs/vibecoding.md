# VibeCoding Safety Bench

A repository-level secure-coding benchmark family. Each task asks a coding
agent to write or complete real-world code; an out-of-process **oracle** then
builds the result in a container and scores two things:

- **functional correctness** — does the code build and pass the task's tests?
- **security** — is the target vulnerability absent (and, where supported, no
  new vulnerability introduced)?

It is exposed under `geh vibe` and is independent of the classification
benchmarks documented elsewhere. Scoring runs candidate code, so Docker is
required for every dataset below.

## Install

```bash
pip install -e ".[vibecoding]"
```

The subsystem itself is pure-Python (stdlib + pydantic); it shells out to the
upstream benchmarks rather than importing them.

## Datasets

| Dataset | What it checks | Live generation |
| --- | --- | --- |
| `baxbench` | Full-file app scaffolds, functional + security | yes |
| `secrepobench` | Repository-level function completion | yes |
| `securevibebench` | Patch-style fixes | yes |
| `susvibes` | Patch-style fixes | yes |
| `seccodebench` | Full-file generation | BYO only |
| `ase` | Whole-repo edits | BYO only |

"Live generation" means `geh vibe run` can drive a model end-to-end. `ase` and
`seccodebench` are scored from pre-generated artifacts via `geh vibe eval
--predictions` (see [Bring-your-own predictions](#bring-your-own-predictions)).

`geh vibe datasets` prints the registered datasets with their required artifact
kind and capabilities as JSON.

## Commands

```bash
# List datasets, oracles, and their capabilities.
geh vibe datasets

# Clone + build a dataset's upstream environment (one-time, before run/eval).
geh vibe acquire --dataset baxbench

# Live: drive an agent, then score (functional + security).
geh vibe run --dataset baxbench --agent claude --model claude-sonnet-4-6 --limit 5

# BYO: score predictions you generated yourself (no live agent).
geh vibe eval --dataset seccodebench --predictions preds.jsonl

# Rebuild summary.json + report.md for an existing run directory.
geh vibe report --run-dir runs/vibecoding/<run-id>

# Probe the upstream checkout, venv, and Docker for a dataset.
geh vibe doctor --dataset baxbench
```

`--agent` selects a driver (`claude`, `openai`, `gpt`, `llm`, ...); `--model`
is the provider model id. API keys are read from the environment
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## Bring-your-own predictions

`geh vibe eval --predictions <file>` scores a JSONL file where each line is one
`AgentArtifact`. This is the path for any dataset (and the only path for `ase`
and `seccodebench`). Each record:

| Field | Required | Notes |
| --- | --- | --- |
| `task_id` | yes | The task id from `geh vibe datasets` / the dataset. |
| `model` | yes | Free-form label recorded in the report. |
| `kind` | yes | One of `patch`, `full_file`, `completion`, `repo_dir`. |
| payload | yes | The field matching `kind` (see below). |
| `metadata` | no | Arbitrary JSON carried into the result. |

The payload field depends on `kind`:

- `patch` -> `patch`: a unified diff string.
- `completion` -> `completion`: the completion text.
- `full_file` -> `files`: a `{relative/path: contents}` map.
- `repo_dir` -> `worktree`: a path to a prepared working tree.

```jsonl
{"task_id": "secrepobench/910", "model": "my-model", "kind": "completion", "completion": "public int add(int a, int b) { return a + b; }"}
{"task_id": "baxbench/Calculator__Go-Fiber", "model": "my-model", "kind": "full_file", "files": {"app/main.go": "package main\n..."}}
```

A missing payload, an empty diff, or a build failure is scored as an
in-denominator model failure rather than dropped, so the denominator reflects
every task attempted.

## Reports

`run`/`eval` write a run directory containing `summary.json` (headline tracks +
per-CWE breakdowns) and `report.md`. The headline tracks pool `target_secure`
success over related datasets; secondary capability-scoped tracks
(e.g. strict-secure) are reported where supported.
