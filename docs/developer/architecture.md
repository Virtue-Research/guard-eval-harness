# Architecture

Guard Eval Harness is a modular evaluation pipeline with two primary extension
surfaces:

- dataset adapters
- model adapters

Benchmark packs and presets sit above those layers as reusable run definitions,
while the CLI and execution pipeline handle config resolution, orchestration,
artifacts, and reporting.

## Project Structure

```text
src/guard_eval_harness/
  cli/
    main.py               # argparse-based CLI entry point
  config/
    loading.py            # YAML/dict config resolution
    models.py             # resolved config models
  datasets/
    base.py               # DatasetAdapter base class
    source_backed.py      # SourceBackedDatasetAdapter
    multimodal_base.py    # MultimodalDatasetAdapter
    ...                   # concrete dataset adapters
  models/
    base.py               # ModelAdapter base class
    templates.py          # prompt and score helpers
    ...                   # concrete model adapters
  registry/
    core.py               # thread-safe registries and entry-point loading
  execution/
    runner.py             # run_benchmark() orchestration
  benchmarks/
    packs.py              # user-facing pack definitions
    presets.py            # reproducible benchmark preset definitions
  reports/
    summary.py            # HTML and summary rebuild logic
  exports/
    summary.py            # CSV, XLSX, and JSON export helpers
  schemas/
    core.py               # normalized contracts and run manifest models
  plugins/
    discovery.py          # built-in module import helper
```

## Runtime Flow

```text
CLI flags or YAML config
  -> config resolution
  -> registry loading
  -> dataset normalization
  -> model prediction
  -> metrics computation
  -> artifact writing
  -> report / compare / export workflows
```

## Registries And Discovery

The harness maintains separate registries for dataset and model adapters.

At startup, `ensure_builtin_registrations()`:

1. imports built-in dataset modules
2. imports built-in model modules
3. loads entry points from `guard_eval_harness.datasets`
4. loads entry points from `guard_eval_harness.models`

That means external plugins can register new adapters without changing the core
repository, as long as they expose the correct entry points.

## Packs Vs Presets

These concepts are related but not the same:

- packs are user-facing suites meant for `geh run --pack ...`
- presets are code-defined benchmark suites exposed through `geh list presets`

Packs optimize for fast, named starter evaluations. Presets are better thought
of as reproducible benchmark definitions used by higher-level workflows and
reproduction efforts.

## Core Contracts

The most important shared schemas are:

- `NormalizedSample`
- `Message`
- `NormalizedPrediction`
- `DatasetMetadata`
- `AdapterCapabilities`
- `RunManifest`

These are Pydantic models defined in `schemas/core.py`, not dataclasses.

## Design Choices Worth Knowing

### Artifact-Centric Execution

The run directory is the source of truth. Most follow-up workflows operate on
stored artifacts instead of recomputing the benchmark.

### Deterministic Sample IDs

Dataset adapters generate stable IDs so resume, comparison, and debugging stay
predictable.

### Capability-Driven Execution

The runner inspects each adapter's declared capabilities to decide how batching,
concurrency, and modality handling should behave.

### Local And Hosted Backends Share The Same Output Contract

This is what makes comparisons between `hf`, `vllm`, `openai_moderation`,
`openai_compatible`, and plugin-provided adapters practical.
