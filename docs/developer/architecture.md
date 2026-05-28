# Architecture

A run is exactly four things glued by the runner:

```
       Guard               Backend
   (prompt + parser)   (inference engine)
        │                    │
        ▼                    ▼
sample ──► messages ──► raw output ──► ParsedLabel ──► metrics
                                            ▲
                                            │
                                       Policy + OutputFormat
                                       (LLM-as-guard only)
```

- **Guard** — owns prompt construction and output parsing. Fixed-taxonomy
  guards (Llama Guard, ShieldGemma) embed their taxonomy. The generic `llm`
  guard accepts any Policy + OutputFormat combo.
- **Backend** — the inference engine (`hf_generate`, `openai_compat`,
  `hf_text_classifier`, `hf_image_classifier`, `hf_vlm`, `mock`). It knows
  nothing about safety taxonomies.
- **Dataset** — loads samples in a normalized shape (`NormalizedSample`).
- **Runner** — glues them together, streams predictions to disk, and
  computes metrics. Resumable by default.

## Project layout

```text
src/guard_eval_harness/
  cli.py                  # argparse entry point (`geh`)
  config/
    loading.py            # YAML resolution + profile expansion
    schema.py             # resolved-config Pydantic models
  registry.py             # thread-safe registries (datasets/guards/backends/...)
  schemas.py              # NormalizedSample, PredictSample, ParsedLabel, …
  datasets/               # one module per adapter
  guards/
    base.py               # Guard base class
    profiles/             # bundled `model:` blocks (slug -> YAML)
  backends/               # hf_generate, openai_compat, hf_text_classifier, …
  output_formats/         # parsers for guard outputs
  policies/               # built-in policies + dataset-scoped registries
  runner.py               # run_from_config orchestration
  metrics/                # binary-classification metrics
```

## Runtime flow

```text
YAML config
  -> ensure_builtin_registrations()
  -> profile expansion (deep-merge model overrides)
  -> resolved config + SHA-256 hash
  -> for each dataset:
       load samples -> apply subset filter -> stream predictions
       -> parse output -> compute per-dataset metrics
  -> write manifest, summary, dataset artifacts
```

## Registries and discovery

A single `Registry[T]` class backs each component type. `ensure_builtin_registrations()` imports the built-in modules (side-effect registers each class) and then loads any entry points published under:

- `guard_eval_harness.datasets`
- `guard_eval_harness.guards`
- `guard_eval_harness.backends`
- `guard_eval_harness.output_formats`

External plugins can ship new adapters without forking — declare an entry
point and the harness picks it up.

## Profiles

A profile is a complete `model:` block stored as YAML under
`guards/profiles/<slug>.yaml`. The config loader resolves
`model.profile: <slug>` to a full payload, then deep-merges any sibling keys
on top. This is the recommended user-facing path; the underlying
`guard + backend` schema is always available for fully inline configs.

## Three-tier policy resolution

For each dataset, the effective policy is the first of these that resolves:

1. `policy:` set on the dataset entry (inline or registry name)
2. `policy_source: upstream|generated|virtue_general` → registry lookup
3. adapter's `default_policy_source`
4. guard's `default_policy`
5. none

## Core contracts (Pydantic models)

- `NormalizedSample` — adapter output (id, messages, label, metadata)
- `PredictSample` — sample minus label (handed to guards/backends; structurally label-free)
- `ParsedLabel` — `(unsafe: bool, score: float, categories: list[str])`
- `Message`, `Part` — chat-message + multimodal parts
- `RunManifest` — top-level run metadata + config hash

## Design choices worth knowing

### Artifact-centric execution

The run directory is the source of truth. Metrics, comparisons, and exports
operate on stored predictions, not in-memory state.

### Ground-truth leak-proof

`PredictSample` structurally excludes labels. A
`PREDICT_METADATA_FIELD_DENYLIST` rejects label-shaped metadata fields at the
config boundary so they can't be smuggled to a model.

### Deterministic sample IDs

Adapters generate stable IDs. Resume, comparison, and debugging stay
predictable across runs.

### Resume by config hash

`manifest.json` carries a SHA-256 of the resolved config. Re-running with the
same config picks up at the next unprocessed sample; running with a different
config in a non-empty dir fails fast.
