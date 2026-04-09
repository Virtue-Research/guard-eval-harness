# Run Modes

Guard Eval Harness supports three primary run modes. Pick the one that matches
how repeatable the run needs to be.

## At A Glance

| Mode | Best for | Command shape | Tradeoff |
| --- | --- | --- | --- |
| Inline | quick checks and local iteration | `geh run --dataset ... --model ...` | fastest, but least shareable |
| Benchmark pack | curated starter suites | `geh run --pack ... --model ...` | opinionated coverage, less granular control |
| YAML config | repeatable team workflows | `geh run --config path.yaml` | most explicit, but more verbose |

## Inline Mode

Use inline mode when you want one command and minimal setup.

```bash
geh run --dataset xstest,toxic_chat \
    --model hf \
    --model-name meta-llama/Llama-Guard-3-8B \
    --batch-size 16
```

Choose this when:

- you are validating an install
- you are comparing a few backends quickly
- you do not need a committed config file yet

## Benchmark Packs

Packs are curated bundles of datasets with stable names such as `core`,
`jailbreak`, `toxicity`, and `prompt_injection`.

```bash
geh run --pack core --model openai_moderation --limit 100
```

Choose this when:

- you want a sensible default suite without hand-picking datasets
- you want a benchmark label that teammates can repeat
- you want versioned aliases such as `core-v1` behind stable shorthand names
  like `core`

## YAML Configs

YAML configs are the best choice when a run needs to be committed, reviewed, or
reused.

```bash
geh run --config examples/run-mock-jsonl.yaml
```

Use YAML when:

- you want stable output locations
- you need backend-specific arguments
- you want local dataset paths, field mappings, or multiple datasets in one file
- you want to review the configuration in code review

## A Good Upgrade Path

Most teams end up with this progression:

1. Start with an inline smoke test.
2. Move to a pack when you want more meaningful coverage quickly.
3. Promote the run to YAML when the setup should become repeatable.

## Related Commands

```bash
geh list datasets
geh list backends
geh list packs
geh validate --config examples/run-mock-jsonl.yaml
```

## Next Step

Use [Benchmark Selection](../user-guide/benchmark-selection.md) once you know
which run mode you want.
