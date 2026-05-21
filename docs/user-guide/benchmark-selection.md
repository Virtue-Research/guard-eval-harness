# Benchmark Selection

The fastest way to get useful results is to choose a benchmark path that
matches your deployment risk, not just the model you happen to be testing.

## Start Here

| If you care about... | Start with | Why |
| --- | --- | --- |
| broad text-safety coverage | `core` pack | balanced baseline across jailbreak, toxicity, harmful Q&A, and abuse |
| jailbreak resistance | `jailbreak` pack | focuses on adversarial prompts and refusal robustness |
| prompt injection style attacks | `prompt_injection` pack | better fit for policy override and red-team inputs |
| moderation quality | `toxicity` or `hate_harassment` | better signal for over-blocking and abuse detection |
| image safety | `safe_vs_unsafe_image_edits`, `unsafebench`, or local image adapters | lets you test multimodal behavior directly |
| native audio safety | `audio` pack | built for audio-first moderation models |

## Recommended Starting Paths

### Broad OSS Baseline

```bash
geh run --pack core --model mock
```

Good when you need a first benchmark suite before optimizing anything.

### Prompt Attack Or Red-Team Work

```bash
geh run --pack prompt_injection --model openai_compatible \
    --model-name gpt-4.1-mini
```

Use this when your system is vulnerable to instruction overrides, tool misuse,
or retrieval-time prompt attacks.

### Policy Moderation Work

```bash
geh run --pack toxicity --model openai_moderation
```

Use `hate_harassment` when the core deployment risk is abusive language and
bias-sensitive content.

### Multimodal Safety

Start with a single dataset before scaling up:

- `safe_vs_unsafe_image_edits` for image-edit request moderation
- `unsafebench` for broader unsafe image safety checks
- `local_image_jsonl` when you already have internal or bespoke image prompts

## How To Decide Between Packs And Individual Datasets

Choose packs when:

- you want a stable shared starting point
- you are benchmarking several models the same way
- you want a named suite in reports or docs

Choose individual datasets when:

- you need one domain only
- you are doing targeted debugging
- you want to mix built-ins with local data

## Mix Built-In And Local Data

The harness is strongest when you pair public benchmarks with the data your
system actually sees.

Examples:

- `xstest` plus `local_jsonl` for a refusal model
- `unsafebench` plus `local_image_jsonl` for a multimodal guardrail

## Related References

- [Benchmark Packs](benchmark-packs.md)
- [Datasets Overview](../datasets/overview.md)
- [Common Workflows](common-workflows.md)
