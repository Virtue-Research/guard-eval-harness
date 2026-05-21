# Benchmark Packs

Benchmark packs are curated bundles of datasets designed for common evaluation
scenarios. They are the fastest way to move from "I installed the tool" to "I
have a meaningful safety baseline."

## How Packs Behave

- `geh run --pack core --model ...` is the public shorthand
- versioned aliases like `core-v1` also work
- packs keep dataset ordering and preset split/options stable

See the currently registered packs with:

```bash
geh list packs
```

## Available Packs

| Pack | Datasets | Best for |
| --- | ---: | --- |
| `core` | 12 | broad text-safety baseline |
| `jailbreak` | 10 | adversarial prompt and refusal robustness |
| `toxicity` | 7 | moderation quality and toxicity filtering |
| `hate_harassment` | 11 | abuse, harassment, and hate-speech detection |
| `prompt_injection` | 6 | prompt override and policy-evasion attacks |
| `audio` | 1 | native audio safety models |

## Pack Details

### `core`

Broad text-safety starter pack covering jailbreaks, toxicity, harmful content,
and abuse. Use this when you want one suite to represent overall guardrail
health.

Representative datasets include:

- `xstest`
- `harmbench_behaviors`
- `toxic_chat`
- `beaver_tails_330k`
- `do_not_answer`

### `jailbreak`

Focused on adversarial instructions and attack-style prompts. Use this when the
question is "does the model stay aligned under pressure?"

Representative datasets include:

- `advbench_behaviors`
- `advbench_strings`
- `jailbreakbench`
- `wildjailbreak`
- `do_anything_now_questions`

### `toxicity`

Focused on moderation quality, especially over-blocking vs. under-blocking.

Representative datasets include:

- `civil_comments`
- `jigsaw_toxicity`
- `toxigen`
- `real_toxicity_prompts`
- `toxic_chat`

### `hate_harassment`

More targeted than `toxicity` when you care specifically about abuse, hate
speech, conversational harassment, or bias-sensitive content.

Representative datasets include:

- `hatecheck`
- `dynahate`
- `hatexplain`
- `social_bias_frames`
- `convabuse`

### `prompt_injection`

Designed for inputs that try to override system intent, abuse retrieval
contexts, or push the model into policy-breaking behavior.

Representative datasets include:

- `hex_phi`
- `i_malicious_instructions`
- `mitre`
- `tdc_red_teaming`

### `audio`

A minimal audio-first entry point for models that operate directly on audio
instead of relying on text-only moderation.

## Usage

```bash
geh run --pack core --model mock
geh run --pack jailbreak --model hf --model-name meta-llama/Llama-Guard-3-8B
geh run --pack toxicity --model openai_moderation --threshold 0.6
```

## What A Pack Stores

Each pack carries:

- a stable `name` and `version`
- an ordered dataset list
- any pack-specific split or option overrides
- optional expected sample counts or recommended thresholds

Use packs when you want a shared starter suite. Use YAML configs when you want
full control over every dataset entry.
