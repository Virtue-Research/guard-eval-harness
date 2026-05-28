# Benchmark selection

There's no `--pack` shortcut — pick the datasets you actually care about and
list them under `datasets:` in your YAML. Use this page as the cheat-sheet
mapping from "what I care about" to "which datasets cover it."

## Cheat sheet

| If you care about… | Start with these datasets |
| --- | --- |
| **Broad text-safety baseline** | `xstest`, `toxic_chat`, `harmbench_behaviors`, `do_not_answer`, `beaver_tails_330k` |
| **Jailbreak / adversarial resistance** | `harmbench_behaviors`, `jbb_behaviors`, `jailbreakbench`, `advbench_behaviors`, `wildjailbreak`, `strong_reject_instructions` |
| **Prompt injection** | `prompt_guard` (model name; pair with prompt-injection datasets), `guardrailsai_jailbreak`, `do_anything_now_questions` |
| **Toxicity / moderation** | `jigsaw_toxicity`, `toxic_chat`, `toxigen`, `civil_comments`, `real_toxicity_prompts`, `tweet_eval_hate` |
| **Hate & harassment** | `hatecheck`, `hatemoji_check`, `implicit_hate`, `measuring_hate_speech`, `social_bias_frames`, `convabuse` |
| **General Q&A safety** | `harmful_qa`, `harmful_q`, `do_not_answer`, `pku_safe_rlhf`, `sorry_bench`, `salad_bench` |
| **Refusal calibration (over-blocking)** | `xstest`, `or_bench`, `i_controversial`, `safe_text` |
| **Image safety** | `unsafebench`, `holisafebench`, `jailbreakv`, `mm_safetybench`, `vlsbench` |
| **Local / bring-your-own** | `local_jsonl`, `local_csv`, `local_image_jsonl`, `local_image_dir` |

Run `uv run geh list datasets` for the full list of 80+ adapters, and check
[`docs/datasets/`](../datasets/overview.md) for adapter-by-adapter details.

## Pick a profile to match

| Task shape | Recommended profile |
| --- | --- |
| Fast text classifier (small, GPU) | `prompt-guard-86m`, `llama-prompt-guard-22m`, `qwen3guard-0.6b` |
| Strong general-purpose safety classifier | `granite-guardian-3.2-5b`, `qwen3guard-4b`, `wildguard` |
| Llama Guard family | `llama-guard-3-1b`, `llama-guard-3-8b` |
| Single-axis policy expert | `shieldgemma-9b` (override its principle) |
| LLM-as-judge | `gpt-4o-mini`, `gpt-5.4-mini`, or build your own with `guard: llm` |
| Image safety | `llavaguard` guard, `hf_image_classifier` backend |

`uv run geh list profiles` shows everything bundled.

## Mix built-in and local data

The harness is strongest when public benchmarks and your real traffic share
one config:

```yaml
datasets:
  - name: xstest
    limit: 100
  - name: my_internal_refusals
    adapter: local_jsonl
    path: data/refusals.jsonl
```

## Related

- [Datasets overview](../datasets/overview.md)
- [Common workflows](common-workflows.md)
- [Configuration reference](configuration.md)
