# Text Benchmarks

The core modality — evaluate text-based guardrails and moderation models across a range of safety dimensions.

## Jailbreak & Adversarial

Benchmarks that test resistance to adversarial prompts designed to bypass safety guardrails.

| Dataset | Adapter | Description | Source |
|---------|---------|-------------|--------|
| XSTest v2 | `xstest` | Prompt suite testing exaggerated safety | HuggingFace |
| HarmBench | `harmbench_behaviors` | Behavioral adversarial dataset | HuggingFace |
| JailbreakBench | `jailbreakbench` | Curated jailbreak attempts | HuggingFace |
| JBB Behaviors | `jbb_behaviors` | JailbreakBench behavior set | HuggingFace |
| AdvBench Behaviors | `advbench_behaviors` | Adversarial behavior prompts | HuggingFace |
| AdvBench Strings | `advbench_strings` | Adversarial string attacks | HuggingFace |
| Do-Anything-Now | `do_anything_now_questions` | DAN-style jailbreak prompts | HuggingFace |
| StrongREJECT | `strong_reject_instructions` | Strong rejection instructions | HuggingFace |
| MaliciousInstruct | `malicious_instruct` | Malicious instruction prompts | HuggingFace |
| WildJailbreak | `wildjailbreak` | Wild jailbreak attempts | HuggingFace |
| WildGuardMix | `wildguardmix` | Mixed jailbreak corpus | HuggingFace |
| GuardrailsAI | `guardrailsai_jailbreak` | GuardrailsAI jailbreak set | HuggingFace |

## Toxicity

Benchmarks for detecting toxic, offensive, or harmful language.

| Dataset | Adapter | Description | Source |
|---------|---------|-------------|--------|
| ToxicChat | `toxic_chat` | Multi-turn toxic conversations | HuggingFace |
| ToxiGen | `toxigen` | Machine-generated toxic text | HuggingFace |
| Civil Comments | `civil_comments` | Toxicity in comment threads | HuggingFace |
| Jigsaw Toxicity | `jigsaw_toxicity` | Jigsaw/Perspective toxicity | HuggingFace |
| RealToxicityPrompts | `real_toxicity_prompts` | Prompted toxic completions | HuggingFace |
| OR-Bench | `or_bench` | Over-refusal benchmark | HuggingFace |
| OLID | `olid` | Offensive language identification | HuggingFace |

## Hate Speech & Harassment

Benchmarks focused on hate speech, bias, and harassment detection.

| Dataset | Adapter | Description | Source |
|---------|---------|-------------|--------|
| HateCheck | `hatecheck` | Functional hate speech tests | HuggingFace |
| HateMoji Check | `hatemoji_check` | Emoji-based hate speech | HuggingFace |
| DynaHate | `dynahate` | Dynamic hate speech dataset | HuggingFace |
| ETHOS | `ethos` | Online hate speech | HuggingFace |
| HatExplain | `hatexplain` | Explainable hate speech | HuggingFace |
| Implicit Hate | `implicit_hate` | Implicit hate corpus | HuggingFace |
| Measuring Hate Speech | `measuring_hate_speech` | Hate speech measurement | HuggingFace |
| Hate Speech Offensive | `hate_speech_offensive` | Hate vs offensive language | HuggingFace |
| Social Bias Frames | `social_bias_frames` | Social bias detection | HuggingFace |
| ConvAbuse | `convabuse` | Conversational abuse | HuggingFace |

## General Safety

Broad safety evaluation benchmarks.

| Dataset | Adapter | Description | Source |
|---------|---------|-------------|--------|
| BeaverTails 330k | `beaver_tails_330k` | Large-scale safety dataset | HuggingFace |
| Do-Not-Answer | `do_not_answer` | Questions that should be refused | HuggingFace |
| Harmful QA | `harmful_qa` | Harmful question-answer pairs | HuggingFace |
| PKU Safe RLHF | `pku_safe_rlhf` | PKU safe RLHF dataset | HuggingFace |
| GuardBench | `guardbench` | General guardrail benchmark | HuggingFace |
| CircleGuardBench | `circleguardbench` | Circle guard evaluation | HuggingFace |
| OpenAI Moderation | `openai_moderation_dataset` | OpenAI moderation dataset | HuggingFace |

## Prompt Injection

Benchmarks for testing resistance to prompt injection attacks.

| Dataset | Adapter | Description | Source |
|---------|---------|-------------|--------|
| Prompt injection benchmarks | Various | Input-filtering guardrail tests | Various |

## Usage Examples

```yaml
# minimal-run.yaml
run_name: text-baseline
model:
  profile: granite-guardian-3.2-5b
datasets:
  - name: xstest
  - name: toxic_chat
    n_samples: 500
  - name: harmful_qa
    n_samples: 1000
output:
  run_dir: out/text-baseline
```

```bash
uv run geh run --config minimal-run.yaml
```

```yaml
datasets:
  - name: xstest
    adapter: xstest
    split: test

  - name: toxic_chat
    adapter: toxic_chat
    split: test
```
