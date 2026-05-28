# Troubleshooting

The issues most likely to bite a first-time user: missing extras, auth, path
resolution, and resource limits.

## `geh` command not found

Make sure you ran the install in the venv you're using right now:

```bash
uv sync --extra hf            # or: pip install -e ".[hf]"
uv run geh --help             # or activate the venv and run `geh --help`
```

## A backend or guard isn't listed by `geh list ...`

You probably need an extra. Inspect what's available:

```bash
uv run geh list backends
uv run geh list guards
uv run geh list profiles
```

Common extras:

- `--extra hf` → `hf_generate`, `hf_text_classifier`, `hf_image_classifier`, `hf_vlm` backends
- `--extra api` → tenacity-based retry for hosted endpoints (OpenAI / vLLM / any `openai_compat`)
- `--extra dev` → tests + lint

## OpenAI or gated HuggingFace auth fails

Check the relevant env vars are exported in the same shell as `geh`:

```bash
echo "$OPENAI_API_KEY"
echo "$HF_TOKEN"
```

For gated HF repos (Llama Guard, WildGuard, …) you also need to accept the
license on the model's HuggingFace page once.

## Reasoning OpenAI models reject `temperature` / `max_tokens`

GPT-5-class reasoning models use `max_completion_tokens` and don't accept
`temperature`. Override in the backend args:

```yaml
backend:
  kind: openai_compat
  args:
    token_param: max_completion_tokens
    omit_temperature: true
    reasoning_effort: medium       # optional
```

## `geh validate` fails on a local dataset path

Paths in YAML are resolved **relative to the YAML file**, not the shell's
working directory. If `examples/foo.yaml` says `path: data/x.jsonl`, the file
must live at `examples/data/x.jsonl`. Use absolute paths to remove ambiguity.

## Dataset split errors

Many adapters don't ship a `test` split — set `split:` explicitly if you see
`Unknown split 'test'. Should be one of [...]`. `geh list datasets` shows the
supported splits per adapter.

## GPU OOM / runs too slow

- Pick a smaller profile (e.g. `granite-guardian-3.2-5b` instead of `gemma4-31b-it`).
- For local models, set `device: cuda:N` in `backend.args` to pin to a specific GPU.
- For HF text generation, reduce `max_new_tokens` if the guard only needs a
  short verdict.

## A run stops partway through

Just re-run with the same config — `resume: true` is the default. The harness
hashes the resolved config and only resumes if it matches.

```bash
geh run --config run.yaml                   # resumes
geh run --config run.yaml --overwrite       # wipe + restart
geh run --config run.yaml --no-resume       # error if non-empty
```

## Known-good smoke tests

```bash
uv run geh run --config examples/mock-jsonl.yaml          # no GPU, no keys
uv run geh run --config examples/hf-text-classifier.yaml  # GPU, ~30s
uv run geh run --config examples/openai-judge.yaml        # needs OPENAI_API_KEY
```
