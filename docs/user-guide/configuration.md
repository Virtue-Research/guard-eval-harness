# Configuration

Every run is driven by a single YAML config:

```bash
geh run --config path/to/run.yaml
```

`geh validate --config path/to/run.yaml` loads the file and confirms every dataset resolves, without running inference.

## Skeleton

```yaml
run_name: my-evaluation             # used in artifact paths
threshold: 0.5                      # score >= threshold => unsafe

model:                              # see "Guard + backend" below
  profile: llama-guard-3-8b

datasets:                           # one or more entries
  - name: xstest
    limit: 100

output:
  run_dir: out/my-evaluation        # required
  resume: true                      # default true
  overwrite: false                  # wipe run_dir before starting
```

## Guard + backend

Two equivalent ways to describe the model.

### Option A — bundled profile (recommended)

```yaml
model:
  profile: granite-guardian-3.2-5b
```

`geh list profiles` shows the 14 known-good profiles. Each is a full `guard + backend + args` payload; reference it by slug and you're done.

Any field you set alongside `profile:` deep-merges onto it — swap the backend, change the model id, tweak `max_new_tokens`, etc.:

```yaml
model:
  profile: llama-guard-3-8b
  backend:
    kind: openai_compat
    name: meta-llama/Llama-Guard-3-8B
    args:
      base_url: http://localhost:8000/v1
      api_key_env: null
```

### Option B — full inline

```yaml
model:
  guard: llm                          # see `geh list guards`
  output_format: safe_unsafe_first_line
  guard_args: {}                      # passed to the Guard constructor
  backend:
    kind: openai_compat               # see `geh list backends`
    name: gpt-4o-mini
    args:
      base_url: https://api.openai.com/v1
      api_key_env: OPENAI_API_KEY
      max_new_tokens: 32
      temperature: 0.0
```

## Datasets

Each entry is a dataset to evaluate. The full field set:

| Field | Default | Description |
|---|---|---|
| `name` | required | Display name; also adapter name unless `adapter:` is set |
| `adapter` | same as `name` | Built-in adapter (see `geh list datasets`) or `local_jsonl`/`local_csv`/`local_image_jsonl`/`local_image_dir` |
| `path` | `null` | Local file/directory (required for `local_*` adapters) |
| `split` | `"test"` | Split name — varies per dataset, see `geh list datasets` |
| `policy` | `null` | Inline `{name, text}` object or a registered policy name |
| `policy_source` | `null` | `"upstream"` · `"generated"` · `"virtue_general"` — dataset-scoped policy lookup |
| `limit` | `null` | Take the first N samples (mutually exclusive with `sample_*`) |
| `sample_ids` | `()` | Run only these sample ids |
| `sample_indices` | `()` | Run only these row indices |
| `options` | `{}` | Adapter-specific options |

Three-tier policy resolution (first match wins): explicit `policy:` → `policy_source:` registry → adapter default → guard default → none.

## Environment variables

Use `${VAR_NAME}` anywhere in a string value:

```yaml
backend:
  args:
    base_url: ${VLLM_BASE_URL}
    api_key_env: OPENAI_API_KEY        # name of the env var holding the key
```

## CLI overrides

```bash
geh run --config run.yaml \
    --threshold 0.6 \
    --output-dir out/custom \
    --overwrite
```

| Flag | Effect |
|---|---|
| `--output-dir DIR` | Override `output.run_dir` |
| `--threshold FLOAT` | Override `threshold` |
| `--no-resume` | Treat the run dir as fresh (errors if non-empty) |
| `--overwrite` | Wipe the run dir before starting |
| `--recompute-metrics` | Skip inference; recompute metrics from existing predictions |

## Resume

The harness fingerprints the resolved config and writes the hash to `manifest.json`. Re-running with the same config picks up where it stopped; mixing in a different config fails fast.

## Output

Every run writes a self-contained directory under `output.run_dir/`. See [Run artifacts](run-artifacts.md) for the full layout.
