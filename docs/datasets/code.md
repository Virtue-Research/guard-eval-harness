# Code Benchmarks

Evaluate models that detect insecure or vulnerable code patterns.

## Available Benchmarks

| Dataset | Adapter | Description | Languages |
|---------|---------|-------------|-----------|
| VulnLLM-R (Function Level) | `vulnllm_r_function_level` | Function-level vulnerability detection | C, Python, Java |
| VulnLLM-R (Repo Level) | `vulnllm_r_repo_level` | Repository-level vulnerability detection | Java |

## Usage Example

```yaml title="examples/code-vuln/run-vulnllm-r-openai.yaml"
version: 1
run_name: code-vuln-gpt4o

model:
  adapter: openai_compatible
  model_name: gpt-4o
  args:
    root_url: https://api.openai.com
    api_key_env: OPENAI_API_KEY
    concurrency: 8
    max_tokens: 8192
    temperature: 0.0

datasets:
  - name: vulnllm_r_function_level_c
    adapter: vulnllm_r_function_level
    split: function_level
    options:
      language: c
      use_cot: true
      use_policy: true

  - name: vulnllm_r_function_level_python
    adapter: vulnllm_r_function_level
    split: function_level
    options:
      language: python
      use_cot: true
      use_policy: true

  - name: vulnllm_r_function_level_java
    adapter: vulnllm_r_function_level
    split: function_level
    options:
      language: java
      use_cot: true
      use_policy: true

  - name: vulnllm_r_repo_level
    adapter: vulnllm_r_repo_level
    split: repo_level
    options:
      language: java
      use_cot: true
      use_policy: true

execution:
  concurrency: 8

output:
  run_dir: out/code-vuln/gpt4o
```

## Dataset Options

### VulnLLM-R

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | `str` | **required** | Programming language (`c`, `python`, `java`) |
| `use_cot` | `bool` | `false` | Enable chain-of-thought reasoning |
| `use_policy` | `bool` | `false` | Include security policy context |

## Code Vulnerability Metrics

Code benchmarks use specialized metrics beyond standard binary classification:

- Standard binary metrics (accuracy, precision, recall, F1)
- Vulnerability-type breakdown (SQL injection, XSS, buffer overflow, etc.)

See [Metrics](../metrics/overview.md) for details.
