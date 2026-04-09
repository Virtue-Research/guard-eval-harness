# HTTP Endpoint Adapter

The `http` adapter sends POST requests to any HTTP endpoint, making it easy to evaluate custom safety models behind a REST API.

## Requirements

```bash
pip install -e ".[api]"
```

## Configuration

```yaml
model:
  adapter: http
  args:
    endpoint_url: https://my-safety-api.example.com/classify
    request_template:
      text: "{prompt}"
      options:
        threshold: 0.5
    response_mapping:
      score_path: "result.unsafe_score"
      label_path: "result.is_unsafe"
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `endpoint_url` | `str` | **required** | API endpoint URL |
| `request_template` | `dict` | **required** | JSON body template with `{prompt}` placeholders |
| `response_mapping` | `dict` | **required** | JSONPath-like mapping for extracting scores |
| `headers` | `dict` | `{}` | Custom HTTP headers |
| `concurrency` | `int` | `1` | Parallel requests |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

## Request Template

The `request_template` defines the JSON body sent to the endpoint. Use `{prompt}` and `{messages_text}` as placeholders:

```yaml
args:
  request_template:
    messages:
      - role: user
        content: "{prompt}"
    model: my-safety-model
```

## Response Mapping

The `response_mapping` tells the adapter how to extract the safety score from the API response:

```yaml
args:
  response_mapping:
    score_path: "data.scores.unsafe"    # Dot-notation path to score
    label_path: "data.is_unsafe"        # Path to boolean label
```

## Example: Custom Safety API

```yaml
version: 1
run_name: custom-api-eval

model:
  adapter: http
  args:
    endpoint_url: http://localhost:5000/v1/moderate
    request_template:
      content: "{messages_text}"
    response_mapping:
      score_path: "unsafe_probability"

datasets:
  - name: xstest
    adapter: xstest

execution:
  concurrency: 8

output:
  run_dir: out/custom-api-eval
```

## Capabilities

| Capability | Supported |
|-----------|:---------:|
| Probability scores | Yes (if API returns scores) |
| Batching | No |
| Concurrency | Yes |
| Category outputs | No |
| Input modalities | Text |
