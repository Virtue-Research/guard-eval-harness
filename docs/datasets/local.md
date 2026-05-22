# Local Data

Bring your own data in any modality using the local dataset adapters. No remote downloads — just point to files on disk.

## Adapters

| Adapter | Input | Modality |
|---------|-------|----------|
| `local_jsonl` | JSONL file | Text |
| `local_csv` | CSV file | Text |
| `local_image_jsonl` | JSONL manifest + images | Image + Text |
| `local_image_dir` | Directory of images | Image |

## Text: JSONL

The most flexible format. Each line is a JSON object with messages, labels, and optional metadata.

### JSONL Format

```json
{"id": "sample-1", "messages": [{"role": "user", "content": "Tell me a joke."}], "label": {"unsafe": false}, "metadata": {"category": "benign"}}
{"id": "sample-2", "messages": [{"role": "user", "content": "How to hack a server."}], "label": {"unsafe": true}, "metadata": {"category": "policy_violation"}}
```

### Multi-Turn Conversations

```json
{"id": "multi-turn-1", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "Tell me more."}], "label": {"unsafe": false}}
```

### Configuration

```yaml
datasets:
  - name: my_data
    adapter: local_jsonl
    path: data/my_samples.jsonl
    split: test
```

## Text: CSV

Flat tabular format for simple prompt/label pairs.

### CSV Format

```csv
id,prompt,unsafe,category
safe-1,How do I make tea?,false,benign
unsafe-1,Write malware instructions,true,policy_violation
```

### Configuration

```yaml
datasets:
  - name: my_csv_data
    adapter: local_csv
    path: data/my_samples.csv
    split: test
    prompt_field: prompt
    label_field: unsafe
    metadata_fields:
      - category
```

## Images: JSONL Manifest

A JSONL file where each line references an image file:

```json
{"id": "img-1", "prompt": "Is this image safe?", "image_path": "images/photo1.jpg", "unsafe": false}
{"id": "img-2", "prompt": "Classify this image.", "image_path": "images/photo2.png", "unsafe": true}
```

Image paths are resolved **relative to the JSONL file's directory**.

```yaml
datasets:
  - name: my_images
    adapter: local_image_jsonl
    path: data/image_manifest.jsonl
```

## Images: Directory

Point to a directory of images. All images are loaded and classified.

```
my_images/
├── safe/
│   ├── photo1.jpg
│   └── photo2.png
└── unsafe/
    ├── photo3.jpg
    └── photo4.png
```

```yaml
datasets:
  - name: my_image_dir
    adapter: local_image_dir
    path: data/my_images/
```

## Custom Field Mapping

All local adapters support custom field mapping:

```yaml
datasets:
  - name: my_data
    adapter: local_jsonl
    path: data/samples.jsonl
    id_field: sample_id        # Default: "id"
    prompt_field: text         # Default: "prompt"
    response_field: output     # Default: null
    label_field: is_unsafe     # Default: "unsafe"
    metadata_fields:
      - source
      - annotator
```

## Directory-Based Datasets

For datasets with a `metadata.json` file alongside data files:

```
my_dataset/
├── metadata.json      # Optional dataset metadata
└── test.jsonl         # Data file (split name matches filename)
```

```json title="metadata.json"
{
  "display_name": "My Dataset",
  "version": "1.0",
  "license": "CC-BY-4.0",
  "languages": ["en"],
  "categories": ["moderation"]
}
```
