# Changelog

All notable changes to `geh` (guard-eval-harness) are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - TBD

Initial public release.

### Added
- CLI-first harness for benchmarking guardrail, moderation, and safety
  classification models against 80+ built-in datasets.
- Adapters for HuggingFace, vLLM (offline + HTTP), OpenAI, Anthropic, and
  generic API endpoints.
- Built-in benchmark packs and YAML-driven run configs.
- HTML / JSON reporting and Google Sheets export.
- Model catalog with curated configurations for popular safety models
  (Llama Guard 3, ShieldGemma, WildGuard, Qwen3-Guard, MD-Judge, and
  more).

[Unreleased]: https://github.com/Virtue-Research/guard-eval-harness/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Virtue-Research/guard-eval-harness/releases/tag/v0.1.0
