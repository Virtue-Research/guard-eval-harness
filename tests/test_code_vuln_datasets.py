"""Tests for code vulnerability detection dataset adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guard_eval_harness.datasets.code_vuln.prompts import (
    CWE_CONSTRAINTS,
    SYSTEM_PROMPTS,
    build_code_vuln_prompt,
    parse_judge_response,
    resolve_system_prompt,
)
from guard_eval_harness.models.templates import score_from_text


# --- Prompt construction ---


class TestBuildCodeVulnPrompt:
    """Tests for build_code_vuln_prompt."""

    def test_basic_prompt_includes_code(self):
        prompt = build_code_vuln_prompt(
            code="int x = 0;",
            cwe_list=["CWE-125"],
        )
        assert "int x = 0;" in prompt

    def test_cot_included_by_default(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79"],
        )
        assert "Step 1:" in prompt

    def test_cot_disabled(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79"],
            use_cot=False,
        )
        assert "Step 1:" not in prompt

    def test_policy_included_by_default(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79", "CWE-89"],
        )
        assert "CWE-79" in prompt
        assert "CWE-89" in prompt
        assert "focus on checking" in prompt

    def test_policy_disabled(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79"],
            use_policy=False,
        )
        assert "focus on checking" not in prompt

    def test_cwe_constraint_hints(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-125"],
            use_cwe_constraint=True,
        )
        assert "buffers or arrays" in prompt

    def test_answer_format_present(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79"],
        )
        assert "#judge:" in prompt
        assert "#type:" in prompt

    def test_no_cwe_list(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=None,
            use_policy=True,
        )
        assert "focus on checking" not in prompt

    def test_cwe_descriptions_in_policy(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-125"],
        )
        assert "Out-of-bounds Read" in prompt

    def test_reasoning_suffix_present(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-79"],
        )
        assert "STRICTLY structure" in prompt

    def test_long_context_template(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-125"],
            has_stack_trace=True,
        )
        assert "stack trace" in prompt
        assert "target function" in prompt

    def test_normal_template_no_stack_trace_mention(self):
        prompt = build_code_vuln_prompt(
            code="x = 1",
            cwe_list=["CWE-125"],
            has_stack_trace=False,
        )
        assert "stack trace" not in prompt


# --- Response parsing ---


class TestParseJudgeResponse:
    """Tests for parse_judge_response."""

    def test_yes_response(self):
        result = parse_judge_response(
            "## Final Answer\n#judge: yes\n#type: CWE-125"
        )
        assert result["vulnerable"] is True
        assert result["unsafe_score"] == 1.0
        assert result["cwe"] == "CWE-125"

    def test_no_response(self):
        result = parse_judge_response("## Final Answer\n#judge: no\n#type: N/A")
        assert result["vulnerable"] is False
        assert result["unsafe_score"] == 0.0
        assert result["cwe"] is None

    def test_case_insensitive(self):
        result = parse_judge_response("#Judge: YES\n#Type: CWE-79")
        assert result["vulnerable"] is True
        assert result["cwe"] == "CWE-79"

    def test_no_match(self):
        result = parse_judge_response("The code looks safe to me.")
        assert result["vulnerable"] is None
        assert result["unsafe_score"] is None
        assert result["cwe"] is None

    def test_embedded_in_reasoning(self):
        text = (
            "After careful analysis...\n"
            "The buffer access at line 5 is unchecked.\n"
            "## Final Answer\n"
            "#judge: yes\n"
            "#type: CWE-787\n"
        )
        result = parse_judge_response(text)
        assert result["vulnerable"] is True
        assert result["cwe"] == "CWE-787"

    def test_uses_last_judge_not_first(self):
        text = (
            "It could be #judge: yes or #judge: no.\n"
            "## Final Answer\n"
            "#judge: no\n"
            "#type: N/A\n"
        )
        result = parse_judge_response(text)
        assert result["vulnerable"] is False

    def test_uses_last_type_not_first(self):
        text = (
            "Possible #type: CWE-79 but after review:\n"
            "## Final Answer\n"
            "#judge: yes\n"
            "#type: CWE-125\n"
        )
        result = parse_judge_response(text)
        assert result["cwe"] == "CWE-125"

    def test_na_type(self):
        result = parse_judge_response("#judge: no\n#type: n/a")
        assert result["vulnerable"] is False
        assert result["cwe"] is None

    def test_multiple_cwes_are_not_coerced_to_one(self):
        result = parse_judge_response("#judge: yes\n#type: CWE-79, CWE-89")
        assert result["vulnerable"] is True
        assert result["cwe"] is None


# --- Score resolution integration ---


class TestScoreFromTextJudge:
    """Tests for #judge: pattern in score_from_text."""

    def test_judge_yes(self):
        assert score_from_text("#judge: yes") == 1.0

    def test_judge_no(self):
        assert score_from_text("#judge: no") == 0.0

    def test_judge_in_longer_text(self):
        text = (
            "Analysis complete.\n## Final Answer\n#judge: yes\n#type: CWE-416"
        )
        assert score_from_text(text) == 1.0

    def test_judge_case_insensitive(self):
        assert score_from_text("#Judge: NO") == 0.0

    def test_uses_last_judge_match(self):
        text = "#judge: yes or no\n## Final Answer\n#judge: no"
        assert score_from_text(text) == 0.0

    def test_no_judge_falls_through(self):
        assert score_from_text("0.75") == 0.75


# --- Local dataset adapter ---


class TestLocalCweJsonDataset:
    """Tests for the local CWE JSON dataset adapter."""

    @pytest.fixture()
    def dataset_dir(self, tmp_path: Path) -> Path:
        """Create a minimal local CWE JSON dataset."""
        c_dir = tmp_path / "c"
        c_dir.mkdir()
        python_dir = tmp_path / "python"
        python_dir.mkdir()

        c_samples = [
            {
                "code": "void f() { free(p); *p = 1; }",
                "target": 1,
                "cwe": ["CWE-416"],
                "idx": 1001,
            },
            {
                "code": "void g() { if (p) *p = 1; }",
                "target": 0,
                "cwe": ["CWE-416"],
                "idx": 1002,
            },
        ]
        (c_dir / "CWE-416.json").write_text(json.dumps(c_samples))

        py_samples = [
            {
                "code": "eval(user_input)",
                "target": 1,
                "cwe": ["CWE-94"],
                "idx": 2001,
            },
        ]
        (python_dir / "CWE-94.json").write_text(json.dumps(py_samples))

        return tmp_path

    def test_load_all_languages(self, dataset_dir: Path):
        from guard_eval_harness.config.models import (
            ResolvedDatasetConfig,
        )
        from guard_eval_harness.datasets.code_vuln.local_cwe_json import (
            LocalCweJsonDataset,
        )

        config = ResolvedDatasetConfig(
            name="test_code_vuln",
            adapter="code_vuln_local",
            path=dataset_dir.as_posix(),
            split="test",
        )
        adapter = LocalCweJsonDataset(config)
        samples = adapter.load()
        assert len(samples) == 3

        unsafe_count = sum(1 for s in samples if s.label.unsafe)
        assert unsafe_count == 2

        categories = {cat for s in samples for cat in s.category_labels}
        assert "CWE-416" in categories
        assert "CWE-94" in categories

    def test_language_filter(self, dataset_dir: Path):
        from guard_eval_harness.config.models import (
            ResolvedDatasetConfig,
        )
        from guard_eval_harness.datasets.code_vuln.local_cwe_json import (
            LocalCweJsonDataset,
        )

        config = ResolvedDatasetConfig(
            name="test_code_vuln",
            adapter="code_vuln_local",
            path=dataset_dir.as_posix(),
            split="test",
            options={"language": "python"},
        )
        adapter = LocalCweJsonDataset(config)
        samples = adapter.load()
        assert len(samples) == 1
        assert samples[0].label.unsafe is True

    def test_cwe_filter(self, dataset_dir: Path):
        from guard_eval_harness.config.models import (
            ResolvedDatasetConfig,
        )
        from guard_eval_harness.datasets.code_vuln.local_cwe_json import (
            LocalCweJsonDataset,
        )

        config = ResolvedDatasetConfig(
            name="test_code_vuln",
            adapter="code_vuln_local",
            path=dataset_dir.as_posix(),
            split="test",
            options={"cwe": ["CWE-94"]},
        )
        adapter = LocalCweJsonDataset(config)
        samples = adapter.load()
        assert len(samples) == 1

    def test_messages_have_system_and_user(self, dataset_dir: Path):
        from guard_eval_harness.config.models import (
            ResolvedDatasetConfig,
        )
        from guard_eval_harness.datasets.code_vuln.local_cwe_json import (
            LocalCweJsonDataset,
        )

        config = ResolvedDatasetConfig(
            name="test_code_vuln",
            adapter="code_vuln_local",
            path=dataset_dir.as_posix(),
            split="test",
        )
        adapter = LocalCweJsonDataset(config)
        samples = adapter.load()
        for sample in samples:
            assert len(sample.messages) == 2
            assert sample.messages[0].role == "system"
            assert sample.messages[1].role == "user"
            assert "#judge:" in sample.messages[1].text_content

    def test_limit_option(self, dataset_dir: Path):
        from guard_eval_harness.config.models import (
            ResolvedDatasetConfig,
        )
        from guard_eval_harness.datasets.code_vuln.local_cwe_json import (
            LocalCweJsonDataset,
        )

        config = ResolvedDatasetConfig(
            name="test_code_vuln",
            adapter="code_vuln_local",
            path=dataset_dir.as_posix(),
            split="test",
            options={"limit": 1},
        )
        adapter = LocalCweJsonDataset(config)
        samples = adapter.load()
        assert len(samples) == 1


# --- Pack registration ---


class TestCodeVulnPack:
    """Tests for the code_vuln pack."""

    def test_pack_registered(self):
        from guard_eval_harness.benchmarks.packs import (
            get_pack,
            list_packs,
        )

        assert "code_vuln" in list_packs()
        pack = get_pack("code_vuln")
        assert pack.name == "code_vuln"
        assert pack.version == "v1"

    def test_pack_datasets(self):
        from guard_eval_harness.benchmarks.packs import (
            get_pack,
        )

        pack = get_pack("code_vuln")
        names = pack.dataset_names()
        assert "vulnllm_r_function_level_c" in names
        assert "vulnllm_r_function_level_python" in names
        assert "vulnllm_r_function_level_java" in names
        assert "vulnllm_r_repo_level" in names


# --- CWE constraints coverage ---


class TestCweConstraints:
    """Verify CWE constraint dict is well-formed."""

    def test_all_keys_are_valid_cwe_ids(self):
        import re

        pattern = re.compile(r"^CWE-\d+$")
        for key in CWE_CONSTRAINTS:
            assert pattern.match(key), f"invalid CWE key: {key}"

    def test_all_values_are_nonempty_strings(self):
        for key, value in CWE_CONSTRAINTS.items():
            assert isinstance(value, str)
            assert len(value) > 10, f"constraint too short for {key}"


# --- System prompt selection ---


class TestResolveSystemPrompt:
    """Tests for model-specific system prompt selection."""

    def test_qwen_model(self):
        prompt = resolve_system_prompt(model_name="Qwen/Qwen2.5-7B-Instruct")
        assert "Qwen" in prompt
        assert "Alibaba" in prompt

    def test_qwq_model(self):
        prompt = resolve_system_prompt(model_name="Qwen/QwQ-32B")
        assert "Qwen" in prompt

    def test_deepseek_r1_distill(self):
        prompt = resolve_system_prompt(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        )
        assert "Qwen" in prompt

    def test_deepseek_reasoner(self):
        prompt = resolve_system_prompt(model_name="deepseek-reasoner")
        assert "code assistant" in prompt

    def test_sky_t1(self):
        prompt = resolve_system_prompt(
            model_name="NovaSky-AI/Sky-T1-32B-Preview"
        )
        assert "begin_of_thought" in prompt

    def test_gpt_uses_no_system_prompt(self):
        prompt = resolve_system_prompt(model_name="gpt-4o")
        assert prompt is None

    def test_explicit_key_overrides(self):
        prompt = resolve_system_prompt(
            model_name="gpt-4o",
            system_prompt_key="qwen",
        )
        assert "Qwen" in prompt

    def test_none_returns_default(self):
        prompt = resolve_system_prompt()
        assert prompt == SYSTEM_PROMPTS["default"]


# --- CWE-aware metrics ---


class TestCodeVulnMetrics:
    """Tests for VulnLLM-R CWE-aware metrics."""

    def test_tp_requires_cwe_match(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import (
            NormalizedPrediction,
            NormalizedSample,
        )

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-125",),
            ),
        ]
        # Correct CWE → TP
        preds_correct = [
            NormalizedPrediction(
                sample_id="s1",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=("CWE-125",),
            ),
        ]
        m = compute_code_vuln_metrics(samples, preds_correct)
        assert m["tp"] == 1
        assert m["fn"] == 0

    def test_wrong_cwe_is_fn(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import (
            NormalizedPrediction,
            NormalizedSample,
        )

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-125",),
            ),
        ]
        # Wrong CWE → FN
        preds_wrong = [
            NormalizedPrediction(
                sample_id="s1",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=("CWE-787",),
            ),
        ]
        m = compute_code_vuln_metrics(samples, preds_wrong)
        assert m["tp"] == 0
        assert m["fn"] == 1

    def test_no_cwe_predicted_is_fn(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import (
            NormalizedPrediction,
            NormalizedSample,
        )

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-125",),
            ),
        ]
        preds_no_cwe = [
            NormalizedPrediction(
                sample_id="s1",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=(),
            ),
        ]
        m = compute_code_vuln_metrics(samples, preds_no_cwe)
        assert m["tp"] == 0
        assert m["fn"] == 1
        assert m["wrong_num"] == 1

    def test_multiple_cwes_are_fn(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import (
            NormalizedPrediction,
            NormalizedSample,
        )

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-79",),
            ),
        ]
        preds_multiple = [
            NormalizedPrediction(
                sample_id="s1",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=("CWE-79", "CWE-89"),
            ),
        ]
        m = compute_code_vuln_metrics(samples, preds_multiple)
        assert m["tp"] == 0
        assert m["fn"] == 1
        assert m["wrong_num"] == 1

    def test_missing_prediction_for_vulnerable_sample_is_fn(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import NormalizedSample

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": True},
                category_labels=("CWE-79",),
            ),
        ]

        m = compute_code_vuln_metrics(samples, [])
        assert m["count"] == 1
        assert m["tp"] == 0
        assert m["fn"] == 1

    def test_missing_prediction_for_benign_sample_is_fp(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import NormalizedSample

        samples = [
            NormalizedSample(
                id="s1",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": False},
            ),
        ]

        m = compute_code_vuln_metrics(samples, [])
        assert m["count"] == 1
        assert m["tn"] == 0
        assert m["fp"] == 1

    def test_overall_f1_matches_vulnllmr(self):
        from guard_eval_harness.metrics.code_vuln import (
            compute_code_vuln_metrics,
        )
        from guard_eval_harness.schemas import (
            NormalizedPrediction,
            NormalizedSample,
        )

        samples = [
            NormalizedSample(
                id=f"s{i}",
                dataset="test",
                split="test",
                messages=[{"role": "user", "content": "code"}],
                label={"unsafe": label},
                category_labels=(("CWE-125",) if label else ()),
            )
            for i, label in enumerate([True, True, False, False])
        ]
        preds = [
            NormalizedPrediction(
                sample_id="s0",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=("CWE-125",),
            ),
            NormalizedPrediction(
                sample_id="s1",
                unsafe_score=0.0,
                unsafe_label=False,
            ),
            NormalizedPrediction(
                sample_id="s2",
                unsafe_score=0.0,
                unsafe_label=False,
            ),
            NormalizedPrediction(
                sample_id="s3",
                unsafe_score=1.0,
                unsafe_label=True,
                predicted_categories=("CWE-787",),
            ),
        ]
        m = compute_code_vuln_metrics(samples, preds)
        assert m["tp"] == 1  # s0
        assert m["fn"] == 1  # s1 (missed)
        assert m["tn"] == 1  # s2
        assert m["fp"] == 1  # s3
        # overall F1 = avg(pos_f1, neg_f1)
        assert m["overall F1"] is not None


# --- extract_judge_categories ---


class TestExtractJudgeCategories:
    """Tests for extract_judge_categories in templates."""

    def test_extracts_cwe(self):
        from guard_eval_harness.models.templates import (
            extract_judge_categories,
        )

        cats = extract_judge_categories("#judge: yes\n#type: CWE-125")
        assert cats == ("CWE-125",)

    def test_na_returns_empty(self):
        from guard_eval_harness.models.templates import (
            extract_judge_categories,
        )

        cats = extract_judge_categories("#judge: no\n#type: N/A")
        assert cats == ()

    def test_no_type_returns_empty(self):
        from guard_eval_harness.models.templates import (
            extract_judge_categories,
        )

        cats = extract_judge_categories("The code looks safe.")
        assert cats == ()

    def test_multi_cwe_extracted(self):
        from guard_eval_harness.models.templates import (
            extract_judge_categories,
        )

        cats = extract_judge_categories("#judge: yes\n#type: CWE-125, CWE-416")
        assert cats == ("CWE-125", "CWE-416")

    def test_multi_cwe_fails_single_check(self):
        from guard_eval_harness.metrics.code_vuln import (
            _has_single_cwe,
        )

        assert _has_single_cwe(("CWE-125", "CWE-416")) is False
        assert _has_single_cwe(("CWE-125",)) is True
        assert _has_single_cwe(()) is False
