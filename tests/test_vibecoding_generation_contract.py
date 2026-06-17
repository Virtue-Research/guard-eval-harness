"""Tests for the oracle-driven generation contract + `geh vibe acquire`.

Covers the generation-side seam that lets a dataset-agnostic agent driver emit
exactly the artifact kind an oracle scores: the default task-typed behavior,
custom prompt/parse threading through the shared engine, the kind-aware empty
fallback (incl. the full_file sentinel + repo_dir fallback), the SecRepoBench
ground-truth snapshot/restore/pin fixups, the BaxBench descriptor + full_file
response parsing, and the acquire command's inline/error paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from guard_eval_harness.vibecoding.agents._engine import (
    ChatResponse,
    artifact_for_kind,
    generate_with,
)
from guard_eval_harness.vibecoding.artifacts import AgentArtifact
from guard_eval_harness.vibecoding.interfaces import GenerationSpec
from guard_eval_harness.vibecoding.oracles.secrepobench import (
    SecRepoBenchOracle,
)
from guard_eval_harness.vibecoding.sources.secrepobench import (
    SecRepoBenchTaskSource,
)
from guard_eval_harness.vibecoding.registry import (
    ensure_vibe_registrations,
    oracle_registry,
)
from guard_eval_harness.vibecoding.schema import TaskEnvironmentRef, VibeTask

_DIFF = "```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n```"


def _task(task_id: str = "t/1", task_type: str = "repo_patch") -> VibeTask:
    return VibeTask(
        id=task_id,
        source_dataset="t",
        task_type=task_type,
        instructions="fix the vulnerability",
        environment=TaskEnvironmentRef(oracle="t"),
    )


def _complete(text: str):
    """A CompleteFn stub returning fixed text and recording the prompt."""
    seen: dict = {}

    def complete(messages, system, model) -> ChatResponse:
        seen["system"] = system
        seen["user"] = messages[-1]["content"]
        seen["model"] = model
        return ChatResponse(text=text, prompt_tokens=1, completion_tokens=2)

    return complete, seen


# --- default generation_spec (legacy parity) ---------------------------


def test_default_spec_kinds_match_task_type():
    ensure_vibe_registrations()
    svb = oracle_registry.get("securevibebench")()
    secrepo = oracle_registry.get("secrepobench")()
    assert svb.generation_spec(_task(task_type="repo_patch")).artifact_kind == (
        "patch"
    )
    # SecRepoBench uses the inherited default for its repo_completion tasks.
    spec = secrepo.generation_spec(_task(task_type="repo_completion"))
    assert spec.artifact_kind == "completion"
    assert spec.prompt is None
    assert spec.parse is None


def test_default_spec_fails_loud_for_unscoreable_kind():
    """An oracle scoring only a richer kind (no patch/completion) and no
    override gets a loud, actionable error from the default -- not a patch every
    candidate would score as unsupported."""
    ensure_vibe_registrations()
    for name, task_type in (
        ("ase", "repo_dir"),
        ("seccodebench", "project_scaffold"),
    ):
        oracle = oracle_registry.get(name)()
        with pytest.raises(ValueError, match=f"not supported for the {name!r}"):
            oracle.generation_spec(_task(f"{name}/x", task_type))


def test_generate_with_no_spec_is_legacy_patch():
    complete, _ = _complete(_DIFF)
    result = generate_with(
        _task(), workdir=None, model="m", default_model="m",
        complete=complete,
    )
    assert result.artifact.kind == "patch"
    assert "good" in result.artifact.patch


# --- custom spec threading ---------------------------------------------


def test_generate_with_custom_prompt_and_parse():
    """A spec's prompt + parse fully override the engine defaults."""
    complete, seen = _complete("FILES: app.py")

    def prompt(task, snapshot):
        return ("SYS", f"build {task.id}")

    def parse(task, model, text):
        assert text == "FILES: app.py"
        return AgentArtifact(
            task_id=task.id, model=model, kind="full_file",
            files={"app.py": "print('hi')"},
        )

    spec = GenerationSpec(artifact_kind="full_file", prompt=prompt, parse=parse)
    result = generate_with(
        _task(), workdir=None, model="m", default_model="m",
        complete=complete, spec=spec,
    )
    assert seen["system"] == "SYS"
    assert seen["user"] == "build t/1"
    assert result.artifact.kind == "full_file"
    assert result.artifact.files == {"app.py": "print('hi')"}


def test_generate_with_spec_parse_none_is_empty_artifact():
    """parse returning None degrades to a kind-pinned empty artifact."""
    complete, _ = _complete("garbled, no usable code")

    spec = GenerationSpec(
        artifact_kind="completion", parse=lambda task, model, text: None,
    )
    result = generate_with(
        _task(task_type="repo_completion"), workdir=None, model="m",
        default_model="m", complete=complete, spec=spec,
    )
    assert result.artifact.kind == "completion"
    assert result.artifact.metadata.get("empty") is True


def test_generate_with_custom_parse_exception_degrades_to_empty():
    """A custom parser that raises on bad output degrades to an empty artifact
    instead of aborting the batch (the driver contract)."""
    complete, _ = _complete("malformed model output")

    def boom(task, model, text):
        raise ValueError("bad json")

    spec = GenerationSpec(artifact_kind="full_file", parse=boom)
    result = generate_with(
        _task(), workdir=None, model="m", default_model="m",
        complete=complete, spec=spec,
    )
    assert result.artifact.metadata.get("empty") is True
    assert "parse failed" in (result.artifact.metadata.get("error") or "")
    # The empty artifact keeps the file-map kind so the oracle scores it as a
    # model failure, not an excluded "unsupported" row (see denominator test).
    assert result.artifact.kind == "full_file"


def test_empty_filemap_generation_preserves_kind_for_denominator():
    """A full_file spec whose parser returns None yields a full_file empty
    artifact (an in-denominator model failure once staged), not a patch the
    project-scaffold oracle would exclude as unsupported."""
    complete, _ = _complete("I will not write this application")
    spec = GenerationSpec(
        artifact_kind="full_file", parse=lambda t, m, x: None,
    )
    result = generate_with(
        _task(task_type="project_scaffold"), workdir=None, model="m",
        default_model="m", complete=complete, spec=spec,
    )
    assert result.artifact.kind == "full_file"
    assert result.artifact.files  # non-empty -> passes the oracle has-files gate
    assert result.artifact.metadata.get("empty") is True


def test_empty_result_repo_dir_does_not_build_invalid_files_artifact():
    """repo_dir scores via a worktree path, not a file map -- the file-map
    sentinel would be a structurally invalid artifact, so empty_result falls
    back to the legacy text sentinel instead of raising."""
    from guard_eval_harness.vibecoding.agents._engine import empty_result

    res = empty_result(_task(task_type="repo_dir"), "m", kind="repo_dir")
    assert res.artifact.kind in ("patch", "completion")
    assert res.artifact.metadata.get("empty") is True


def test_generate_with_prompt_exception_degrades_to_empty():
    """A spec.prompt that raises (e.g. BaxBench's venv prompt build failing)
    degrades that task to an empty artifact instead of aborting the batch."""
    def boom_prompt(task, snapshot):
        raise RuntimeError("venv missing")

    complete, _ = _complete("unused")
    spec = GenerationSpec(artifact_kind="full_file", prompt=boom_prompt)
    res = generate_with(
        _task(task_type="project_scaffold"), workdir=None, model="m",
        default_model="m", complete=complete, spec=spec,
    )
    assert res.artifact.kind == "full_file"
    assert res.artifact.metadata.get("empty") is True
    assert "prompt build failed" in (res.artifact.metadata.get("error") or "")


def test_generate_with_spec_kind_drives_engine_wrap():
    """With no custom parse, the spec's kind selects the engine wrapper."""
    complete, _ = _complete("```\nsome completion body\n```")
    spec = GenerationSpec(artifact_kind="completion")
    result = generate_with(
        _task(task_type="repo_patch"), workdir=None, model="m",
        default_model="m", complete=complete, spec=spec,
    )
    # task_type is repo_patch but the oracle declared completion -> completion.
    assert result.artifact.kind == "completion"
    assert "some completion body" in result.artifact.completion


# --- artifact_for_kind guard -------------------------------------------


def test_artifact_for_kind_wraps_text_kinds():
    patch = artifact_for_kind(_task(), "m", "diff-body", "patch")
    assert patch.kind == "patch" and patch.patch == "diff-body"
    comp = artifact_for_kind(_task(), "m", "comp-body", "completion")
    assert comp.kind == "completion" and comp.completion == "comp-body"


def test_artifact_for_kind_rejects_filemap_kind():
    with pytest.raises(ValueError, match="GenerationSpec.parse"):
        artifact_for_kind(_task(), "m", "body", "full_file")


# --- SecRepoBench acquisition fixup ------------------------------------


class _Resolved:
    def __init__(self, upstream_dir: str) -> None:
        self.upstream_dir = upstream_dir


def test_secrepobench_prepare_snapshots_pristine_ids(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "ids.txt").write_text("a\nb\nc\n")
    SecRepoBenchOracle().prepare_acquisition(_Resolved(str(tmp_path)))
    # First acquire snapshots the full list; ids.txt is untouched.
    assert (assets / "ids.txt.full").read_text() == "a\nb\nc\n"
    assert (assets / "ids.txt").read_text() == "a\nb\nc\n"


def test_secrepobench_prepare_restores_truncated_ids(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "ids.txt.full").write_text("a\nb\nc\nd\n")
    (assets / "ids.txt").write_text("a\n")  # a prior run truncated it
    SecRepoBenchOracle().prepare_acquisition(_Resolved(str(tmp_path)))
    assert (assets / "ids.txt").read_text() == "a\nb\nc\nd\n"


def test_secrepobench_prepare_is_noop_without_ids(tmp_path: Path):
    # No assets dir at all: must not raise.
    SecRepoBenchOracle().prepare_acquisition(_Resolved(str(tmp_path)))
    assert not (tmp_path / "assets").exists()


def test_secrepobench_materialize_gz_files(tmp_path: Path):
    """Upstream ships report.json + sample_metadata.json gzipped; acquire
    gunzips them (idempotent, keeps the .gz, never clobbers an existing plain
    file)."""
    import gzip

    from guard_eval_harness.vibecoding.oracles.secrepobench import (
        materialize_gz_files,
    )

    (tmp_path / "sample_metadata.json.gz").write_bytes(
        gzip.compress(b'{"1": {}}')
    )
    (tmp_path / "report.json.gz").write_bytes(gzip.compress(b"{}"))
    materialize_gz_files(tmp_path)
    assert (tmp_path / "sample_metadata.json").read_text() == '{"1": {}}'
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "sample_metadata.json.gz").is_file()  # .gz kept
    # idempotent: an existing plain file is not clobbered
    (tmp_path / "sample_metadata.json").write_text("CUSTOM")
    materialize_gz_files(tmp_path)
    assert (tmp_path / "sample_metadata.json").read_text() == "CUSTOM"


def test_secrepobench_source_load_from_gzipped_checkout(tmp_path: Path):
    """A fresh checkout ships sample_metadata.json only as a tracked .gz; the
    source still loads the full set (the snapshot seeds from the git .gz)."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    import gzip

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "ids.txt").write_text("id\n1\n2\n")
    meta = {
        str(i): {"project_name": "p", "changed_file": "f.c",
                 "fixing_commit": "c", "crash_type": "Null-dereference"}
        for i in (1, 2)
    }
    (tmp_path / "sample_metadata.json.gz").write_bytes(
        gzip.compress(json.dumps(meta).encode())
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "i"], cwd=tmp_path, check=True, env=env,
    )
    # No plain sample_metadata.json: load seeds the snapshot from the .gz blob.
    tasks = SecRepoBenchTaskSource(root=tmp_path).load()
    assert {t.id for t in tasks} == {"secrepobench/1", "secrepobench/2"}


# --- SecRepoBench load survives scorer truncation (the denominator P1) --


def _write_secrepo_checkout(root: Path, ids: list[int]) -> None:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "ids.txt").write_text(
        "id\n" + "\n".join(str(i) for i in ids) + "\n"
    )
    meta = {
        str(i): {
            "project_name": "p",
            "changed_file": "f.c",
            "fixing_commit": "abc123",
            "crash_type": "Null-dereference",
        }
        for i in ids
    }
    (root / "sample_metadata.json").write_text(json.dumps(meta))


def test_source_load_prefers_full_snapshot_over_truncated_ids(tmp_path: Path):
    """A pre-existing ``ids.txt.full`` is read even if ``ids.txt`` is truncated."""
    _write_secrepo_checkout(tmp_path, [101, 102, 103])
    # Simulate a checkout a prior scorer left truncated, with a full snapshot.
    (tmp_path / "assets" / "ids.txt.full").write_text("id\n101\n102\n103\n")
    (tmp_path / "assets" / "ids.txt").write_text("id\n101\n")
    tasks = SecRepoBenchTaskSource(root=tmp_path).load()
    assert {t.id for t in tasks} == {
        "secrepobench/101", "secrepobench/102", "secrepobench/103",
    }


def test_source_load_snapshots_then_survives_truncation(tmp_path: Path):
    """First load snapshots the pristine set; a later truncation cannot shrink
    the denominator."""
    _write_secrepo_checkout(tmp_path, [101, 102, 103])
    src = SecRepoBenchTaskSource(root=tmp_path)
    assert len(src.load()) == 3  # captures assets/ids.txt.full (+ metadata)
    # The upstream scorer rewrites the shared files to its scored subset.
    (tmp_path / "assets" / "ids.txt").write_text("id\n101\n")
    (tmp_path / "sample_metadata.json").write_text(
        json.dumps({"101": {"project_name": "p", "changed_file": "f.c",
                            "fixing_commit": "abc123",
                            "crash_type": "Null-dereference"}})
    )
    assert len(src.load()) == 3  # still full, via the snapshot


def test_secrepobench_pin_ground_truth_subset(tmp_path: Path):
    """Before eval, the working ground-truth files are rebuilt to exactly the
    staged subset from the full snapshot -- not whatever a prior run left."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "ids.txt.full").write_text("id\n1\n2\n3\n")
    (tmp_path / "assets" / "ids.txt").write_text("id\n9\n")  # stale leftover
    full_meta = {
        str(i): {"project_name": "p", "changed_file": "f.c",
                 "fixing_commit": "c", "crash_type": "Null-dereference"}
        for i in (1, 2, 3)
    }
    (tmp_path / "sample_metadata.json.full").write_text(json.dumps(full_meta))
    (tmp_path / "sample_metadata.json").write_text(json.dumps({"9": {}}))
    SecRepoBenchOracle._pin_ground_truth_subset(tmp_path, ["2", "3"])
    ids = [
        ln for ln in (tmp_path / "assets" / "ids.txt").read_text().splitlines()
        if ln.strip() and ln.strip() != "id"
    ]
    assert ids == ["2", "3"]
    assert set(json.loads(
        (tmp_path / "sample_metadata.json").read_text()
    )) == {"2", "3"}


def test_snapshot_seeds_from_git_when_working_pretruncated(tmp_path: Path):
    """A pre-existing checkout whose tracked ground-truth was already truncated
    (and has no snapshot yet) seeds the snapshot from the pinned git blob, not
    the truncated working file."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    from guard_eval_harness.vibecoding.oracles.secrepobench import (
        ensure_ground_truth_snapshots,
    )

    _write_secrepo_checkout(tmp_path, [101, 102, 103])
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True, env=env,
    )
    # A prior scorer truncated the tracked working file; no .full snapshot yet.
    (tmp_path / "assets" / "ids.txt").write_text("id\n101\n")
    ensure_ground_truth_snapshots(tmp_path)
    snap_ids = [
        ln for ln in (tmp_path / "assets" / "ids.txt.full")
        .read_text().splitlines() if ln.strip() and ln.strip() != "id"
    ]
    assert len(snap_ids) == 3  # from git HEAD, not the truncated working file
    assert len(SecRepoBenchTaskSource(root=tmp_path).load()) == 3


# --- geh vibe acquire ---------------------------------------------------


def test_acquire_cli_inline_oracle_is_noop(capsys):
    """`geh vibe acquire` on an inline oracle (mock) reports nothing to do and
    never touches an EnvProvider."""
    from guard_eval_harness.vibecoding import cli as vibe_cli

    rc = vibe_cli.dispatch(argparse.Namespace(
        vibe_action="acquire", dataset="mock", cache_dir=None, force=False,
    ))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dataset"] == "mock"
    assert out["status"] == "inline"


def test_acquire_cli_unknown_dataset_errors(capsys):
    from guard_eval_harness.vibecoding import cli as vibe_cli

    rc = vibe_cli.dispatch(argparse.Namespace(
        vibe_action="acquire", dataset="nope", cache_dir=None, force=False,
    ))
    assert rc == 1
    assert "Error" in capsys.readouterr().out


def test_acquire_cli_reports_setup_failure_cleanly(monkeypatch, capsys):
    """A clone/checkout/venv-install failure (ensure_ready -> RuntimeError) is
    reported as a structured error + exit 1, not a traceback."""
    from guard_eval_harness.vibecoding import cli as vibe_cli
    from guard_eval_harness.vibecoding import envs as envs_mod

    def boom(self, *, force=False):
        raise RuntimeError("git clone failed")

    monkeypatch.setattr(envs_mod.EnvProvider, "ensure_ready", boom)
    rc = vibe_cli.dispatch(argparse.Namespace(
        vibe_action="acquire", dataset="securevibebench",
        cache_dir=None, force=False,
    ))
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "error"
    assert out["stage"] == "ensure_ready"
    assert "git clone failed" in out["detail"]


# --- BaxBench acquire materializes scenario.json descriptors -----------


class _BaxResolved:
    def __init__(self, upstream_dir, venv_python) -> None:
        self.upstream_dir = upstream_dir
        self.venv_python = venv_python


def test_baxbench_prepare_acquisition_materializes_descriptors(tmp_path: Path):
    """The pinned upstream ships flat scenarios/*.py; acquire writes the
    <id>/scenario.json the task source globs (run in the dataset venv)."""
    from guard_eval_harness.vibecoding.oracles.baxbench import BaxBenchOracle

    src = tmp_path / "src"
    pkg = src / "scenarios"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "class _CWE:\n"
        "    def __init__(self, n): self.value = {'num': n}\n"
        "class _S:\n"
        "    id = 'DemoApp'\n"
        "    short_app_description = 'demo'\n"
        "    api_spec = 'openapi: 3.0'\n"
        "    text_spec = ''\n"
        "    potential_cwes = [_CWE(79), _CWE(89)]\n"
        "all_scenarios = [_S()]\n"
    )
    # The script also imports the upstream `env` module for the env id list.
    (src / "env.py").write_text(
        "class _E:\n"
        "    def __init__(self, i): self.id = i\n"
        "all_envs = [_E('Python-Flask'), _E('Go-Gin')]\n"
    )
    # The materialization runs in the dataset's venv; the test python stands in.
    BaxBenchOracle().prepare_acquisition(
        _BaxResolved(str(tmp_path), sys.executable)
    )
    desc = pkg / "DemoApp" / "scenario.json"
    assert desc.is_file()
    data = json.loads(desc.read_text())
    assert data["id"] == "DemoApp"
    assert data["cwes"] == ["CWE-79", "CWE-89"]
    assert "openapi" in data["instructions"]
    # All upstream env ids are written so a full run enumerates the whole grid.
    assert data["envs"] == ["Python-Flask", "Go-Gin"]


def test_baxbench_prepare_acquisition_noop_without_paths():
    from guard_eval_harness.vibecoding.oracles.baxbench import BaxBenchOracle

    # Missing upstream/venv must not raise (e.g. inline/misconfigured env).
    BaxBenchOracle().prepare_acquisition(_BaxResolved(None, None))


def test_baxbench_extract_files_formats():
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _extract_baxbench_files,
    )

    # single-file <CODE> -> the env entrypoint
    assert _extract_baxbench_files("<CODE>\nx=1\n</CODE>", "app.py") == {
        "app.py": "x=1",
    }
    # multi-file <FILEPATH>/<CODE> pairs (entrypoint irrelevant)
    assert _extract_baxbench_files(
        "<FILEPATH>a.py</FILEPATH><CODE>1</CODE>"
        "<FILEPATH>b/c.py</FILEPATH><CODE>2</CODE>",
        None,
    ) == {"a.py": "1", "b/c.py": "2"}
    # markdown-fence fallback -> the env entrypoint
    assert _extract_baxbench_files("```go\npackage main\n```", "main.go") == {
        "main.go": "package main",
    }
    # nothing usable -> None (engine emits an empty artifact)
    assert _extract_baxbench_files("no code here", "app.py") is None


def test_baxbench_extract_multifile_with_markdown_fences():
    """Upstream pairs <FILEPATH> markers with markdown fences by position (the
    multi-file delivery format the strict <FILEPATH><CODE> regex missed)."""
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _extract_baxbench_files,
    )

    text = (
        "<FILEPATH>app/models.py</FILEPATH>\n```python\nx = 1\n```\n"
        "<FILEPATH>app/views.py</FILEPATH>\n```python\ny = 2\n```\n"
    )
    assert _extract_baxbench_files(text, None) == {
        "app/models.py": "x = 1", "app/views.py": "y = 2",
    }


def test_baxbench_extract_single_file_prefers_markdown():
    """Single-file: a markdown fence wins over a <CODE> block (upstream order)."""
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _extract_baxbench_files,
    )

    text = "```python\nFROM_FENCE\n```\n<CODE>FROM_CODE</CODE>"
    assert _extract_baxbench_files(text, "app.py") == {"app.py": "FROM_FENCE"}


def test_baxbench_extract_bare_fences():
    """A bare ``` fence (no language tag) parses for single- and multi-file,
    and a ```bash fence is never treated as app code."""
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _extract_baxbench_files,
    )

    assert _extract_baxbench_files("```\npackage main\n```", "main.go") == {
        "main.go": "package main",
    }
    multi = (
        "<FILEPATH>a.py</FILEPATH>\n```\nx = 1\n```\n"
        "<FILEPATH>b.py</FILEPATH>\n```\ny = 2\n```\n"
    )
    assert _extract_baxbench_files(multi, None) == {"a.py": "x = 1", "b.py": "y = 2"}
    assert _extract_baxbench_files("```bash\nrm -rf /\n```", "app.py") is None


def test_baxbench_extract_heading_paths_and_nested_md():
    """Mirror upstream: ``### path`` headings mark multi-file paths, and a
    <CODE> block wrapping a nested markdown fence is unwrapped."""
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _extract_baxbench_files,
    )

    headings = (
        "### app/models.py\n```python\nx = 1\n```\n"
        "### app/views.py\n```python\ny = 2\n```\n"
    )
    assert _extract_baxbench_files(headings, None) == {
        "app/models.py": "x = 1", "app/views.py": "y = 2",
    }
    nested = "<FILEPATH>a.py</FILEPATH>\n<CODE>\n```python\nz = 3\n```\n</CODE>"
    assert _extract_baxbench_files(nested, None) == {"a.py": "z = 3"}


def test_baxbench_batch_timeout_scales_and_overrides():
    """The whole-batch cap scales by ceil(tasks/workers), floors at the default,
    and honors run_config.extra['timeout_s']."""
    from guard_eval_harness.vibecoding.oracles.baxbench import (
        _DEFAULT_TIMEOUT_S,
        _PER_TASK_TIMEOUT_S,
        BaxBenchOracle,
    )
    from guard_eval_harness.vibecoding.interfaces import (
        OracleRunConfig,
        StagedOracleInput,
    )

    def staged(n):
        return StagedOracleInput(
            adapter_name="baxbench", inputs_dir="x",
            task_ids=[f"baxbench/s{i}" for i in range(n)],
        )

    rc = OracleRunConfig(run_id="r", run_dir="x")
    # small batch -> floored at the default
    assert BaxBenchOracle._batch_timeout(staged(1), 10, rc) == _DEFAULT_TIMEOUT_S
    # large batch -> per-task budget * waves
    assert BaxBenchOracle._batch_timeout(staged(392), 10, rc) == (
        _PER_TASK_TIMEOUT_S * 40
    )
    # override wins
    rc_over = OracleRunConfig(
        run_id="r", run_dir="x", extra={"timeout_s": 42}
    )
    assert BaxBenchOracle._batch_timeout(staged(392), 10, rc_over) == 42.0


def test_baxbench_validate_rejects_escaping_file_keys():
    """A BYO full_file key that escapes the staging tree is a per-candidate
    unsupported row (UnsupportedArtifactError), not a stage-time ValueError that
    aborts the whole batch."""
    from guard_eval_harness.vibecoding.artifacts import AgentArtifact
    from guard_eval_harness.vibecoding.interfaces import (
        UnsupportedArtifactError,
    )

    ensure_vibe_registrations()
    oracle = oracle_registry.get("baxbench")()
    tid = "baxbench/Calculator__Python-Flask"
    for bad in ("../evil.py", "/etc/passwd", "a/../../b"):
        art = AgentArtifact(
            task_id=tid, model="m", kind="full_file", files={bad: "x"},
        )
        with pytest.raises(UnsupportedArtifactError, match="traversal|absolute"):
            oracle.validate(art)
    # a normal key validates
    oracle.validate(AgentArtifact(
        task_id=tid, model="m", kind="full_file", files={"app.py": "x"},
    ))


def test_baxbench_generation_spec_emits_full_file():
    """generation_spec drives the upstream prompt and parses into full_file --
    so a baxbench live run stages instead of scoring unsupported."""
    from guard_eval_harness.vibecoding.oracles.baxbench import BaxBenchOracle

    oracle = BaxBenchOracle()
    # Pre-seed the per-(scenario, env) prompt cache so no dataset venv is needed.
    # Cache is keyed (scenario, env, cache_dir); generation_spec defaults
    # cache_dir to None, so pre-seed that key to avoid a real venv build.
    oracle.__dict__["_prompt_cache"] = {
        ("Calculator", "Python-Flask", None): {
            "prompt": "implement the calculator app",
            "code_filename": "app.py",
        }
    }
    task = _task("baxbench/Calculator__Python-Flask", "project_scaffold")
    spec = oracle.generation_spec(task)
    assert spec.artifact_kind == "full_file"
    system, user = spec.prompt(task, "")
    assert user == "implement the calculator app" and system
    art = spec.parse(task, "m", "<CODE>\nfrom flask import Flask\n</CODE>")
    assert art is not None
    assert art.kind == "full_file"
    assert art.files == {"app.py": "from flask import Flask"}
    # Unusable output -> None (degrades to an empty artifact, not unsupported).
    assert spec.parse(task, "m", "sorry, I cannot") is None
