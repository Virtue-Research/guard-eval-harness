"""Tests for the live-agent sandbox: git seal, network policy, anti-cheat."""

from __future__ import annotations

from guard_eval_harness.vibecoding.sandbox.anti_cheat import (
    DEFAULT_POLICY_ID,
    AntiCheat,
    _iter_workspace_files,
)
from guard_eval_harness.vibecoding.sandbox.git_seal import seal_git_history
from guard_eval_harness.vibecoding.sandbox.network import build_policy


def test_git_seal_removes_history_and_solution(tmp_path):
    wt = tmp_path / "wt"
    (wt / ".git").mkdir(parents=True)
    (wt / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (wt / "golden_patch.diff").write_text("the upstream fix")
    (wt / "main.py").write_text("print('ok')")

    res = seal_git_history(wt)

    assert res.git_sealed is True
    assert res.anything_sealed is True
    # golden solution file is gone; ordinary source is preserved.
    assert not (wt / "golden_patch.diff").exists()
    assert (wt / "main.py").exists()


def test_network_allowlist_default_deny():
    policy = build_policy("allowlist", allow=["api.anthropic.com"])
    assert policy.is_host_allowed("api.anthropic.com") is True
    assert policy.is_url_allowed("https://api.anthropic.com/v1/messages")
    assert policy.is_host_allowed("evil.example") is False
    # loopback always reachable (local LLM endpoints).
    assert policy.is_host_allowed("localhost") is True
    env = policy.proxy_env()
    assert "HTTPS_PROXY" in env  # allowlist routes through a blackhole proxy
    assert policy.policy_hash()


def test_network_denylist_permits_others():
    policy = build_policy("denylist", deny=["github.com"])
    assert policy.is_host_allowed("github.com") is False
    assert policy.is_host_allowed("example.org") is True


def test_anti_cheat_flags_marker_and_workspace(tmp_path):
    ac = AntiCheat(enforced=True)

    clean = ac.scan_text("--- a/x\n+++ b/x\n@@ -1 +1 @@\n+safe code\n")
    assert clean.cheating_flagged is False
    assert clean.anti_cheat_enforced is True

    dirty = ac.scan_text("// reproduced from golden_patch reference\n")
    assert dirty.cheating_flagged is True
    assert dirty.findings

    # before/after workspace diff: a leaked solution file appears post-run.
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "main.py").write_text("print('ok')")
    before = ac.scan_workspace_before(wt)
    (wt / "leaked.py").write_text("// copied from golden_patch\n")
    after = ac.scan_workspace_after(wt, before)
    assert after.cheating_flagged is True
    assert ac.policy_hash()


def test_iter_workspace_files_yields_regular_file_named_dot_git(tmp_path):
    """Only directory components are skip-matched, not the filename.

    A regular FILE named ``.git`` (e.g. a submodule pointer) must be
    snapshotted, while files inside a ``.git`` DIRECTORY stay skipped.
    """
    wt = tmp_path / "wt"
    (wt / ".git").mkdir(parents=True)
    (wt / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (wt / "sub").mkdir()
    (wt / "sub" / ".git").write_text("gitdir: ../.git/modules/sub")
    (wt / "main.py").write_text("print('ok')")

    rels = [p.relative_to(wt).as_posix() for p in _iter_workspace_files(wt)]

    assert "sub/.git" in rels
    assert "main.py" in rels
    assert ".git/HEAD" not in rels


def test_anti_cheat_detects_new_file_named_dot_git(tmp_path):
    """An agent creating a file literally named ``.git`` shows in new_files."""
    ac = AntiCheat(enforced=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "main.py").write_text("print('ok')")
    before = ac.scan_workspace_before(wt)
    (wt / ".git").write_text("gitdir: /elsewhere")
    after = ac.scan_workspace_after(wt, before)
    assert ".git" in after.new_files


def test_policy_id_single_sourced_with_runner():
    # The runner's cache-key/provenance token must be the SAME object as the
    # sandbox policy id: a drift here would split oracle cache keys from the
    # declared anti-cheat policy identity (a bug class this pins).
    from guard_eval_harness.vibecoding.runner import (
        _DEFAULT_ANTI_CHEAT_POLICY,
    )

    assert _DEFAULT_ANTI_CHEAT_POLICY is DEFAULT_POLICY_ID
    # v0: detectors exist but are not enforced; the placeholder token is
    # "none" and must stay stable or every cached verdict is invalidated.
    assert DEFAULT_POLICY_ID == "none"
