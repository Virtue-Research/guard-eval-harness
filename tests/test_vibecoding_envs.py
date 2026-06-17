"""Tests for the vibecoding EnvProvider / subprocess / resources slice.

These run with no real network, Docker, or git: the upstream/venv probes
exercise an empty temp cache dir, and the docker probe is driven through
injected ``which``/``runner`` stubs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from guard_eval_harness.vibecoding.envs import EnvProvider, ResolvedEnv
from guard_eval_harness.vibecoding.resources import compute_resource_budget
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    OracleParallelism,
    ResourceEstimate,
)
from guard_eval_harness.vibecoding.subprocess import (
    CommandResult,
    run_command,
    which,
)

PY = sys.executable


def _python_bin() -> str:
    return which("python") or which("python3") or PY


class RunCommandTest(unittest.TestCase):
    """run_command: capture, exit codes, timeout, file capture."""

    def test_echo_capture(self) -> None:
        result = run_command(
            [PY, "-c", "print('hello-vibe')"],
            timeout_s=30,
        )
        self.assertIsInstance(result, CommandResult)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)
        self.assertIn("hello-vibe", result.stdout)
        self.assertGreaterEqual(result.duration_s, 0.0)

    def test_nonzero_does_not_raise(self) -> None:
        result = run_command([PY, "-c", "import sys; sys.exit(7)"])
        self.assertEqual(result.returncode, 7)
        self.assertFalse(result.timed_out)

    def test_stderr_capture(self) -> None:
        result = run_command(
            [PY, "-c", "import sys; sys.stderr.write('boom')"],
        )
        self.assertIn("boom", result.stderr)

    def test_timeout_sets_timed_out_and_none_rc(self) -> None:
        result = run_command(
            [PY, "-c", "import time; time.sleep(30)"],
            timeout_s=0.5,
        )
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.returncode)

    def test_missing_binary_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            run_command(["definitely-not-a-real-binary-xyz"])

    def test_capture_to_files_writes_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_command(
                [PY, "-c", "print('to-file')"],
                capture_to_files=tmp,
            )
            self.assertIsNotNone(result.stdout_path)
            self.assertTrue(Path(result.stdout_path).exists())
            self.assertIn(
                "to-file", Path(result.stdout_path).read_text()
            )
            self.assertTrue(Path(result.stderr_path).exists())


class WhichTest(unittest.TestCase):
    """which resolves real binaries and returns None for missing ones."""

    def test_which_finds_python(self) -> None:
        self.assertIsNotNone(_python_bin())

    def test_which_missing_returns_none(self) -> None:
        self.assertIsNone(which("definitely-not-a-real-binary-xyz"))


class RedactedEnvTest(unittest.TestCase):
    """redacted_env / env_fingerprint mask secrets."""

    def test_api_key_masked_in_redacted_env(self) -> None:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENAI_API_KEY": "sk-supersecret-value",
            "PLAIN_VAR": "ok",
        }
        result = run_command(
            [_python_bin(), "-c", "print('x')"], env=env
        )
        self.assertEqual(
            result.redacted_env["OPENAI_API_KEY"], "***REDACTED***"
        )
        self.assertEqual(result.redacted_env["PLAIN_VAR"], "ok")
        self.assertNotIn(
            "sk-supersecret-value", str(result.redacted_env)
        )
        self.assertIsNotNone(result.env_fingerprint)


class ChildEnvAllowlistTest(unittest.TestCase):
    """Oracle subprocesses receive only allowlisted/declared host vars."""

    def _provider(self, **spec_kw) -> EnvProvider:
        spec = EnvSpec(name="allowtest", **spec_kw)
        return EnvProvider(spec, cache_dir="/tmp/fake-cache")

    def setUp(self) -> None:
        self._saved = {
            key: os.environ.get(key)
            for key in (
                "FAKE_HOST_SECRET",
                "AWS_SECRET_ACCESS_KEY",
                "LC_ALL",
                "DOCKER_CONFIG",
            )
        }
        os.environ["FAKE_HOST_SECRET"] = "do-not-leak"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "do-not-leak-either"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["DOCKER_CONFIG"] = "/tmp/docker-config"

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_undeclared_host_secrets_never_reach_the_child(self) -> None:
        # Redaction protects what is PERSISTED; the live child must simply
        # never receive undeclared host secrets in the first place.
        provider = self._provider()
        env = provider._venv_env(provider.resolve())
        self.assertNotIn("FAKE_HOST_SECRET", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)

    def test_allowlisted_basics_and_prefixes_pass(self) -> None:
        provider = self._provider()
        resolved = provider.resolve()
        env = provider._venv_env(resolved)
        # Process basics + Docker client config + LC_ prefix.
        self.assertIn("HOME", env)
        self.assertEqual(env["DOCKER_CONFIG"], "/tmp/docker-config")
        self.assertEqual(env["LC_ALL"], "C.UTF-8")
        # Venv wiring is intact: venv bin leads PATH, VIRTUAL_ENV is set.
        venv_bin = str(Path(resolved.venv_dir) / "bin")
        self.assertTrue(env["PATH"].startswith(venv_bin))
        self.assertEqual(env["VIRTUAL_ENV"], resolved.venv_dir)

    def test_declared_passthrough_still_delivers_secrets(self) -> None:
        # The EnvSpec.env ${VAR} declaration is the explicit opt-in seam for
        # children that genuinely need a host secret (e.g. judge API keys).
        provider = self._provider(
            env={"FAKE_HOST_SECRET": "${FAKE_HOST_SECRET}"}
        )
        env = provider._venv_env(provider.resolve())
        self.assertEqual(env["FAKE_HOST_SECRET"], "do-not-leak")

    def test_pip_mirror_config_only_reaches_install_steps(self) -> None:
        # PIP_INDEX_URL commonly embeds mirror credentials: it must reach the
        # venv install commands but NEVER the oracle/eval children that run
        # untrusted candidate code.
        os.environ["PIP_INDEX_URL"] = "https://user:secret@mirror.local/simple"
        try:
            provider = self._provider()
            resolved = provider.resolve()
            child = provider._venv_env(resolved)
            self.assertNotIn("PIP_INDEX_URL", child)
            install = provider._install_env(resolved)
            self.assertEqual(
                install["PIP_INDEX_URL"],
                "https://user:secret@mirror.local/simple",
            )
            # The install env is the child allowlist PLUS pip config.
            self.assertNotIn("FAKE_HOST_SECRET", install)
        finally:
            os.environ.pop("PIP_INDEX_URL", None)

    def test_credentialed_proxy_only_reaches_install_steps(self) -> None:
        # A user:pass@ proxy URL is a host secret: children get only
        # credential-free proxies, the install steps get them all.
        os.environ["HTTPS_PROXY"] = "http://user:pass@proxy.local:3128"
        os.environ["HTTP_PROXY"] = "http://proxy.local:3128"
        os.environ["ALL_PROXY"] = "socks5://gw.local:1080"
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        try:
            provider = self._provider()
            resolved = provider.resolve()
            child = provider._venv_env(resolved)
            self.assertNotIn("HTTPS_PROXY", child)
            self.assertEqual(child["HTTP_PROXY"], "http://proxy.local:3128")
            # SOCKS-only environments rely on ALL_PROXY: credential-free
            # values pass to children like the HTTP(S) variants.
            self.assertEqual(child["ALL_PROXY"], "socks5://gw.local:1080")
            self.assertEqual(child["NO_PROXY"], "localhost,127.0.0.1")
            install = provider._install_env(resolved)
            self.assertEqual(
                install["HTTPS_PROXY"], "http://user:pass@proxy.local:3128"
            )
            self.assertEqual(install["ALL_PROXY"], "socks5://gw.local:1080")
        finally:
            for key in (
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
            ):
                os.environ.pop(key, None)

    def test_extra_env_still_wins(self) -> None:
        provider = self._provider()
        env = provider._venv_env(
            provider.resolve(), extra_env={"INJECTED": "yes"}
        )
        self.assertEqual(env["INJECTED"], "yes")


class ComputeResourceBudgetTest(unittest.TestCase):
    """compute_resource_budget clamps workers to host + estimates."""

    def test_clamps_to_host_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = EnvSpec(
                name="x",
                parallelism=OracleParallelism(
                    model="batch_internal",
                    default_workers=8,
                    max_workers=16,
                ),
                resource_estimate=ResourceEstimate(
                    cpu_per_worker=4,
                    memory_gb_per_worker=4.0,
                    disk_gb_per_worker=20.0,
                ),
            )
            budget = compute_resource_budget(spec, cache_dir=tmp)
        # Never exceed the adapter's declared max.
        self.assertLessEqual(budget.max_workers, 16)
        self.assertGreaterEqual(budget.max_workers, 1)
        # cpu_cores reflects the actual host, not the per-worker estimate.
        self.assertEqual(budget.cpu_cores, max(1, os.cpu_count() or 1))
        # memory tracks the chosen worker count.
        self.assertEqual(budget.memory_gb, 4.0 * budget.max_workers)

    def test_override_lowers_workers(self) -> None:
        spec = EnvSpec(
            name="x",
            parallelism=OracleParallelism(
                model="batch_internal",
                default_workers=8,
                max_workers=16,
            ),
            resource_estimate=ResourceEstimate(cpu_per_worker=1),
        )
        budget = compute_resource_budget(spec, max_workers_override=2)
        self.assertLessEqual(budget.max_workers, 2)

    def test_no_docker_zero_containers(self) -> None:
        spec = EnvSpec(name="x", requires_docker=False)
        budget = compute_resource_budget(spec)
        self.assertEqual(budget.docker_containers, 0)

    def test_docker_containers_track_workers(self) -> None:
        spec = EnvSpec(
            name="x",
            requires_docker=True,
            parallelism=OracleParallelism(
                model="batch_internal",
                default_workers=2,
                max_workers=4,
            ),
            resource_estimate=ResourceEstimate(cpu_per_worker=1),
        )
        budget = compute_resource_budget(spec)
        self.assertEqual(
            budget.docker_containers, budget.max_workers
        )


def _docker_ok_runner(argv, **kwargs):
    """Fake runner: succeeds for `docker info`, generic success otherwise."""
    return CommandResult(
        argv=list(argv),
        returncode=0,
        timed_out=False,
        stdout="27.0.0\n",
        stderr="",
    )


def _no_git_runner(argv, **kwargs):
    """Fake runner returning empty HEAD (no real git invoked)."""
    return CommandResult(
        argv=list(argv), returncode=128, timed_out=False, stderr="no repo"
    )


class EnvProviderResolveTest(unittest.TestCase):
    """resolve() honors cache-dir precedence and derives paths."""

    def test_resolve_uses_explicit_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnvProvider(
                EnvSpec(
                    name="susvibes",
                    upstream_url="https://example.invalid/repo",
                    upstream_ref="dd28a7e",
                ),
                cache_dir=tmp,
            )
            resolved = provider.resolve()
            self.assertIsInstance(resolved, ResolvedEnv)
            self.assertEqual(resolved.cache_dir, str(Path(tmp).resolve()))
            self.assertTrue(
                resolved.upstream_dir.endswith("upstreams/susvibes")
            )
            self.assertTrue(
                resolved.venv_python.endswith("envs/susvibes/bin/python")
            )

    def test_geh_cache_dir_env_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prior = os.environ.get("GEH_CACHE_DIR")
            os.environ["GEH_CACHE_DIR"] = tmp
            try:
                provider = EnvProvider(EnvSpec(name="x"))
                resolved = provider.resolve()
            finally:
                if prior is None:
                    os.environ.pop("GEH_CACHE_DIR", None)
                else:
                    os.environ["GEH_CACHE_DIR"] = prior
            self.assertEqual(
                resolved.cache_dir, str(Path(tmp).resolve())
            )


class EnvProviderCheckTest(unittest.TestCase):
    """check() doctor probes on an empty cache dir."""

    def _provider(self, tmp: str, **kwargs) -> EnvProvider:
        spec = EnvSpec(
            name="susvibes",
            upstream_url="https://example.invalid/repo",
            upstream_ref="dd28a7e224b09e3ee666ffbcb56b95d109d2f8d7",
            requires_docker=True,
        )
        return EnvProvider(spec, cache_dir=tmp, **kwargs)

    def test_empty_cache_reports_missing_checkout_and_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(
                tmp,
                runner=_docker_ok_runner,
                which_fn=lambda b: "/usr/bin/docker"
                if b == "docker"
                else "/usr/bin/git",
            )
            checks = {c.name: c for c in provider.check()}
            self.assertIn("upstream_checkout", checks)
            self.assertFalse(checks["upstream_checkout"].ok)
            self.assertIn(
                "missing", checks["upstream_checkout"].detail
            )
            self.assertIn("venv", checks)
            self.assertFalse(checks["venv"].ok)
            self.assertTrue(checks["venv"].detail)
            self.assertEqual(checks["venv"].severity, "error")

    def test_docker_probe_uses_injected_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(
                tmp,
                runner=_docker_ok_runner,
                which_fn=lambda b: "/usr/bin/docker",
            )
            checks = {c.name: c for c in provider.check()}
            self.assertIn("docker_daemon", checks)
            self.assertTrue(checks["docker_daemon"].ok)

    def test_docker_missing_binary_fails_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(
                tmp,
                runner=_docker_ok_runner,
                which_fn=lambda b: None,
            )
            checks = {c.name: c for c in provider.check()}
            self.assertFalse(checks["docker_daemon"].ok)
            self.assertIn("PATH", checks["docker_daemon"].detail)

    def test_check_no_docker_required_skips_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(tmp, runner=_no_git_runner)
            checks = {
                c.name: c
                for c in provider.check(require_docker=False)
            }
            self.assertNotIn("docker_daemon", checks)


if __name__ == "__main__":
    unittest.main()
