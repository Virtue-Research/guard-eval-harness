"""EnvProvider: resolve, doctor-check, acquire, and run upstream envs.

The provider owns external process execution for the vibecoding subsystem. It
resolves an :class:`EnvSpec` into concrete on-disk paths under a ``.geh/``
cache, runs read-only ``check`` probes (the ``doctor`` command), acquires the
upstream checkout (git clone + pinned-ref checkout) and an isolated venv when
asked, and runs upstream commands via :mod:`vibecoding.subprocess` with the
venv on ``PATH``.

Adapters never spawn processes themselves; they call :meth:`EnvProvider.run`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from guard_eval_harness.execution.artifacts import (
    dump_model,
    sanitize_payload_for_artifacts,
)
from guard_eval_harness.vibecoding.cache import resolve_cache_dir
from guard_eval_harness.vibecoding.interfaces import (
    OracleRunConfig,
    RawOracleResult,
    StagedOracleInput,
)
from guard_eval_harness.vibecoding.schema import (
    EnvSpec,
    ResourceBudget,
    VibeModel,
)
from guard_eval_harness.vibecoding.subprocess import (
    CommandResult,
    run_command,
    which,
)

Severity = Literal["error", "warning", "info"]


class ResolvedEnv(VibeModel):
    """Concrete on-disk locations resolved from an :class:`EnvSpec`."""

    name: str = Field(min_length=1)
    cache_dir: str
    upstream_dir: str
    venv_dir: str
    venv_python: str
    upstream_url: str | None = None
    upstream_ref: str | None = None
    workdir: str


class EnvCheck(VibeModel):
    """One read-only doctor probe result."""

    name: str = Field(min_length=1)
    ok: bool
    detail: str = ""
    severity: Severity = "error"


class EnvProvider:
    """Resolve, check, acquire, and run a single upstream environment."""

    def __init__(
        self,
        env_spec: EnvSpec,
        *,
        cache_dir: str | Path | None = None,
        runner: Any = run_command,
        which_fn: Any = which,
    ) -> None:
        self.env_spec = env_spec
        self._cache_dir = resolve_cache_dir(cache_dir)
        # Injected for testability; default to the real implementations.
        self._run_command = runner
        self._which = which_fn
        self._resolved: ResolvedEnv | None = None

    # --- resolution ---------------------------------------------------

    def resolve(self) -> ResolvedEnv:
        """Resolve concrete on-disk paths (cached after first call)."""
        if self._resolved is not None:
            return self._resolved
        name = self.env_spec.name
        cache = self._cache_dir
        upstream_dir = (
            Path(self.env_spec.root)
            if self.env_spec.root
            else cache / "upstreams" / name
        )
        venv_dir = cache / "envs" / name
        if self.env_spec.python:
            venv_python = Path(self.env_spec.python)
        else:
            venv_python = venv_dir / "bin" / "python"
        workdir = (
            Path(self.env_spec.workdir)
            if self.env_spec.workdir
            else upstream_dir
        )
        self._resolved = ResolvedEnv(
            name=name,
            cache_dir=str(cache),
            upstream_dir=str(upstream_dir),
            venv_dir=str(venv_dir),
            venv_python=str(venv_python),
            upstream_url=self.env_spec.upstream_url,
            upstream_ref=self.env_spec.upstream_ref,
            workdir=str(workdir),
        )
        return self._resolved

    # --- doctor (read-only) -------------------------------------------

    def check(
        self,
        *,
        budget: ResourceBudget | None = None,
        require_docker: bool | None = None,
    ) -> list[EnvCheck]:
        """Run read-only doctor probes; never mutate the environment."""
        resolved = self.resolve()
        checks: list[EnvCheck] = []
        checks.append(self._check_python())
        checks.append(self._check_upstream_checkout(resolved))
        checks.append(self._check_upstream_ref(resolved))
        checks.append(self._check_venv(resolved))
        want_docker = (
            self.env_spec.requires_docker
            if require_docker is None
            else require_docker
        )
        if want_docker:
            checks.append(self._check_docker_daemon())
            checks.append(self._check_images())
        checks.append(self._check_disk(resolved, budget))
        checks.append(self._check_secrets())
        checks.append(self._check_dataset_files(resolved))
        return checks

    def _check_python(self) -> EnvCheck:
        return EnvCheck(
            name="python",
            ok=True,
            detail=f"host python {sys.version.split()[0]}",
            severity="info",
        )

    def _check_upstream_checkout(self, resolved: ResolvedEnv) -> EnvCheck:
        path = Path(resolved.upstream_dir)
        ok = (path / ".git").exists()
        if ok:
            detail = f"checkout present at {path}"
        else:
            detail = (
                f"upstream checkout missing at {path}; "
                f"run `geh vibe acquire --dataset {resolved.name}` to clone it"
            )
        return EnvCheck(name="upstream_checkout", ok=ok, detail=detail)

    def _check_upstream_ref(self, resolved: ResolvedEnv) -> EnvCheck:
        path = Path(resolved.upstream_dir)
        want = resolved.upstream_ref
        if want is None:
            return EnvCheck(
                name="upstream_ref",
                ok=True,
                detail="no pinned ref declared",
                severity="warning",
            )
        if not (path / ".git").exists():
            return EnvCheck(
                name="upstream_ref",
                ok=False,
                detail=f"cannot verify ref {want}: no checkout",
            )
        result = self._git(
            ["rev-parse", "HEAD"], cwd=resolved.upstream_dir
        )
        head = (result.stdout or "").strip()
        ok = bool(head) and (
            head.startswith(want) or want.startswith(head[:7])
        )
        detail = f"HEAD={head or '?'} want={want}"
        return EnvCheck(name="upstream_ref", ok=ok, detail=detail)

    def _check_venv(self, resolved: ResolvedEnv) -> EnvCheck:
        python = Path(resolved.venv_python)
        ok = python.exists()
        if ok:
            detail = f"venv python at {python}"
        else:
            detail = (
                f"isolated venv missing at {resolved.venv_dir}; "
                f"run `geh vibe acquire --dataset {resolved.name}` to build it"
            )
        return EnvCheck(name="venv", ok=ok, detail=detail)

    def _check_docker_daemon(self) -> EnvCheck:
        docker = self._which("docker")
        if not docker:
            return EnvCheck(
                name="docker_daemon",
                ok=False,
                detail="docker binary not found on PATH",
            )
        result = self._run_command(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            timeout_s=30,
        )
        ok = (not result.timed_out) and result.returncode == 0
        if ok:
            detail = f"docker server {result.stdout.strip()}"
        else:
            detail = "docker daemon not reachable"
        return EnvCheck(name="docker_daemon", ok=ok, detail=detail)

    def _check_images(self) -> EnvCheck:
        # Image presence is dataset-specific; surface as a warning so the
        # doctor never hard-fails on pullable images.
        return EnvCheck(
            name="images",
            ok=True,
            detail="image presence not statically known; pulled on demand",
            severity="warning",
        )

    def _check_disk(
        self, resolved: ResolvedEnv, budget: ResourceBudget | None
    ) -> EnvCheck:
        need_gb = self.env_spec.disk_gb_estimate
        from guard_eval_harness.vibecoding.resources import _host_disk_gb

        free_gb = _host_disk_gb(resolved.cache_dir)
        ok = free_gb >= need_gb
        detail = f"free={free_gb:.1f}GiB estimate={need_gb:.1f}GiB"
        return EnvCheck(
            name="disk",
            ok=ok,
            detail=detail,
            severity="error" if not ok else "info",
        )

    def _check_secrets(self) -> EnvCheck:
        missing = [
            key
            for key, value in self.env_spec.env.items()
            if value.startswith("${")
            and value.endswith("}")
            and not os.environ.get(value[2:-1])
        ]
        ok = not missing
        if ok:
            detail = "required secrets present"
        else:
            detail = f"missing env secrets: {', '.join(sorted(missing))}"
        return EnvCheck(
            name="secrets",
            ok=ok,
            detail=detail,
            severity="warning",
        )

    def _check_dataset_files(self, resolved: ResolvedEnv) -> EnvCheck:
        files = self.env_spec.env.get("__dataset_files__")
        if not files:
            return EnvCheck(
                name="dataset_files",
                ok=True,
                detail="no dataset files declared",
                severity="info",
            )
        base = Path(resolved.upstream_dir)
        missing = [f for f in files.split(",") if not (base / f).exists()]
        ok = not missing
        detail = (
            "dataset files present"
            if ok
            else f"missing dataset files: {', '.join(missing)}"
        )
        return EnvCheck(name="dataset_files", ok=ok, detail=detail)

    # --- acquisition --------------------------------------------------

    def ensure_upstream(self, *, force: bool = False) -> ResolvedEnv:
        """Clone + checkout the pinned upstream ref (idempotent)."""
        resolved = self.resolve()
        url = resolved.upstream_url
        if not url:
            raise ValueError(
                f"env '{resolved.name}' has no upstream_url to clone"
            )
        upstream = Path(resolved.upstream_dir)
        has_git = (upstream / ".git").exists()
        if has_git and not force:
            self._checkout_ref(resolved)
            return resolved
        if force and upstream.exists():
            import shutil as _shutil

            _shutil.rmtree(upstream)
        upstream.parent.mkdir(parents=True, exist_ok=True)
        clone = self._git(
            [
                "clone",
                "--filter=blob:none",
                url,
                str(upstream),
            ],
        )
        if clone.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {resolved.name}: {clone.stderr}"
            )
        self._checkout_ref(resolved)
        return resolved

    def _checkout_ref(self, resolved: ResolvedEnv) -> None:
        ref = resolved.upstream_ref
        if not ref:
            return
        fetch = self._git(
            ["fetch", "--depth", "1", "origin", ref],
            cwd=resolved.upstream_dir,
        )
        # Some refs are already present after a full clone; don't hard-fail.
        _ = fetch
        checkout = self._git(
            ["checkout", ref], cwd=resolved.upstream_dir
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                f"git checkout {ref} failed for {resolved.name}: "
                f"{checkout.stderr}"
            )
        head = self._git(["rev-parse", "HEAD"], cwd=resolved.upstream_dir)
        got = (head.stdout or "").strip()
        if not (got.startswith(ref) or ref.startswith(got[:7])):
            raise RuntimeError(
                f"HEAD {got} does not match pinned ref {ref}"
            )

    def ensure_venv(self, *, force: bool = False) -> ResolvedEnv:
        """Create the isolated venv + run declared install steps."""
        resolved = self.resolve()
        python = Path(resolved.venv_python)
        venv_dir = Path(resolved.venv_dir)
        if python.exists() and not force:
            return resolved
        if force and venv_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(venv_dir)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        create = self._run_command(
            [sys.executable, "-m", "venv", str(venv_dir)],
        )
        if create.returncode != 0:
            raise RuntimeError(
                f"venv creation failed for {resolved.name}: {create.stderr}"
            )
        for step in self.env_spec.install:
            argv = step.split() if isinstance(step, str) else list(step)
            result = self._run_command(
                argv,
                cwd=resolved.upstream_dir,
                env=self._install_env(resolved),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"install step failed ({step!r}): {result.stderr}"
                )
        return resolved

    def ensure_ready(self, *, force: bool = False) -> ResolvedEnv:
        """Acquire upstream checkout + venv so the env can run."""
        self.ensure_upstream(force=force)
        return self.ensure_venv(force=force)

    # --- execution ----------------------------------------------------

    def run(
        self,
        argv: list[str],
        *,
        run_dir: str | Path,
        timeout_s: float | None = None,
        budget: ResourceBudget | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run an upstream command with the venv on PATH + resolved env.

        Persists a redacted copy of the :class:`CommandResult` under
        ``run_dir/upstream/<name>/logs/`` and captures stdout/stderr there.
        """
        resolved = self.resolve()
        env = self._venv_env(resolved, extra_env=extra_env)
        logs_dir = (
            Path(run_dir) / "upstream" / resolved.name / "logs"
        )
        result = self._run_command(
            list(argv),
            cwd=resolved.workdir,
            env=env,
            timeout_s=timeout_s,
            capture_to_files=str(logs_dir),
        )
        self._persist_result(result, logs_dir)
        return result

    # Host vars forwarded to oracle subprocesses. Everything else is DROPPED:
    # the children execute upstream eval tooling and, transitively, untrusted
    # candidate code, so host secrets (API keys, tokens, cloud creds) must
    # reach them only via an explicit ``EnvSpec.env`` declaration (the
    # ``${VAR}`` passthrough) or ``extra_env`` -- never by inheriting the
    # whole parent environment. The list covers process basics, locale/temp,
    # the Docker client config the Docker-backed oracles need to drive the
    # daemon, and CA settings so corp networks keep working. ``LC_*`` passes
    # as a prefix. Proxy vars are handled separately below: they are
    # forwarded only when the URL carries NO credentials (a
    # ``user:pass@proxy`` URL is a host secret like any other). ``PIP_*`` is
    # deliberately NOT forwarded here: authenticated mirrors commonly embed
    # credentials in PIP_INDEX_URL / PIP_EXTRA_INDEX_URL, and oracle children
    # (which run untrusted candidate code) must never see them -- pip config
    # and credentialed proxies reach ONLY the venv install steps via
    # :meth:`_install_env`.
    _HOST_ENV_ALLOWLIST = frozenset(
        {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LANGUAGE",
            "TZ",
            "TERM",
            "COLUMNS",
            "DOCKER_HOST",
            "DOCKER_CONFIG",
            "DOCKER_CERT_PATH",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CONTEXT",
            "DOCKER_API_VERSION",
            "NO_PROXY",
            "no_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
        }
    )
    _HOST_ENV_ALLOWED_PREFIXES = ("LC_",)
    # Proxy URL vars: forwarded to children only when credential-free (the
    # NO_PROXY host lists above are always safe and stay in the allowlist).
    _PROXY_ENV_VARS = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )

    @staticmethod
    def _proxy_has_credentials(value: str) -> bool:
        """True when a proxy URL embeds userinfo (``user:pass@host``).

        Unparseable values are treated as credentialed (conservative): when
        in doubt, the var stays out of the child env.
        """
        from urllib.parse import urlsplit

        try:
            parsed = urlsplit(
                value if "://" in value else f"http://{value}"
            )
            return (
                parsed.username is not None or parsed.password is not None
            )
        except ValueError:
            return True

    def _install_env(self, resolved: ResolvedEnv) -> dict[str, str]:
        """Env for the venv install steps: the child allowlist + ``PIP_*``.

        Pip mirror config (PIP_INDEX_URL etc.) and credentialed proxy URLs
        may carry secrets, so they are forwarded ONLY to the install
        commands -- never to the oracle/eval subprocesses built by
        :meth:`_venv_env` (which forwards only credential-free proxies).
        """
        env = self._venv_env(resolved)
        env.update(
            {
                key: value
                for key, value in os.environ.items()
                if key.startswith("PIP_")
            }
        )
        for key in self._PROXY_ENV_VARS:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _venv_env(
        self,
        resolved: ResolvedEnv,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build the child env: allowlisted host vars + venv PATH + EnvSpec.env.

        The host environment is filtered through
        :data:`_HOST_ENV_ALLOWLIST` (see its note): artifact redaction only
        protects what is PERSISTED, so the live child must simply never
        receive undeclared host secrets in the first place.
        """
        env = {
            key: value
            for key, value in os.environ.items()
            if key in self._HOST_ENV_ALLOWLIST
            or key.startswith(self._HOST_ENV_ALLOWED_PREFIXES)
        }
        for key in self._PROXY_ENV_VARS:
            value = os.environ.get(key)
            if value and not self._proxy_has_credentials(value):
                env[key] = value
        venv_bin = str(Path(resolved.venv_dir) / "bin")
        existing_path = env.get("PATH", "")
        env["PATH"] = (
            f"{venv_bin}{os.pathsep}{existing_path}"
            if existing_path
            else venv_bin
        )
        env["VIRTUAL_ENV"] = resolved.venv_dir
        for key, value in self.env_spec.env.items():
            if key == "__dataset_files__":
                continue
            if value.startswith("${") and value.endswith("}"):
                host = os.environ.get(value[2:-1])
                if host is not None:
                    env[key] = host
                continue
            env[key] = value
        if extra_env:
            env.update(extra_env)
        return env

    def _persist_result(
        self, result: CommandResult, logs_dir: Path
    ) -> None:
        """Write a redacted CommandResult JSON next to the captured logs."""
        redacted_payload = sanitize_payload_for_artifacts(
            result.model_dump(mode="json")
        )
        redacted = CommandResult.model_validate(redacted_payload)
        dump_model(logs_dir / "command_result.json", redacted)

    def _git(
        self,
        args: list[str],
        *,
        cwd: str | Path | None = None,
    ) -> CommandResult:
        """Run a git subcommand via the injected runner."""
        git = self._which("git") or "git"
        return self._run_command([git, *args], cwd=cwd)

    # --- interfaces.EnvProvider Protocol ------------------------------

    def evaluate(
        self,
        env: Any,
        staged: StagedOracleInput,
        run_config: OracleRunConfig,
        resource_budget: ResourceBudget,
    ) -> RawOracleResult:
        """Protocol seam: adapters drive staging then call :meth:`run`.

        The concrete provider exposes ``run`` for adapters to invoke; the
        generic ``evaluate`` entry point is intentionally not the path used
        by oracle adapters and raises to make misuse loud.
        """
        raise NotImplementedError(
            "EnvProvider.run() is the execution seam; oracle adapters call "
            "run() with their staged argv rather than evaluate()"
        )


__all__ = ["EnvProvider", "ResolvedEnv", "EnvCheck"]
