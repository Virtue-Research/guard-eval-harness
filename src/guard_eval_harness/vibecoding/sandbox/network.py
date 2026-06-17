"""Network egress policy for live-agent generation.

This module is policy **data** plus a host-allow **predicate** -- it does not
install a kernel firewall. The sandbox/agent driver consults the predicate
(and the proxy env hints) to decide what the agent may reach.

The official default is **allowlist egress**: deny everything except the
model/API endpoints explicitly required to drive the agent. A ``denylist``
mode is also provided as extra defense for benchmarks that explicitly permit
network access: it allows everything except known-leak hosts (CVE advisories,
issue trackers, benchmark solution files).

Host matching is suffix-based on the registrable hostname: an allow entry of
``api.openai.com`` matches ``api.openai.com`` exactly and any subdomain of it,
but never a host that merely contains it as a substring (``notopenai.com``).
Port and scheme are ignored; the predicate operates on hostnames only.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import Field

from guard_eval_harness.execution.artifacts import sha256_payload
from guard_eval_harness.vibecoding.schema import VibeModel

# Two egress modes. ``allowlist`` is the official default (deny-all-except).
NetworkMode = str  # "allowlist" | "denylist"

_ALLOWLIST = "allowlist"
_DENYLIST = "denylist"
_VALID_MODES = (_ALLOWLIST, _DENYLIST)

# Always denied in denylist mode (known leak surfaces for upstream fixes).
_DEFAULT_DENY = (
    "cve.mitre.org",
    "nvd.nist.gov",
    "github.com",
    "githubusercontent.com",
    "gitlab.com",
    "huggingface.co",
    "bugzilla.redhat.com",
    "security-tracker.debian.org",
)


def _hostname(host_or_url: str) -> str:
    """Extract a bare lowercase hostname from a host or URL string.

    Accepts ``example.com``, ``https://example.com:443/path``, or
    ``example.com:8080`` and returns ``example.com``. Returns an empty string
    when no host can be parsed.
    """
    raw = (host_or_url or "").strip().lower()
    if not raw:
        return ""
    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
    else:
        # Strip any ``:port`` / path that may be present on a bare host.
        host = raw.split("/", 1)[0].split(":", 1)[0]
    return host.rstrip(".")


def _host_matches(host: str, pattern: str) -> bool:
    """True when ``host`` equals ``pattern`` or is a subdomain of it.

    Matching is registrable-name aware: ``api.openai.com`` matches the pattern
    ``openai.com`` (subdomain) and ``api.openai.com`` (exact) but ``openai.com``
    does NOT match the pattern ``api.openai.com`` (not a subdomain of it), and
    ``notopenai.com`` never matches ``openai.com`` (boundary-checked).
    """
    if not host or not pattern:
        return False
    if host == pattern:
        return True
    return host.endswith("." + pattern)


class NetworkPolicy(VibeModel):
    """A resolved egress policy: a mode plus allow/deny host lists.

    Use :meth:`is_host_allowed` / :meth:`is_url_allowed` as the predicate the
    sandbox consults. ``proxy_env`` produces ``*_PROXY``/``NO_PROXY`` hints a
    subprocess-launched agent can honor.
    """

    mode: str = _ALLOWLIST
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    # Hosts always reachable regardless of mode (e.g. loopback for local LLMs).
    always_allow: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )

    def is_host_allowed(self, host_or_url: str) -> bool:
        """Decide whether ``host_or_url`` may be reached under this policy.

        - allowlist mode: allowed only if it matches ``always_allow`` or an
          ``allow`` entry (default deny).
        - denylist mode: allowed unless it matches a ``deny`` entry; an
          ``always_allow`` match still wins over a deny.
        """
        host = _hostname(host_or_url)
        if not host:
            # Unparseable target: deny in allowlist mode, deny in denylist too
            # (a target we can't reason about is never safe to permit).
            return False
        if any(_host_matches(host, p) for p in self.always_allow):
            return True
        if self.mode == _DENYLIST:
            return not any(_host_matches(host, p) for p in self.deny)
        # allowlist (default): deny unless explicitly allowed.
        return any(_host_matches(host, p) for p in self.allow)

    def is_url_allowed(self, url: str) -> bool:
        """Convenience wrapper: allow-check a full URL by its hostname."""
        return self.is_host_allowed(url)

    def proxy_env(self) -> dict[str, str]:
        """Produce ``*_PROXY``/``NO_PROXY`` hints for a subprocess agent.

        In allowlist mode we point traffic at a blackhole proxy (so any host
        not in ``NO_PROXY`` is effectively cut off) and list the allowed hosts
        in ``NO_PROXY`` as direct-connect exceptions. In denylist mode we leave
        the proxy unset (network permitted) and only publish the deny set for
        an out-of-band filter to honor. These are *hints*; enforcement is the
        sandbox's job.
        """
        if self.mode == _DENYLIST:
            return {
                "GEH_NETWORK_MODE": _DENYLIST,
                "GEH_DENY_HOSTS": ",".join(sorted(set(self.deny))),
            }
        no_proxy = sorted(set(self.always_allow) | set(self.allow))
        blackhole = "http://127.0.0.1:9"  # discard port; no listener
        return {
            "GEH_NETWORK_MODE": _ALLOWLIST,
            "HTTP_PROXY": blackhole,
            "HTTPS_PROXY": blackhole,
            "http_proxy": blackhole,
            "https_proxy": blackhole,
            "NO_PROXY": ",".join(no_proxy),
            "no_proxy": ",".join(no_proxy),
            "GEH_ALLOW_HOSTS": ",".join(no_proxy),
        }

    def policy_id(self) -> str:
        """Stable, human-readable id for this policy's shape."""
        return f"network:{self.mode}"

    def policy_hash(self) -> str:
        """Canonical content hash of this policy (for cache/provenance keys)."""
        return sha256_payload(self.model_dump(mode="json"))


def build_policy(
    mode: str = _ALLOWLIST,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
) -> NetworkPolicy:
    """Build a :class:`NetworkPolicy`.

    ``mode`` is ``"allowlist"`` (official default: deny all except ``allow``)
    or ``"denylist"`` (permit all except ``deny``; the default leak-host deny
    set is always merged in). ``allow``/``deny`` are host or URL strings; URLs
    are reduced to hostnames.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"unknown network mode {mode!r}; expected one of {_VALID_MODES}"
        )
    allow_hosts = sorted(
        {_hostname(h) for h in (allow or []) if _hostname(h)}
    )
    deny_hosts = {_hostname(h) for h in (deny or []) if _hostname(h)}
    if mode == _DENYLIST:
        deny_hosts |= set(_DEFAULT_DENY)
    return NetworkPolicy(
        mode=mode,
        allow=allow_hosts,
        deny=sorted(deny_hosts),
    )
