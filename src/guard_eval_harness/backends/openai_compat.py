"""OpenAI-compatible HTTP backend (works for OpenAI, vLLM server, LiteLLM, etc.)."""

import logging
import os
from typing import Any, Sequence

from guard_eval_harness.backends.base import (
    BackendConfig,
    GenerationBackend,
    backend_registry,
)
from guard_eval_harness.schemas import MediaPart, Message, TextPart

_log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT = 60.0


@backend_registry.register("openai_compat")
class OpenAICompatibleBackend(GenerationBackend):
    """Chat-completions backend for any OpenAI-compatible endpoint.

    Configurable via ``args``:
      - ``base_url`` (str): defaults to ``https://api.openai.com/v1``.
        Override to point at a self-hosted vLLM / LiteLLM endpoint.
      - ``api_key_env`` (str): env var holding the bearer token.
        Defaults to ``OPENAI_API_KEY``. Set to empty / null for
        endpoints that don't require auth.
      - ``timeout`` (float): per-request timeout in seconds (default 60).
      - ``max_retries`` (int): exponential-backoff retries (default 3).
      - ``concurrency`` (int): unused today (sequential generate).
      - ``extra_headers`` (dict): merged into every request.
      - ``token_param`` (str): the body field name for the output-token
        budget. ``"max_tokens"`` (default, classic chat/completions) or
        ``"max_completion_tokens"`` (required for OpenAI o-series /
        reasoning models — gpt-5.x, o1, o3, …).
      - ``reasoning_effort`` (str | null): if set, forwarded verbatim as
        ``reasoning_effort`` in the request body (e.g. ``"low"`` /
        ``"medium"`` / ``"high"``). Required for reasoning models.
      - ``omit_temperature`` (bool): default ``False``. Set ``True`` for
        models that reject the ``temperature`` field outright
        (some reasoning endpoints).

    The configured ``BackendConfig.model`` is the model name passed in
    the request body (e.g. ``gpt-4o-mini``, ``meta-llama/Llama-3.1-8B-Instruct``).
    """

    kind = "openai_compat"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        args = config.args
        self.base_url: str = str(
            args.get("base_url", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout: float = float(args.get("timeout", _DEFAULT_TIMEOUT))
        self.max_retries: int = int(args.get("max_retries", 3))
        self.extra_headers: dict[str, str] = dict(
            args.get("extra_headers", {})
        )
        token_param = str(args.get("token_param", "max_tokens"))
        if token_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "openai_compat.args.token_param must be 'max_tokens' "
                f"or 'max_completion_tokens', got {token_param!r}"
            )
        self.token_param: str = token_param
        self.reasoning_effort: str | None = args.get("reasoning_effort")
        self.omit_temperature: bool = bool(
            args.get("omit_temperature", False)
        )

        api_key_env = args.get("api_key_env", "OPENAI_API_KEY")
        self._api_key: str | None = None
        if api_key_env:
            self._api_key = os.environ.get(api_key_env)
            if not self._api_key:
                raise RuntimeError(
                    f"OpenAICompatibleBackend: env var {api_key_env!r} "
                    "is unset; provide an API key or set "
                    "args.api_key_env to a different variable, or "
                    "to null for keyless endpoints."
                )

        if config.model is None:
            raise ValueError(
                "OpenAICompatibleBackend requires backend.name "
                "(the model name to pass to the endpoint)."
            )
        self.model_name: str = config.model

        self._client = None  # lazy-init httpx.Client

    # ---------------------------------------------------------------
    # Message serialization
    # ---------------------------------------------------------------

    @staticmethod
    def _serialize_content(message: Message) -> Any:
        """Convert a Message to OpenAI-style content (str or parts list)."""
        if isinstance(message.content, str):
            return message.content
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, MediaPart):
                if part.media.modality != "image":
                    raise ValueError(
                        "OpenAICompatibleBackend only supports image media "
                        f"parts; got {part.media.modality!r}"
                    )
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.media.uri},
                    }
                )
        return parts

    @classmethod
    def _serialize_messages(
        cls,
        messages: Sequence[Message],
    ) -> list[dict[str, Any]]:
        """Convert harness Messages to OpenAI chat format."""
        return [
            {
                "role": message.role,
                "content": cls._serialize_content(message),
            }
            for message in messages
        ]

    # ---------------------------------------------------------------
    # HTTP plumbing
    # ---------------------------------------------------------------

    def _get_client(self):
        """Lazy-init an httpx.Client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "OpenAICompatibleBackend requires httpx: "
                    "pip install httpx"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self.extra_headers)
        return headers

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /chat/completions with exponential-backoff retry."""
        import time

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleBackend requires httpx: "
                "pip install httpx"
            ) from exc

        url = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().post(
                    url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code == 200:
                    return response.json()
                if (
                    response.status_code in {401, 403}
                    or response.status_code < 500
                ):
                    response.raise_for_status()
                last_exc = httpx.HTTPStatusError(
                    f"server error: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                last_exc = exc
            if attempt < self.max_retries:
                time.sleep(2.0 ** attempt)
        assert last_exc is not None
        raise last_exc

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
    ) -> list[str]:
        """Generate raw text for each conversation in the batch.

        Note: serial-by-conversation today (one HTTP call per item).
        OpenAI-compatible endpoints don't have a generic batch API.
        """
        outputs: list[str] = []
        for messages in batch:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": self._serialize_messages(messages),
                self.token_param: self.max_new_tokens,
            }
            if not self.omit_temperature:
                payload["temperature"] = self.temperature
            if self.reasoning_effort is not None:
                payload["reasoning_effort"] = self.reasoning_effort
            response = self._post_chat(payload)
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "OpenAICompatibleBackend: malformed response: "
                    f"{response!r}"
                ) from exc
            outputs.append(content or "")
        return outputs
