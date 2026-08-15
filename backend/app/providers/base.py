"""The contract every provider adapter satisfies.

Cascade routes the same question across providers and compares what comes back,
so the adapters must be interchangeable in more than name: identical failure
vocabulary (see ``core.errors``), identical token accounting, and identical
handling of a 200 response that carries no usable answer.

That last point is why ``Completion.text`` is guaranteed non-empty. An adapter
that lets a null or blank completion through would corrupt the verifier's
signal — the escalation loop would read a provider bug as question difficulty.
Adapters raise ``ProviderResponseError`` instead.
"""

from __future__ import annotations

import abc
from typing import ClassVar, Literal

import httpx
from pydantic import BaseModel, Field, field_validator

Role = Literal["system", "user", "assistant"]

# Reasoning models spend completion tokens before emitting any answer text. A
# budget that is merely "enough for the answer" returns finish_reason="length"
# with content: null. Observed on four separate OpenRouter free models during
# the Phase 0 gate; 1024 clears it comfortably for study-assistant answers.
DEFAULT_MAX_TOKENS = 1024

DEFAULT_TIMEOUT_SECONDS = 60.0


class Message(BaseModel):
    """One turn of a chat exchange, in the shape Groq and OpenRouter accept natively."""

    role: Role
    content: str = Field(min_length=1)


class Completion(BaseModel):
    """A normalised successful completion.

    Token counts are what the counterfactual cost calculation multiplies against
    published per-token pricing, so an adapter that reports them inconsistently
    silently corrupts the project's headline metric. The contract test suite
    checks these explicitly for every adapter.
    """

    text: str = Field(min_length=1)
    provider: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    finish_reason: str

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, v: str) -> str:
        """Whitespace-only answers are as useless as empty ones and must not reach the verifier."""
        if not v.strip():
            raise ValueError("completion text is blank")
        return v

    @field_validator("finish_reason")
    @classmethod
    def _normalise_finish_reason(cls, v: str) -> str:
        """Fold case so callers can compare finish reasons without knowing the provider.

        Live calls return ``"stop"`` from Groq and ``"STOP"`` from Gemini for the
        same outcome. Anything downstream that branches on this value — the
        verifier's truncation heuristic especially — would silently miss one
        provider's spelling.
        """
        return v.strip().lower()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(abc.ABC):
    """Base class for provider adapters.

    The HTTP client is injected rather than constructed per call so connection
    pooling survives across requests — on Render's free tier, TLS handshakes on
    every call are a meaningful share of latency.
    """

    name: ClassVar[str]

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this adapter created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """Return the provider's currently advertised model ids.

        Used by the catalog poller to detect models disappearing out from under
        a tier definition, which free tiers do without notice.
        """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Completion:
        """Generate one completion.

        Raises a ``ProviderError`` subclass on every failure path — adapters never
        return a sentinel value or a partially populated ``Completion``.
        """
