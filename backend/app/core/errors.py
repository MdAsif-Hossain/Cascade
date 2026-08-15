"""Cascade's error taxonomy.

Provider APIs disagree about how to signal the same failure: Groq and OpenRouter
speak OpenAI-style status codes, Gemini uses its own envelope, and each wraps
upstream failures differently. The router cannot make good failover decisions
against three different vocabularies, so every adapter normalises its failures
into the exceptions below.

The distinction that matters to the router is not which HTTP code came back but
whether the same request is worth sending again (``retryable``) and whether a
different provider in the tier should be tried instead (``should_failover``).
"""

from __future__ import annotations

from typing import ClassVar


class CascadeError(Exception):
    """Base for every error Cascade raises deliberately.

    Carries a stable ``error_code`` so the API layer can serialise errors without
    inspecting exception types or leaking a stack trace to the client.
    """

    error_code: ClassVar[str] = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProviderError(CascadeError):
    """A call to an upstream model provider failed.

    ``retryable`` means the identical request may succeed if sent again to the
    same provider after a delay. ``should_failover`` means another provider in
    the tier should be tried; it is false only when retrying elsewhere is
    pointless or actively harmful.
    """

    error_code: ClassVar[str] = "provider_error"
    retryable: ClassVar[bool] = False
    should_failover: ClassVar[bool] = True

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.model = model

    def __str__(self) -> str:
        parts = [self.provider]
        if self.model:
            parts.append(self.model)
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        return f"[{' '.join(parts)}] {self.message}"


class ProviderAuthError(ProviderError):
    """Credentials were rejected (401/403).

    Never retried: a bad or revoked key will stay bad, and hammering an auth
    endpoint is a good way to get an account flagged. Failover is still allowed
    because the other providers use unrelated credentials.
    """

    error_code: ClassVar[str] = "provider_auth_error"
    retryable: ClassVar[bool] = False


class ProviderQuotaError(ProviderError):
    """The provider requires payment or the free allowance is exhausted (402).

    Distinct from a rate limit: waiting does not help, because nothing about the
    request is the problem. Observed live on Cerebras during the Phase 0 gate,
    which is why it has its own class rather than being folded into 4xx.
    """

    error_code: ClassVar[str] = "provider_quota_error"
    retryable: ClassVar[bool] = False


class ProviderRateLimitError(ProviderError):
    """The provider is throttling us (429).

    Retryable after a delay. ``retry_after`` carries the provider's own hint in
    seconds when it sends one, so backoff can respect it instead of guessing.
    """

    error_code: ClassVar[str] = "provider_rate_limit"
    retryable: ClassVar[bool] = True

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(provider, message, status_code=status_code, model=model)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """The provider is down, unreachable, or returned a 5xx."""

    error_code: ClassVar[str] = "provider_unavailable"
    retryable: ClassVar[bool] = True


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the configured timeout."""

    error_code: ClassVar[str] = "provider_timeout"
    retryable: ClassVar[bool] = True


class ProviderResponseError(ProviderError):
    """The provider returned HTTP 200 with a body we cannot use.

    This is not a theoretical case. Reasoning models return ``content: null``
    with ``finish_reason: "length"`` when the token budget is consumed before
    the model leaves its reasoning phase — a successful HTTP call carrying no
    answer. Treating that as an empty answer would hand the student a blank
    response and, worse, feed the verifier something it would score as a
    failure and escalate, hiding a bug as a difficulty signal.

    Not retryable: the same request produces the same truncation.
    """

    error_code: ClassVar[str] = "provider_response_error"
    retryable: ClassVar[bool] = False
