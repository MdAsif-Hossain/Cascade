"""HTTP failure mapping shared by every adapter, regardless of wire dialect.

Lives apart from any one provider's request format because the router's failover
logic depends on a 429 from Gemini being indistinguishable from a 429 from Groq.
If this mapping were duplicated per adapter it would drift, and the drift would
show up as inexplicable routing behaviour rather than as a test failure.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Read the provider's own backoff hint, so retries wait the requested time.

    Guessing a backoff when the provider has already told us the answer is how a
    free tier turns a soft throttle into a hard ban.
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        # RFC 7231 also permits an HTTP-date here. Callers fall back to their own
        # backoff rather than parsing a date format providers rarely send.
        return None


def extract_error_message(response: httpx.Response) -> str:
    """Pull a human-readable message out of whichever envelope the provider used.

    Groq and OpenRouter nest it under ``error.message``; Gemini uses the same key
    but adds a ``status`` enum; some upstream failures arrive as bare text.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if isinstance(error, str):
            return error
        if "message" in body:
            return str(body["message"])
    return response.text[:300] or f"HTTP {response.status_code}"


def raise_for_status(provider: str, response: httpx.Response, model: str | None = None) -> None:
    """Translate an HTTP failure into Cascade's error vocabulary."""
    if response.is_success:
        return

    status = response.status_code
    message = extract_error_message(response)
    kwargs: dict[str, Any] = {"status_code": status, "model": model}

    if status in (401, 403):
        raise ProviderAuthError(provider, message, **kwargs)
    if status == 402:
        raise ProviderQuotaError(provider, message, **kwargs)
    if status == 429:
        raise ProviderRateLimitError(
            provider, message, retry_after=parse_retry_after(response.headers), **kwargs
        )
    if status >= 500:
        raise ProviderUnavailableError(provider, message, **kwargs)
    raise ProviderError(provider, message, **kwargs)
