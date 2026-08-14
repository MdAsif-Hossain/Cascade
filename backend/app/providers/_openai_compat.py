"""Shared implementation for providers that speak the OpenAI chat-completions dialect.

Groq and OpenRouter both expose ``/v1/chat/completions`` and ``/v1/models`` with
OpenAI's request and response shapes. Duplicating the parsing and error mapping
across both is exactly how two adapters drift apart, so the shared behaviour
lives here and each provider subclasses it to declare only what differs: base
URL, provider name, and any extra headers.

Gemini does not fit this dialect and has its own adapter.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import httpx

from app.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import DEFAULT_MAX_TOKENS, Completion, Message, Provider


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Read the provider's own backoff hint, so retries wait the requested time.

    Guessing a backoff when the provider has told us the answer is how a free
    tier turns a soft throttle into a hard ban.
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        # RFC 7231 also permits an HTTP-date here. Callers fall back to their
        # own backoff rather than parsing a date format providers rarely send.
        return None


def raise_for_status(provider: str, response: httpx.Response, model: str | None = None) -> None:
    """Translate an HTTP failure into Cascade's error vocabulary.

    Shared by every adapter so that a 429 from Groq and a 429 from OpenRouter
    are indistinguishable to the router.
    """
    if response.is_success:
        return

    status = response.status_code
    message = _extract_error_message(response)
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


def _extract_error_message(response: httpx.Response) -> str:
    """Pull a human-readable message out of whichever envelope the provider used."""
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


class OpenAICompatProvider(Provider):
    """Adapter for providers exposing OpenAI-compatible chat completions."""

    base_url: ClassVar[str]
    extra_headers: ClassVar[dict[str, str]] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", **self.extra_headers}

    async def list_models(self) -> list[str]:
        try:
            response = await self.client.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"model list timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(self.name, f"model list failed: {exc}") from exc

        raise_for_status(self.name, response)

        try:
            data = response.json()["data"]
            return [str(entry["id"]) for entry in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                self.name, f"unexpected model list shape: {exc}", status_code=response.status_code
            ) from exc

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.name, f"completion timed out: {exc}", model=model
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                self.name, f"completion failed: {exc}", model=model
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        raise_for_status(self.name, response, model)
        return self._parse_completion(response, model, latency_ms)

    def _parse_completion(
        self, response: httpx.Response, model: str, latency_ms: int
    ) -> Completion:
        try:
            body = response.json()
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                self.name,
                f"unexpected completion shape: {exc}",
                status_code=response.status_code,
                model=model,
            ) from exc

        text = choice.get("message", {}).get("content")
        finish_reason = str(choice.get("finish_reason") or "unknown")

        if not text or not text.strip():
            # A reasoning model that spent its whole budget thinking returns
            # HTTP 200 with null content. Surfacing it as an error keeps the
            # verifier from scoring a provider truncation as a wrong answer.
            raise ProviderResponseError(
                self.name,
                f"no answer text in response (finish_reason={finish_reason})",
                status_code=response.status_code,
                model=model,
            )

        usage = body.get("usage") or {}
        return Completion(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
