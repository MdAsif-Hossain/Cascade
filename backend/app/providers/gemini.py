"""Google AI Studio (Gemini) adapter.

Verified during the Phase 0 gate: 37 text models, 785 ms on
``gemini-flash-lite-latest``. The same key also serves ``gemini-embedding-001``,
which is what makes the classifier's embedder free (see ADR-0001).

Gemini does not speak the OpenAI dialect, so this adapter translates in both
directions:

* the model id is part of the URL path, not the request body
* the key travels in ``x-goog-api-key``, not a bearer header
* messages are ``contents`` with ``parts``, and the assistant role is ``model``
* a system message is a separate top-level ``systemInstruction``, not a turn
* usage is ``usageMetadata`` with ``promptTokenCount`` / ``candidatesTokenCount``

Only the normalised result crosses back out, so the router cannot tell the
difference between this adapter and the OpenAI-compatible ones.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import httpx

from app.core.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers._http import raise_for_status
from app.providers.base import DEFAULT_MAX_TOKENS, Completion, Message, Provider

_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiProvider(Provider):
    name: ClassVar[str] = "gemini"
    base_url: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> dict[str, str]:
        # Sent as a header rather than the ?key= query parameter so the secret
        # never lands in a URL, where it would leak into logs and error strings.
        return {"x-goog-api-key": self._api_key}

    async def list_models(self) -> list[str]:
        """Return advertised model ids with Gemini's ``models/`` prefix stripped.

        Unfiltered on purpose: the catalog poller's job is to notice when a model
        Cascade depends on stops being advertised, and filtering here would hide
        exactly that signal for any model whose capabilities changed.
        """
        try:
            response = await self.client.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"model list timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(self.name, f"model list failed: {exc}") from exc

        raise_for_status(self.name, response)

        try:
            models = response.json()["models"]
            return [str(m["name"]).removeprefix("models/") for m in models]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                self.name, f"unexpected model list shape: {exc}", status_code=response.status_code
            ) from exc

    def _build_payload(
        self, messages: list[Message], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, str]] = []

        for message in messages:
            if message.role == "system":
                # Gemini rejects a "system" turn inside contents; it belongs in
                # systemInstruction. Collapsing several into one preserves order.
                system_parts.append({"text": message.content})
                continue
            contents.append({"role": _ROLE_MAP[message.role], "parts": [{"text": message.content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        return payload

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Completion:
        url = f"{self.base_url}/models/{model}:generateContent"
        payload = self._build_payload(messages, max_tokens, temperature)

        started = time.perf_counter()
        try:
            response = await self.client.post(url, headers=self._headers(), json=payload)
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
            candidate = body["candidates"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                self.name,
                f"unexpected completion shape: {exc}",
                status_code=response.status_code,
                model=model,
            ) from exc

        finish_reason = str(candidate.get("finishReason") or "unknown")

        # A candidate blocked by a safety filter, or truncated by MAX_TOKENS
        # before emitting text, arrives with no parts at all. Same failure mode
        # as a reasoning model returning null content: HTTP 200, no answer.
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)

        if not text.strip():
            raise ProviderResponseError(
                self.name,
                f"no answer text in response (finishReason={finish_reason})",
                status_code=response.status_code,
                model=model,
            )

        usage = body.get("usageMetadata") or {}
        return Completion(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("promptTokenCount", 0)),
            completion_tokens=int(usage.get("candidatesTokenCount", 0)),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
