"""Shared contract suite that every provider adapter must pass.

Three adapters written independently will look correct and behave differently:
each provider reports token usage under a different key, wraps errors in a
different envelope, and signals a truncated reasoning response differently. Those
divergences do not surface as crashes — they surface as a corrupted cost metric
and an escalation loop that reacts to provider bugs as if they were question
difficulty. One shared suite is what catches them.

Adapters are tested by subclassing ``ProviderContractSuite`` and supplying only
the provider-specific wire format. The suite class itself is deliberately not
named ``Test*`` so pytest does not collect it directly.
"""

from __future__ import annotations

import abc
from typing import Any

import httpx
import pytest
import respx

from app.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import Completion, Message, Provider

QUESTION = [Message(role="user", content="What is the derivative of x squared?")]


class ProviderContractSuite(abc.ABC):
    """Assertions that hold for every adapter regardless of provider."""

    # --- what each adapter must declare -------------------------------------

    @property
    @abc.abstractmethod
    def provider_cls(self) -> type[Provider]: ...

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """A model id to request. Only used to build URLs and echo back."""

    @property
    @abc.abstractmethod
    def completions_url(self) -> str: ...

    @property
    @abc.abstractmethod
    def models_url(self) -> str: ...

    @abc.abstractmethod
    def success_body(
        self,
        text: str = "The derivative is 2x.",
        prompt_tokens: int = 11,
        completion_tokens: int = 7,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        """A well-formed successful completion in this provider's wire format."""

    @abc.abstractmethod
    def no_content_body(self) -> dict[str, Any]:
        """A HTTP 200 body carrying no usable answer text.

        Reasoning models produce this when the token budget is exhausted before
        any answer is emitted — observed live on four OpenRouter free models.
        """

    @abc.abstractmethod
    def models_body(self, ids: list[str]) -> dict[str, Any]:
        """A model-list response in this provider's wire format."""

    def error_body(self, message: str = "boom") -> dict[str, Any]:
        """This provider's error envelope. Overridden where the shape differs."""
        return {"error": {"message": message}}

    # --- helpers ------------------------------------------------------------

    def build(self) -> Provider:
        return self.provider_cls("test-key-not-real")

    async def complete(self, provider: Provider) -> Completion:
        return await provider.complete(QUESTION, self.model)

    # --- success path -------------------------------------------------------

    @respx.mock
    async def test_completion_returns_the_answer_text(self):
        respx.post(self.completions_url).respond(200, json=self.success_body())
        result = await self.complete(self.build())
        assert result.text == "The derivative is 2x."

    @respx.mock
    async def test_completion_reports_its_own_provider_name(self):
        respx.post(self.completions_url).respond(200, json=self.success_body())
        result = await self.complete(self.build())
        assert result.provider == self.provider_cls.name

    @respx.mock
    async def test_completion_echoes_the_requested_model(self):
        respx.post(self.completions_url).respond(200, json=self.success_body())
        result = await self.complete(self.build())
        assert result.model == self.model

    @respx.mock
    async def test_token_counts_are_normalised_across_providers(self):
        respx.post(self.completions_url).respond(
            200, json=self.success_body(prompt_tokens=11, completion_tokens=7)
        )
        result = await self.complete(self.build())
        assert (result.prompt_tokens, result.completion_tokens) == (11, 7)

    @respx.mock
    async def test_total_tokens_is_the_sum(self):
        respx.post(self.completions_url).respond(
            200, json=self.success_body(prompt_tokens=11, completion_tokens=7)
        )
        result = await self.complete(self.build())
        assert result.total_tokens == 18

    @respx.mock
    async def test_latency_is_measured(self):
        respx.post(self.completions_url).respond(200, json=self.success_body())
        result = await self.complete(self.build())
        assert result.latency_ms >= 0

    @respx.mock
    async def test_finish_reason_is_reported(self):
        respx.post(self.completions_url).respond(200, json=self.success_body())
        result = await self.complete(self.build())
        assert result.finish_reason

    # --- the 200-with-no-answer case ----------------------------------------

    @respx.mock
    async def test_missing_content_raises_rather_than_returning_empty(self):
        respx.post(self.completions_url).respond(200, json=self.no_content_body())
        with pytest.raises(ProviderResponseError):
            await self.complete(self.build())

    @respx.mock
    async def test_blank_content_raises_rather_than_returning_whitespace(self):
        respx.post(self.completions_url).respond(200, json=self.success_body(text="   \n  "))
        with pytest.raises(ProviderResponseError):
            await self.complete(self.build())

    @respx.mock
    async def test_unparseable_body_raises_response_error(self):
        respx.post(self.completions_url).respond(200, json={"unexpected": "shape"})
        with pytest.raises(ProviderResponseError):
            await self.complete(self.build())

    # --- error mapping ------------------------------------------------------

    @respx.mock
    async def test_unauthorized_maps_to_auth_error(self):
        respx.post(self.completions_url).respond(401, json=self.error_body("bad key"))
        with pytest.raises(ProviderAuthError):
            await self.complete(self.build())

    @respx.mock
    async def test_forbidden_maps_to_auth_error(self):
        respx.post(self.completions_url).respond(403, json=self.error_body("forbidden"))
        with pytest.raises(ProviderAuthError):
            await self.complete(self.build())

    @respx.mock
    async def test_payment_required_maps_to_quota_error(self):
        respx.post(self.completions_url).respond(402, json=self.error_body("pay up"))
        with pytest.raises(ProviderQuotaError):
            await self.complete(self.build())

    @respx.mock
    async def test_too_many_requests_maps_to_rate_limit_error(self):
        respx.post(self.completions_url).respond(429, json=self.error_body("slow down"))
        with pytest.raises(ProviderRateLimitError):
            await self.complete(self.build())

    @respx.mock
    async def test_retry_after_header_is_captured(self):
        respx.post(self.completions_url).respond(
            429, json=self.error_body("slow down"), headers={"retry-after": "7"}
        )
        with pytest.raises(ProviderRateLimitError) as exc:
            await self.complete(self.build())
        assert exc.value.retry_after == 7.0

    @respx.mock
    async def test_server_error_maps_to_unavailable(self):
        respx.post(self.completions_url).respond(500, json=self.error_body("oops"))
        with pytest.raises(ProviderUnavailableError):
            await self.complete(self.build())

    @respx.mock
    async def test_bad_gateway_maps_to_unavailable(self):
        respx.post(self.completions_url).respond(502, json=self.error_body("oops"))
        with pytest.raises(ProviderUnavailableError):
            await self.complete(self.build())

    @respx.mock
    async def test_connection_failure_maps_to_unavailable(self):
        respx.post(self.completions_url).mock(side_effect=httpx.ConnectError("no route"))
        with pytest.raises(ProviderUnavailableError):
            await self.complete(self.build())

    @respx.mock
    async def test_timeout_maps_to_timeout_error(self):
        respx.post(self.completions_url).mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(ProviderTimeoutError):
            await self.complete(self.build())

    # --- error metadata the router depends on -------------------------------

    @respx.mock
    async def test_errors_identify_the_failing_provider(self):
        respx.post(self.completions_url).respond(500, json=self.error_body())
        with pytest.raises(ProviderError) as exc:
            await self.complete(self.build())
        assert exc.value.provider == self.provider_cls.name

    @respx.mock
    async def test_errors_carry_the_status_code(self):
        respx.post(self.completions_url).respond(500, json=self.error_body())
        with pytest.raises(ProviderError) as exc:
            await self.complete(self.build())
        assert exc.value.status_code == 500

    @respx.mock
    async def test_rate_limit_is_retryable_but_auth_failure_is_not(self):
        respx.post(self.completions_url).respond(429, json=self.error_body())
        with pytest.raises(ProviderRateLimitError) as rate_limited:
            await self.complete(self.build())

        respx.post(self.completions_url).respond(401, json=self.error_body())
        with pytest.raises(ProviderAuthError) as rejected:
            await self.complete(self.build())

        assert rate_limited.value.retryable is True
        assert rejected.value.retryable is False

    # --- catalog listing ----------------------------------------------------

    @respx.mock
    async def test_list_models_returns_model_ids(self):
        respx.get(self.models_url).respond(200, json=self.models_body(["model-a", "model-b"]))
        assert await self.build().list_models() == ["model-a", "model-b"]

    @respx.mock
    async def test_list_models_maps_auth_failure(self):
        respx.get(self.models_url).respond(401, json=self.error_body("bad key"))
        with pytest.raises(ProviderAuthError):
            await self.build().list_models()

    @respx.mock
    async def test_list_models_maps_server_error(self):
        respx.get(self.models_url).respond(503, json=self.error_body("down"))
        with pytest.raises(ProviderUnavailableError):
            await self.build().list_models()

    # --- request shape ------------------------------------------------------

    @respx.mock
    async def test_api_key_is_sent(self):
        route = respx.post(self.completions_url).respond(200, json=self.success_body())
        await self.complete(self.build())
        sent = route.calls.last.request
        assert "test-key-not-real" in str(sent.headers) or "test-key-not-real" in str(sent.url)
