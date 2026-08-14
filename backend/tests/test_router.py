"""Router tier resolution, failover, retry, and circuit-breaker integration."""

from __future__ import annotations

import pytest
import respx

from app.core.config import Settings
from app.core.errors import ProviderAuthError, ProviderRateLimitError
from app.providers.base import Message
from app.routing.breaker import FAILURE_THRESHOLD, CircuitBreaker
from app.routing.router import NoProviderAvailableError, Router
from app.routing.tiers import candidates

QUESTION = [Message(role="user", content="What is 2+2?")]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def settings(**overrides: str) -> Settings:
    defaults = {
        "groq_api_key": "g-key",
        "google_ai_studio_api_key": "m-key",
        "openrouter_api_key": "o-key",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[arg-type]


def openai_ok(text: str = "4") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def gemini_ok(text: str = "4") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
    }


class TestHappyPath:
    @respx.mock
    async def test_it_uses_the_first_candidate_in_the_tier(self):
        respx.post(GROQ_URL).respond(200, json=openai_ok())
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        first = candidates("T1")[0]
        assert result.spec.provider == first.provider
        assert result.completion.text == "4"

    @respx.mock
    async def test_a_successful_call_records_one_accepted_step(self):
        respx.post(GROQ_URL).respond(200, json=openai_ok())
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert [s.outcome for s in result.steps] == ["accepted"]

    @respx.mock
    async def test_it_does_not_call_later_candidates_when_the_first_works(self):
        groq = respx.post(GROQ_URL).respond(200, json=openai_ok())
        gemini = respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert groq.called
        assert not gemini.called


class TestFailover:
    @respx.mock
    async def test_it_fails_over_to_the_next_provider(self):
        """A dead first choice must not fail the request when the tier has depth."""
        respx.post(GROQ_URL).respond(500, json={"error": {"message": "down"}})
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert result.spec.provider == "gemini"
        assert result.completion.text == "4"

    @respx.mock
    async def test_the_failed_attempt_is_kept_in_the_trace(self):
        """The student sees why routing moved on, so failures cannot be silent."""
        respx.post(GROQ_URL).respond(500, json={"error": {"message": "down"}})
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert [s.outcome for s in result.steps] == ["provider_error", "accepted"]
        assert result.steps[0].error is not None

    @respx.mock
    async def test_it_raises_when_every_candidate_fails(self):
        for spec in candidates("T1"):
            if spec.provider == "groq":
                respx.post(GROQ_URL).respond(500, json={"error": {"message": "down"}})
            elif spec.provider == "gemini":
                respx.post(gemini_url(spec.model)).respond(500, json={"error": {"message": "x"}})
            else:
                respx.post(OPENROUTER_URL).respond(500, json={"error": {"message": "x"}})
        with pytest.raises(NoProviderAvailableError):
            await Router(settings()).complete_in_tier(QUESTION, "T1")


class TestRetry:
    @respx.mock
    async def test_a_retryable_failure_is_retried_on_the_same_model(self):
        route = respx.post(GROQ_URL)
        route.side_effect = [
            respx.MockResponse(429, json={"error": {"message": "slow down"}}),
            respx.MockResponse(200, json=openai_ok()),
        ]
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert result.spec.provider == "groq"
        assert route.call_count == 2

    @respx.mock
    async def test_a_non_retryable_failure_moves_straight_on(self):
        """Retrying a rejected API key wastes time and can get an account flagged."""
        route = respx.post(GROQ_URL).respond(401, json={"error": {"message": "bad key"}})
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        result = await Router(settings()).complete_in_tier(QUESTION, "T1")
        assert route.call_count == 1
        assert result.spec.provider == "gemini"

    def test_provider_retry_after_beats_our_backoff(self):
        exc = ProviderRateLimitError("groq", "slow", retry_after=2.0)
        assert Router._backoff_for(exc, attempt=1) == 2.0

    def test_backoff_is_capped_so_a_request_cannot_hang(self):
        exc = ProviderRateLimitError("groq", "slow", retry_after=9999.0)
        assert Router._backoff_for(exc, attempt=1) <= 5.0

    def test_backoff_falls_back_to_exponential_without_a_hint(self):
        exc = ProviderAuthError("groq", "nope")
        assert Router._backoff_for(exc, attempt=2) > Router._backoff_for(exc, attempt=1)


class TestBreakerIntegration:
    @respx.mock
    async def test_repeated_failures_open_the_circuit(self):
        respx.post(GROQ_URL).respond(500, json={"error": {"message": "down"}})
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        router = Router(settings())
        for _ in range(FAILURE_THRESHOLD):
            await router.complete_in_tier(QUESTION, "T1")
        assert router.breaker.is_open("groq/llama-3.1-8b-instant") is True

    @respx.mock
    async def test_an_open_circuit_is_skipped_without_a_call(self):
        breaker = CircuitBreaker()
        target = f"{candidates('T1')[0].provider}/{candidates('T1')[0].model}"
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure(target)

        groq = respx.post(GROQ_URL).respond(200, json=openai_ok())
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())

        result = await Router(settings(), breaker=breaker).complete_in_tier(QUESTION, "T1")
        assert not groq.called
        assert result.spec.provider == "gemini"


class TestProviderSelectionRules:
    @respx.mock
    async def test_providers_without_a_key_are_skipped(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        result = await Router(settings(groq_api_key="")).complete_in_tier(QUESTION, "T1")
        assert result.spec.provider == "gemini"

    @respx.mock
    async def test_excluded_providers_are_never_used(self):
        """Backs the verifier rule that a judge cannot share the answerer's provider."""
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok())
        result = await Router(settings()).complete_in_tier(
            QUESTION, "T1", exclude_providers=frozenset({"groq"})
        )
        assert result.spec.provider != "groq"

    @respx.mock
    async def test_no_configured_provider_raises_rather_than_hanging(self):
        router = Router(
            settings(groq_api_key="", google_ai_studio_api_key="", openrouter_api_key="")
        )
        with pytest.raises(NoProviderAvailableError, match="no API key"):
            await router.complete_in_tier(QUESTION, "T1")
