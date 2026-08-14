"""The escalation loop, the classifier's degradation behaviour, and the embedder."""

from __future__ import annotations

import pytest
import respx

from app.core.config import Settings
from app.providers.embeddings import Embedder
from app.routing.classifier import FALLBACK_TIER, DifficultyClassifier, Prediction
from app.routing.router import Router
from app.routing.tiers import candidates
from app.services.ask import AnswerUnavailableError, AskService
from app.verification.verifier import Verifier

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/" "gemini-embedding-001:embedContent"
)


def gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def settings(**overrides: str) -> Settings:
    defaults = {
        "groq_api_key": "g-key",
        "google_ai_studio_api_key": "m-key",
        "openrouter_api_key": "o-key",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[arg-type]


def openai_ok(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30},
    }


def gemini_ok(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 30},
    }


class StubClassifier(DifficultyClassifier):
    """A classifier that returns a fixed tier, so loop tests do not need an artifact."""

    def __init__(self, tier: str = "T1", confidence: float = 0.9) -> None:
        self._tier = tier
        self._confidence = confidence

    async def predict(self, question: str, subject: str = "general") -> Prediction:  # type: ignore[override]
        return Prediction(self._tier, self._confidence, "model")  # type: ignore[arg-type]


def build_service(tier: str = "T1", **override: str) -> AskService:
    router = Router(settings(**override))
    return AskService(router, StubClassifier(tier), Verifier(router))


class TestHappyPath:
    @respx.mock
    async def test_an_accepted_answer_is_returned_with_its_trace(self):
        respx.post(GROQ_URL).respond(200, json=openai_ok("The derivative is 2x."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        result = await build_service("T1").ask("What is the derivative of x squared?")

        assert result.answer == "The derivative is 2x."
        assert result.escalated is False
        assert result.final_tier == "T1"
        assert [s.outcome for s in result.steps] == ["accepted"]

    @respx.mock
    async def test_cost_is_computed_against_the_top_tier_baseline(self):
        respx.post(GROQ_URL).respond(200, json=openai_ok("2x"))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        result = await build_service("T1").ask("q")

        assert result.estimated_cost_usd > 0
        assert result.baseline_cost_usd > result.estimated_cost_usd
        assert result.estimated_saving_usd > 0

    @respx.mock
    async def test_the_verifier_score_appears_in_the_trace(self):
        respx.post(GROQ_URL).respond(200, json=openai_ok("2x"))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        result = await build_service("T1").ask("q")

        assert result.steps[0].verifier_score == pytest.approx(0.95)


class TestEscalation:
    @respx.mock
    async def test_a_failed_verification_escalates_a_tier(self):
        respx.post(GROQ_URL).side_effect = [
            respx.MockResponse(200, json=openai_ok("a weak answer")),  # T1 answer
            respx.MockResponse(200, json=openai_ok("a strong answer")),  # T2 answer
        ]
        # The judge lives on gemini for both rounds.
        respx.post(gemini_url(candidates("T1")[1].model)).side_effect = [
            respx.MockResponse(200, json=gemini_ok("0.2")),
            respx.MockResponse(200, json=gemini_ok("0.95")),
        ]

        result = await build_service("T1").ask("q")

        assert result.escalated is True
        assert result.final_tier == "T2"
        assert [s.outcome for s in result.steps] == ["escalated", "accepted"]

    @respx.mock
    async def test_escalation_is_capped(self):
        """Past the cap the best answer is returned rather than looping forever."""
        respx.post(GROQ_URL).respond(200, json=openai_ok("a weak answer"))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.1"))
        respx.post(gemini_url(candidates("T3")[0].model)).respond(200, json=gemini_ok("still weak"))
        respx.post(gemini_url(candidates("T2")[2].model)).respond(200, json=gemini_ok("weak"))

        result = await build_service("T1").ask("q")

        assert len(result.steps) <= 3
        assert result.answer

    @respx.mock
    async def test_a_refusal_escalates_without_calling_a_judge(self):
        respx.post(GROQ_URL).side_effect = [
            respx.MockResponse(200, json=openai_ok("I can't help with that.")),
            respx.MockResponse(200, json=openai_ok("A proper answer.")),
        ]
        judge = respx.post(gemini_url(candidates("T1")[1].model)).respond(
            200, json=gemini_ok("0.95")
        )

        result = await build_service("T1").ask("q")

        assert result.escalated is True
        # One judge call for the T2 answer, none for the refused T1 answer.
        assert judge.call_count == 1

    @respx.mock
    async def test_a_dead_tier_escalates_rather_than_failing(self):
        for spec in candidates("T1"):
            if spec.provider == "groq":
                respx.post(GROQ_URL).respond(503, json={"error": {"message": "down"}})
            elif spec.provider == "gemini":
                respx.post(gemini_url(spec.model)).respond(503, json={"error": {"message": "x"}})
            else:
                respx.post(OPENROUTER_URL).respond(503, json={"error": {"message": "x"}})

        respx.post(gemini_url(candidates("T2")[2].model)).respond(
            200, json=gemini_ok("A T2 answer.")
        )

        result = await build_service("T1").ask("q")

        assert result.final_tier == "T2"
        assert result.steps[0].outcome == "provider_error"


class TestTotalFailure:
    @respx.mock
    async def test_every_tier_failing_raises_a_typed_error(self):
        respx.post(GROQ_URL).respond(503, json={"error": {"message": "down"}})
        respx.post(OPENROUTER_URL).respond(503, json={"error": {"message": "down"}})
        respx.route(host="generativelanguage.googleapis.com").respond(
            503, json={"error": {"message": "down"}}
        )

        with pytest.raises(AnswerUnavailableError) as exc:
            await build_service("T1").ask("q")
        assert exc.value.error_code == "answer_unavailable"


class TestClassifierDegradation:
    async def test_a_missing_artifact_falls_back_rather_than_raising(self, tmp_path):
        classifier = DifficultyClassifier(
            model_path=tmp_path / "absent.joblib", metadata_path=tmp_path / "absent.json"
        )
        assert classifier.is_loaded is False

        prediction = await classifier.predict("What is 2+2?")
        assert prediction.tier == FALLBACK_TIER
        assert prediction.is_fallback is True

    async def test_the_fallback_is_not_the_cheapest_tier(self, tmp_path):
        """Without a prediction, an over-spend is safer than a wrong answer."""
        classifier = DifficultyClassifier(
            model_path=tmp_path / "absent.joblib", metadata_path=tmp_path / "absent.json"
        )
        prediction = await classifier.predict("q")
        assert prediction.tier != "T1"


class TestEmbedder:
    @respx.mock
    async def test_a_vector_is_returned_and_cached(self):
        route = respx.post(EMBED_URL).respond(200, json={"embedding": {"values": [0.1, 0.2, 0.3]}})
        embedder = Embedder("k")

        first = await embedder.embed("What is 2+2?")
        second = await embedder.embed("What is 2+2?")

        assert first == [0.1, 0.2, 0.3]
        assert second == first
        assert route.call_count == 1, "second call should have been served from cache"

    @respx.mock
    async def test_a_failure_returns_none_rather_than_raising(self):
        """The classifier treats None as 'no prediction' and falls back."""
        respx.post(EMBED_URL).respond(503, json={"error": {"message": "down"}})
        assert await Embedder("k").embed("q") is None

    async def test_a_missing_key_returns_none(self):
        assert await Embedder("").embed("q") is None
