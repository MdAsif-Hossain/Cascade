"""Endpoint behaviour: happy paths, error shapes, caching, and persistence."""

from __future__ import annotations

import pytest
import respx
from fastapi.testclient import TestClient

from app.api.deps import build_state, set_state
from app.core.config import Settings
from app.db import session as db_session
from app.main import create_app
from app.routing.classifier import Prediction
from app.routing.tiers import candidates

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client backed by a throwaway SQLite file and a stubbed classifier.

    A real artifact is not required: the classifier degrades to a fallback tier,
    and these tests are about the HTTP layer, not about prediction quality.
    """
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        groq_api_key="g-key",
        google_ai_studio_api_key="m-key",
        openrouter_api_key="o-key",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        environment="test",
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.db.session.get_settings", lambda: settings)
    db_session.reset_engine()

    state = build_state(settings)

    async def fixed_prediction(question: str, subject: str = "general") -> Prediction:
        return Prediction("T1", 0.88, "model")

    state.classifier.predict = fixed_prediction  # type: ignore[method-assign]
    # Polling would make live network calls during app startup.
    state.poller._providers = {}  # noqa: SLF001

    set_state(state)
    with TestClient(create_app()) as test_client:
        yield test_client
    set_state(None)
    db_session.reset_engine()


class TestHealth:
    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_reports_classifier_state(self, client):
        """A degraded classifier must be visible, not hidden behind a green light."""
        assert "classifier_loaded" in client.get("/health").json()

    def test_health_makes_no_provider_calls(self, client):
        """Otherwise a rate-limited provider could get the service restarted."""
        with respx.mock:
            assert client.get("/health").status_code == 200


class TestAsk:
    @respx.mock
    def test_a_question_is_answered_with_its_trace(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("Ice floats because it is less dense."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        response = client.post("/api/v1/ask", json={"question": "Why does ice float?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"].startswith("Ice floats")
        assert body["trace"][0]["outcome"] == "accepted"
        assert body["cached"] is False
        assert body["request_id"]

    @respx.mock
    def test_cost_fields_are_present_and_labelled_counterfactual(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        body = client.post("/api/v1/ask", json={"question": "Why does ice float?"}).json()

        assert body["estimated_cost_usd"] > 0
        assert body["baseline_cost_usd"] >= body["estimated_cost_usd"]

    @respx.mock
    def test_a_repeated_question_is_served_from_cache(self, client):
        route = respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        payload = {"question": "Why does ice float?"}
        first = client.post("/api/v1/ask", json=payload).json()
        second = client.post("/api/v1/ask", json=payload).json()

        assert first["cached"] is False
        assert second["cached"] is True
        assert route.call_count == 1

    @respx.mock
    def test_case_and_punctuation_do_not_defeat_the_cache(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))

        client.post("/api/v1/ask", json={"question": "Why does ice float?"})
        second = client.post("/api/v1/ask", json={"question": "  why does ICE float  "})

        assert second.json()["cached"] is True

    def test_a_short_question_is_rejected_with_the_standard_error_shape(self, client):
        response = client.post("/api/v1/ask", json={"question": "hi"})
        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"error_code", "message", "request_id"}
        assert body["error_code"] == "validation_error"

    def test_an_invalid_subject_is_rejected(self, client):
        response = client.post(
            "/api/v1/ask", json={"question": "A valid question?", "subject": "astrology"}
        )
        assert response.status_code == 422

    @respx.mock
    def test_all_providers_failing_returns_503_not_500(self, client):
        """The service is fine; an upstream is not. The client should retry."""
        respx.route().respond(503, json={"error": {"message": "down"}})

        response = client.post("/api/v1/ask", json={"question": "Why does ice float?"})

        assert response.status_code == 503
        assert response.json()["error_code"] == "answer_unavailable"

    @respx.mock
    def test_no_stack_trace_ever_reaches_the_client(self, client):
        respx.route().respond(503, json={"error": {"message": "down"}})
        body = client.post("/api/v1/ask", json={"question": "Why does ice float?"}).text
        assert "Traceback" not in body
        assert 'File "' not in body


class TestHistoryAndFeedback:
    @respx.mock
    def test_an_answered_question_appears_in_history(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))
        client.post("/api/v1/ask", json={"question": "Why does ice float?"})

        page = client.get("/api/v1/requests").json()

        assert page["total"] == 1
        assert page["items"][0]["question"] == "Why does ice float?"

    @respx.mock
    def test_a_single_trace_can_be_fetched(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))
        request_id = client.post("/api/v1/ask", json={"question": "Why does ice float?"}).json()[
            "request_id"
        ]

        detail = client.get(f"/api/v1/requests/{request_id}").json()

        assert detail["id"] == request_id
        assert detail["trace"]

    def test_an_unknown_trace_returns_404(self, client):
        response = client.get("/api/v1/requests/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error_code"] == "http_404"

    @respx.mock
    def test_feedback_is_recorded(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))
        request_id = client.post("/api/v1/ask", json={"question": "Why does ice float?"}).json()[
            "request_id"
        ]

        response = client.post("/api/v1/feedback", json={"request_id": request_id, "helpful": True})

        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_feedback_on_an_unknown_request_is_rejected(self, client):
        response = client.post("/api/v1/feedback", json={"request_id": "nope", "helpful": True})
        assert response.status_code == 404


class TestMetricsAndCatalog:
    def test_metrics_work_with_no_data(self, client):
        """An empty dashboard must render, not divide by zero."""
        body = client.get("/api/v1/metrics").json()
        assert body["total_requests"] == 0
        assert body["escalation_rate"] == 0.0
        assert body["estimated_saving_percent"] == 0.0

    def test_metrics_always_state_the_cost_basis(self, client):
        """Every cost surface must say the figures are estimated (CLAUDE.md section 2)."""
        basis = client.get("/api/v1/metrics").json()["cost_basis"]
        assert "free" in basis.lower()
        assert "no money" in basis.lower()

    @respx.mock
    def test_metrics_aggregate_answered_questions(self, client):
        respx.post(GROQ_URL).respond(200, json=openai_ok("An answer."))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))
        client.post("/api/v1/ask", json={"question": "Why does ice float?"})

        body = client.get("/api/v1/metrics").json()

        assert body["total_requests"] == 1
        assert body["tier_distribution"]["T1"] == 1
        assert body["estimated_saving_usd"] > 0

    def test_catalog_lists_every_tier_model(self, client):
        models = client.get("/api/v1/catalog").json()["models"]
        assert {m["tier"] for m in models} == {"T1", "T2", "T3"}


class TestDocs:
    def test_openapi_schema_is_generated(self, client):
        """/docs is the API documentation, so it has to actually build."""
        schema = client.get("/openapi.json").json()
        assert "/api/v1/ask" in schema["paths"]
        assert "/api/v1/metrics" in schema["paths"]
