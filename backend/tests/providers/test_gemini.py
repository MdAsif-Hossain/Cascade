"""Gemini adapter against the shared contract suite, plus its dialect translation."""

from __future__ import annotations

import json
from typing import Any

import pytest
import respx

from app.core.errors import ProviderResponseError
from app.providers.base import Message, Provider
from app.providers.gemini import GeminiProvider
from tests.providers.contract import ProviderContractSuite

MODEL = "gemini-flash-lite-latest"
BASE = "https://generativelanguage.googleapis.com/v1beta"
COMPLETIONS_URL = f"{BASE}/models/{MODEL}:generateContent"


class TestGeminiContract(ProviderContractSuite):
    @property
    def provider_cls(self) -> type[Provider]:
        return GeminiProvider

    @property
    def model(self) -> str:
        return MODEL

    @property
    def completions_url(self) -> str:
        return COMPLETIONS_URL

    @property
    def models_url(self) -> str:
        return f"{BASE}/models"

    def success_body(
        self,
        text: str = "The derivative is 2x.",
        prompt_tokens: int = 11,
        completion_tokens: int = 7,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
        }

    def no_content_body(self) -> dict[str, Any]:
        """A candidate truncated before emitting text — no parts at all."""
        return {
            "candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 23, "candidatesTokenCount": 32},
        }

    def models_body(self, ids: list[str]) -> dict[str, Any]:
        return {
            "models": [
                {"name": f"models/{i}", "supportedGenerationMethods": ["generateContent"]}
                for i in ids
            ]
        }


class TestGeminiDialect:
    """Translation into Gemini's request shape — not part of the shared contract."""

    @respx.mock
    async def test_user_turns_become_contents_with_parts(self):
        route = respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "2x"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
        await GeminiProvider("k").complete([Message(role="user", content="hi")], MODEL)
        body = route.calls.last.request.read()
        payload = json.loads(body)
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]

    @respx.mock
    async def test_assistant_role_is_renamed_to_model(self):
        route = respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "2x"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
        await GeminiProvider("k").complete(
            [
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
                Message(role="user", content="bye"),
            ],
            MODEL,
        )
        payload = json.loads(route.calls.last.request.read())
        assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]

    @respx.mock
    async def test_system_message_is_lifted_out_of_contents(self):
        route = respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "2x"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
        await GeminiProvider("k").complete(
            [
                Message(role="system", content="Be terse."),
                Message(role="user", content="hi"),
            ],
            MODEL,
        )
        payload = json.loads(route.calls.last.request.read())
        assert payload["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
        assert all(c["role"] != "system" for c in payload["contents"])

    @respx.mock
    async def test_key_is_sent_as_header_not_query_param(self):
        route = respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "2x"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
        await GeminiProvider("secret-key").complete([Message(role="user", content="hi")], MODEL)
        request = route.calls.last.request
        assert request.headers["x-goog-api-key"] == "secret-key"
        assert "secret-key" not in str(request.url)

    @respx.mock
    async def test_safety_blocked_candidate_raises_rather_than_returning_empty(self):
        respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [{"finishReason": "SAFETY"}],
                "usageMetadata": {"promptTokenCount": 9},
            },
        )
        with pytest.raises(ProviderResponseError, match="SAFETY"):
            await GeminiProvider("k").complete([Message(role="user", content="hi")], MODEL)

    @respx.mock
    async def test_multipart_answers_are_joined(self):
        respx.post(COMPLETIONS_URL).respond(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "The answer "}, {"text": "is 2x."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 4},
            },
        )
        result = await GeminiProvider("k").complete([Message(role="user", content="hi")], MODEL)
        assert result.text == "The answer is 2x."

    @respx.mock
    async def test_model_prefix_is_stripped_from_catalog(self):
        respx.get(f"{BASE}/models").respond(
            200, json={"models": [{"name": "models/gemini-2.5-flash"}]}
        )
        assert await GeminiProvider("k").list_models() == ["gemini-2.5-flash"]
