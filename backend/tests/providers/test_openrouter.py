"""OpenRouter adapter against the shared contract suite."""

from __future__ import annotations

from typing import Any

import respx

from app.providers.base import Message, Provider
from app.providers.openrouter import OpenRouterProvider
from tests.providers.contract import ProviderContractSuite


class TestOpenRouterContract(ProviderContractSuite):
    @property
    def provider_cls(self) -> type[Provider]:
        return OpenRouterProvider

    @property
    def model(self) -> str:
        return "openai/gpt-oss-20b:free"

    @property
    def completions_url(self) -> str:
        return "https://openrouter.ai/api/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return "https://openrouter.ai/api/v1/models"

    def success_body(
        self,
        text: str = "The derivative is 2x.",
        prompt_tokens: int = 11,
        completion_tokens: int = 7,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": 0,
            },
        }

    def no_content_body(self) -> dict[str, Any]:
        """The exact shape a reasoning model returned live during the Phase 0 gate."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "The user is asking a very simple math question",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 23,
                "completion_tokens": 32,
                "total_tokens": 55,
                "cost": 0,
                "completion_tokens_details": {"reasoning_tokens": 32},
            },
        }

    def models_body(self, ids: list[str]) -> dict[str, Any]:
        return {"data": [{"id": i, "name": i} for i in ids]}


class TestOpenRouterAttribution:
    """OpenRouter-specific behaviour that is not part of the shared contract."""

    @respx.mock
    async def test_attribution_headers_are_sent(self):
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").respond(
            200,
            json={
                "choices": [{"message": {"content": "2x"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        provider = OpenRouterProvider("test-key-not-real")
        await provider.complete([Message(role="user", content="hi")], "openai/gpt-oss-20b:free")
        headers = route.calls.last.request.headers
        assert headers["x-title"] == "Cascade"
        assert "github.com" in headers["http-referer"]
