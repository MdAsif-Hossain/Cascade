"""Groq adapter against the shared contract suite."""

from __future__ import annotations

from typing import Any

from app.providers.base import Provider
from app.providers.groq import GroqProvider
from tests.providers.contract import ProviderContractSuite


class TestGroqContract(ProviderContractSuite):
    @property
    def provider_cls(self) -> type[Provider]:
        return GroqProvider

    @property
    def model(self) -> str:
        return "llama-3.1-8b-instant"

    @property
    def completions_url(self) -> str:
        return "https://api.groq.com/openai/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return "https://api.groq.com/openai/v1/models"

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
            },
        }

    def no_content_body(self) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"role": "assistant", "content": None}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 23, "completion_tokens": 32, "total_tokens": 55},
        }

    def models_body(self, ids: list[str]) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
