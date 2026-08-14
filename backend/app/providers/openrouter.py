"""OpenRouter adapter.

Verified during the Phase 0 gate: 411 models, 15 of them ``:free``, key reports
``is_free_tier: true``.

OpenRouter aggregates upstream providers, so a single ``:free`` model can be
rate-limited while its neighbours answer normally — observed live on
``google/gemma-4-31b-it:free`` in the same second three other free models
succeeded. Model health therefore has to be tracked per model, not per provider.
"""

from __future__ import annotations

from typing import ClassVar

from app.providers._openai_compat import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider):
    name: ClassVar[str] = "openrouter"
    base_url: ClassVar[str] = "https://openrouter.ai/api/v1"

    # OpenRouter attributes traffic to an app via these headers and surfaces the
    # title in its dashboard. Sending them keeps our free-tier usage identifiable
    # rather than anonymous, which is what their rate limiting keys off.
    extra_headers: ClassVar[dict[str, str]] = {
        "HTTP-Referer": "https://github.com/MdAsif-Hossain/Cascade",
        "X-Title": "Cascade",
    }
