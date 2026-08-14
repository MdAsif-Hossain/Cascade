"""Groq adapter.

Verified during the Phase 0 gate: 15 models, 122 ms round trip on
``llama-3.1-8b-instant``. Groq's binding free-tier constraint is tokens per
minute (6,000), not requests per day (14,400), which matters when pacing the
Phase 2 labeling run.
"""

from __future__ import annotations

from typing import ClassVar

from app.providers._openai_compat import OpenAICompatProvider


class GroqProvider(OpenAICompatProvider):
    name: ClassVar[str] = "groq"
    base_url: ClassVar[str] = "https://api.groq.com/openai/v1"
