"""Tier definitions: capability bands, and the pricing used for counterfactual cost.

A tier is a capability band, not a provider. Each band lists candidate models
across several providers so that when one provider deletes a model or starts
throttling — which free tiers do without notice — the tier still resolves.

## About the prices

Every model Cascade calls is used on a free tier, so **no money is ever spent**.
The prices below are the published list prices of the same models on paid
infrastructure, and they exist solely to answer the counterfactual question the
project is built around: *what would this have cost if it had been paid for, and
how much of that did routing avoid?*

Every surface that displays a figure derived from these numbers must label it as
estimated (CLAUDE.md section 2).

Prices were retrieved on 2026-08-14 from OpenRouter's public ``/api/v1/models``
endpoint, which publishes machine-readable per-token pricing for each model. They
are recorded here rather than fetched at runtime so that a cost figure computed
today can be reproduced tomorrow, and so a provider silently repricing a model
cannot retroactively change results already reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["T1", "T2", "T3"]

TIER_ORDER: tuple[Tier, ...] = ("T1", "T2", "T3")

PRICING_RETRIEVED = "2026-08-14"
PRICING_SOURCE = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One candidate model within a tier, with the published price of its paid twin.

    ``price_reference`` names the model whose published price is being used. For a
    model served free by one provider, the reference is the same model sold by
    another — the honest comparison, and the one a reader can check.
    """

    provider: str
    model: str
    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float
    price_reference: str

    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Counterfactual cost of this call at published list prices.

        Never a real charge — see the module docstring.
        """
        return (
            prompt_tokens * self.prompt_usd_per_mtok
            + completion_tokens * self.completion_usd_per_mtok
        ) / 1_000_000


# T1 — small fast models. Factual recall, definitions, simple arithmetic.
_T1: tuple[ModelSpec, ...] = (
    ModelSpec("groq", "llama-3.1-8b-instant", 0.05, 0.08, "meta-llama/llama-3.1-8b-instruct"),
    ModelSpec("openrouter", "openai/gpt-oss-20b:free", 0.03, 0.13, "openai/gpt-oss-20b"),
    ModelSpec("gemini", "gemini-flash-lite-latest", 0.10, 0.40, "google/gemini-2.5-flash-lite"),
)

# T2 — mid models. Multi-step reasoning, explanation, short derivations.
_T2: tuple[ModelSpec, ...] = (
    ModelSpec("groq", "llama-3.3-70b-versatile", 0.10, 0.32, "meta-llama/llama-3.3-70b-instruct"),
    ModelSpec("groq", "openai/gpt-oss-120b", 0.03, 0.17, "openai/gpt-oss-120b"),
    ModelSpec("gemini", "gemini-2.5-flash", 0.30, 2.50, "google/gemini-2.5-flash"),
)

# T3 — strongest free models available. Hard multi-step maths, subtle reasoning.
_T3: tuple[ModelSpec, ...] = (
    ModelSpec("gemini", "gemini-2.5-pro", 1.25, 10.00, "google/gemini-2.5-pro"),
    ModelSpec(
        "openrouter",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        0.60,
        3.60,
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
)

TIERS: dict[Tier, tuple[ModelSpec, ...]] = {"T1": _T1, "T2": _T2, "T3": _T3}


def candidates(tier: Tier) -> tuple[ModelSpec, ...]:
    """Candidate models for a tier, in preference order.

    Ordering is deliberate: measured latency, not price. During the Phase 0 gate
    Groq answered in 126 ms, Gemini in 908 ms and OpenRouter's free tier in
    27.5 s, so the fastest healthy provider is tried first and the slow one is
    kept as the fallback that stops a tier from failing outright.
    """
    return TIERS[tier]


def next_tier(tier: Tier) -> Tier | None:
    """The tier to escalate into, or None when already at the top."""
    index = TIER_ORDER.index(tier)
    if index + 1 >= len(TIER_ORDER):
        return None
    return TIER_ORDER[index + 1]


def baseline_spec() -> ModelSpec:
    """The model an always-use-the-best-model system would have called.

    Cascade's headline claim is cost reduction against this baseline, so it has
    to be one specific documented model rather than a tier-wide average.
    """
    return _T3[0]


def baseline_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Counterfactual cost had the question gone straight to the strongest tier."""
    return baseline_spec().estimated_cost_usd(prompt_tokens, completion_tokens)
