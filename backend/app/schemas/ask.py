"""Request and response contracts for the ask endpoint.

Defined before the logic that fills them (CLAUDE.md §6), and doubling as the
public documentation: FastAPI renders these into /docs, so a field description
here is what an integrator reads.

Cost fields are named ``estimated_*`` throughout and described as counterfactual
in every place they appear. Every model call runs on a free tier — no money is
ever spent — and a field called ``cost_usd`` would invite exactly the wrong
reading (§2).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["T1", "T2", "T3"]
Subject = Literal["math", "science", "history", "language", "general"]
Level = Literal["school", "undergrad"]
Outcome = Literal["accepted", "escalated", "provider_error"]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The student's question.",
        examples=["Why does ice float on water?"],
    )
    subject: Subject = Field(
        default="general", description="Subject hint; feeds the difficulty classifier."
    )
    level: Level = Field(default="school", description="Study level the answer should suit.")


class RoutingStep(BaseModel):
    """One attempt in the routing trace.

    Shown to the student, not just logged. A failed or escalated step stays in
    the trace so the journey is visible rather than tidied away.
    """

    tier: Tier
    provider: str = Field(description="Provider that served this step, or '-' if none could.")
    model: str
    latency_ms: int = Field(ge=0)
    verifier_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Judge score, or null if no judge ran."
    )
    outcome: Outcome
    detail: str | None = Field(
        default=None, description="Why this step was accepted, escalated, or failed."
    )


class AskResponse(BaseModel):
    answer: str
    trace: list[RoutingStep] = Field(description="Every tier attempted, in order.")
    predicted_tier: Tier = Field(description="Tier the classifier chose before answering.")
    final_tier: Tier = Field(description="Tier that produced the returned answer.")
    escalated: bool
    prediction_confidence: float = Field(
        ge=0.0, le=1.0, description="Classifier confidence in predicted_tier."
    )
    prediction_source: str = Field(
        description="'model' when the classifier ran; otherwise why it could not."
    )
    estimated_cost_usd: float = Field(
        ge=0.0,
        description=(
            "Counterfactual cost at published list prices. No money was spent — "
            "every call runs on a free tier."
        ),
    )
    baseline_cost_usd: float = Field(
        ge=0.0,
        description="Counterfactual cost had the question gone straight to the top tier.",
    )
    total_latency_ms: int = Field(ge=0)
    cached: bool = Field(description="True when served from the answer cache.")
    request_id: str
    warnings: list[str] = Field(
        default_factory=list,
        description="Notes about degraded operation, e.g. no cross-provider judge available.",
    )


class ErrorResponse(BaseModel):
    """The only error shape the API returns (§10). Never a bare 500, never a stack trace."""

    error_code: str
    message: str
    request_id: str
