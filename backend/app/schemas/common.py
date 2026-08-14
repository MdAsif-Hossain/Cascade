"""Response contracts for history, feedback, metrics, catalog, and health."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.ask import RoutingStep, Tier


class HealthResponse(BaseModel):
    """Liveness, used by the keep-alive ping that stops Render's free tier sleeping."""

    status: Literal["ok"] = "ok"
    environment: str
    classifier_loaded: bool = Field(
        description="False means routing is falling back to a fixed tier."
    )
    providers_configured: list[str]


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, description="The request being rated.")
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    recorded: bool
    request_id: str


class RequestSummary(BaseModel):
    """One row of history."""

    id: str
    created_at: datetime
    question: str
    subject: str
    predicted_tier: Tier
    final_tier: Tier
    escalated: bool
    estimated_cost_usd: float
    baseline_cost_usd: float
    total_latency_ms: int
    cached: bool


class RequestDetail(RequestSummary):
    answer: str
    trace: list[RoutingStep]
    prediction_confidence: float
    prediction_source: str


class RequestPage(BaseModel):
    items: list[RequestSummary]
    total: int
    limit: int
    offset: int


class TierStats(BaseModel):
    tier: Tier
    count: int
    escalation_rate: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: int
    p95_latency_ms: int


class MetricsResponse(BaseModel):
    """Aggregates for the dashboard.

    Cost fields are counterfactual throughout: every call ran on a free tier, so
    these describe what the same work would have cost at published list prices,
    never money that changed hands.
    """

    total_requests: int
    escalation_rate: float = Field(ge=0.0, le=1.0)
    estimated_total_cost_usd: float
    estimated_baseline_cost_usd: float
    estimated_saving_usd: float
    estimated_saving_percent: float
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    tier_distribution: dict[str, int]
    tier_stats: list[TierStats]
    feedback_positive: int
    feedback_negative: int
    cost_basis: str = Field(
        default=(
            "Estimated from published per-token list prices. All calls ran on free "
            "tiers; no money was spent."
        ),
        description="Always shown alongside any cost figure.",
    )


class ModelHealth(BaseModel):
    provider: str
    model: str
    tier: Tier
    advertised: bool = Field(description="Present in the provider's current catalog.")
    circuit_open: bool = Field(description="Temporarily skipped after repeated failures.")


class DriftEventOut(BaseModel):
    provider: str
    model: str
    kind: str
    detail: str


class ProviderHealthOut(BaseModel):
    provider: str
    reachable: bool
    model_count: int
    last_checked: datetime | None
    last_error: str | None


class CatalogResponse(BaseModel):
    providers: list[ProviderHealthOut]
    models: list[ModelHealth]
    recent_drift: list[DriftEventOut]
