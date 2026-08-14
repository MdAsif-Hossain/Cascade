"""GET /api/v1/metrics, /requests, /catalog, and POST /api/v1/feedback.

Everything the dashboard and history views read. Aggregation runs in SQL where it
can, because the free Postgres tier is a better place to count rows than a
512 MB container is.
"""

from __future__ import annotations

import statistics

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from app.api.deps import AppState, get_state
from app.db.models import FeedbackRecord, RequestRecord
from app.db.session import session_scope
from app.routing.tiers import TIERS
from app.schemas.ask import RoutingStep
from app.schemas.common import (
    CatalogResponse,
    DriftEventOut,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse,
    ModelHealth,
    ProviderHealthOut,
    RequestDetail,
    RequestPage,
    RequestSummary,
    TierStats,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["insight"])


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


@router.get("/requests", response_model=RequestPage, summary="Past questions")
def list_requests(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    tier: str | None = Query(default=None, description="Filter by final tier."),
    escalated: bool | None = Query(default=None),
) -> RequestPage:
    with session_scope() as session:
        query = select(RequestRecord)
        count_query = select(func.count()).select_from(RequestRecord)

        if tier:
            query = query.where(RequestRecord.final_tier == tier)
            count_query = count_query.where(RequestRecord.final_tier == tier)
        if escalated is not None:
            query = query.where(RequestRecord.escalated == escalated)
            count_query = count_query.where(RequestRecord.escalated == escalated)

        total = session.scalar(count_query) or 0
        rows = session.scalars(
            query.order_by(RequestRecord.created_at.desc()).limit(limit).offset(offset)
        ).all()

        return RequestPage(
            items=[
                RequestSummary(
                    id=r.id,
                    created_at=r.created_at,
                    question=r.question,
                    subject=r.subject,
                    predicted_tier=r.predicted_tier,  # type: ignore[arg-type]
                    final_tier=r.final_tier,  # type: ignore[arg-type]
                    escalated=r.escalated,
                    estimated_cost_usd=r.estimated_cost_usd,
                    baseline_cost_usd=r.baseline_cost_usd,
                    total_latency_ms=r.total_latency_ms,
                    cached=r.cached,
                )
                for r in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/requests/{request_id}", response_model=RequestDetail, summary="One trace")
def get_request(request_id: str) -> RequestDetail:
    with session_scope() as session:
        record = session.get(RequestRecord, request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="request not found")
        return RequestDetail(
            id=record.id,
            created_at=record.created_at,
            question=record.question,
            subject=record.subject,
            answer=record.answer,
            predicted_tier=record.predicted_tier,  # type: ignore[arg-type]
            final_tier=record.final_tier,  # type: ignore[arg-type]
            escalated=record.escalated,
            estimated_cost_usd=record.estimated_cost_usd,
            baseline_cost_usd=record.baseline_cost_usd,
            total_latency_ms=record.total_latency_ms,
            cached=record.cached,
            prediction_confidence=record.prediction_confidence,
            prediction_source=record.prediction_source,
            trace=[RoutingStep(**step) for step in record.trace],
        )


@router.post("/feedback", response_model=FeedbackResponse, summary="Rate an answer")
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    with session_scope() as session:
        if session.get(RequestRecord, payload.request_id) is None:
            raise HTTPException(status_code=404, detail="request not found")
        session.add(
            FeedbackRecord(
                request_id=payload.request_id,
                helpful=payload.helpful,
                comment=payload.comment,
            )
        )
    logger.info("feedback_recorded", request_id=payload.request_id, helpful=payload.helpful)
    return FeedbackResponse(recorded=True, request_id=payload.request_id)


@router.get("/metrics", response_model=MetricsResponse, summary="Dashboard aggregates")
def get_metrics(state: AppState = Depends(get_state)) -> MetricsResponse:
    with session_scope() as session:
        rows = session.scalars(select(RequestRecord)).all()
        positive = (
            session.scalar(
                select(func.count())
                .select_from(FeedbackRecord)
                .where(FeedbackRecord.helpful.is_(True))
            )
            or 0
        )
        negative = (
            session.scalar(
                select(func.count())
                .select_from(FeedbackRecord)
                .where(FeedbackRecord.helpful.is_(False))
            )
            or 0
        )

    total = len(rows)
    estimated = sum(r.estimated_cost_usd for r in rows)
    baseline = sum(r.baseline_cost_usd for r in rows)
    saving = max(0.0, baseline - estimated)

    tier_stats: list[TierStats] = []
    distribution: dict[str, int] = {}
    for tier in TIERS:
        in_tier = [r for r in rows if r.final_tier == tier]
        distribution[tier] = len(in_tier)
        latencies = [r.total_latency_ms for r in in_tier]
        tier_stats.append(
            TierStats(
                tier=tier,
                count=len(in_tier),
                escalation_rate=(
                    sum(r.escalated for r in in_tier) / len(in_tier) if in_tier else 0.0
                ),
                p50_latency_ms=int(statistics.median(latencies)) if latencies else 0,
                p95_latency_ms=_percentile(latencies, 0.95),
            )
        )

    return MetricsResponse(
        total_requests=total,
        escalation_rate=(sum(r.escalated for r in rows) / total) if total else 0.0,
        estimated_total_cost_usd=round(estimated, 8),
        estimated_baseline_cost_usd=round(baseline, 8),
        estimated_saving_usd=round(saving, 8),
        estimated_saving_percent=round(100 * saving / baseline, 2) if baseline else 0.0,
        cache_hit_rate=state.cache.hit_rate,
        tier_distribution=distribution,
        tier_stats=tier_stats,
        feedback_positive=positive,
        feedback_negative=negative,
    )


@router.get("/catalog", response_model=CatalogResponse, summary="Provider and model health")
def get_catalog(state: AppState = Depends(get_state)) -> CatalogResponse:
    health = state.poller.health
    breaker = state.router.breaker

    models: list[ModelHealth] = []
    for tier, specs in TIERS.items():
        for spec in specs:
            known = health.get(spec.provider)
            catalog = known.models if known is not None else []
            models.append(
                ModelHealth(
                    provider=spec.provider,
                    model=spec.model,
                    tier=tier,
                    # Before the first poll the catalog is empty and nothing is
                    # known. Reported as advertised rather than implying every
                    # model has vanished the moment the service starts.
                    advertised=spec.model in catalog if catalog else True,
                    circuit_open=breaker.is_open(f"{spec.provider}/{spec.model}"),
                )
            )

    return CatalogResponse(
        providers=[
            ProviderHealthOut(
                provider=h.provider,
                reachable=h.reachable,
                model_count=len(h.models),
                last_checked=h.last_checked,
                last_error=h.last_error,
            )
            for h in health.values()
        ],
        models=models,
        recent_drift=[
            DriftEventOut(provider=d.provider, model=d.model, kind=d.kind, detail=d.detail)
            for d in reversed(state.poller.recent_drifts[-20:])
        ],
    )
