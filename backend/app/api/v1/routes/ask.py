"""POST /api/v1/ask — the endpoint the whole system exists to serve.

Validates, checks the cache, delegates to the service, persists the trace, and
serialises. No routing or verification logic lives here (§13).
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Request

from app.api.deps import AppState, get_state
from app.core.cache import cache_key
from app.core.errors import CascadeError
from app.db.models import RequestRecord
from app.db.session import session_scope
from app.schemas.ask import AskRequest, AskResponse, RoutingStep
from app.services.ask import AskResult

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["ask"])


def _to_response(result: AskResult, request_id: str, *, cached: bool) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        trace=[
            RoutingStep(
                tier=step.tier,
                provider=step.provider,
                model=step.model,
                latency_ms=step.latency_ms,
                verifier_score=step.verifier_score,
                outcome=step.outcome,  # type: ignore[arg-type]
                detail=step.detail,
            )
            for step in result.steps
        ],
        predicted_tier=result.predicted_tier,
        final_tier=result.final_tier,
        escalated=result.escalated,
        prediction_confidence=result.prediction_confidence,
        prediction_source=result.prediction_source,
        estimated_cost_usd=result.estimated_cost_usd,
        baseline_cost_usd=result.baseline_cost_usd,
        total_latency_ms=result.total_latency_ms,
        cached=cached,
        request_id=request_id,
        warnings=result.warnings,
    )


def _persist(record_id: str, payload: AskRequest, result: AskResult, cached: bool) -> None:
    """Write the trace.

    A database failure must not lose the student their answer, which they already
    have — so this logs and returns rather than raising.
    """
    try:
        with session_scope() as session:
            session.add(
                RequestRecord(
                    id=record_id,
                    question=payload.question,
                    subject=payload.subject,
                    level=payload.level,
                    answer=result.answer,
                    predicted_tier=result.predicted_tier,
                    final_tier=result.final_tier,
                    escalated=result.escalated,
                    prediction_confidence=result.prediction_confidence,
                    prediction_source=result.prediction_source,
                    trace=[
                        {
                            "tier": s.tier,
                            "provider": s.provider,
                            "model": s.model,
                            "latency_ms": s.latency_ms,
                            "verifier_score": s.verifier_score,
                            "outcome": s.outcome,
                            "detail": s.detail,
                        }
                        for s in result.steps
                    ],
                    estimated_cost_usd=result.estimated_cost_usd,
                    baseline_cost_usd=result.baseline_cost_usd,
                    total_latency_ms=result.total_latency_ms,
                    cached=cached,
                )
            )
    except Exception as exc:  # noqa: BLE001 - persistence is not worth failing a served answer
        logger.error("trace_persist_failed", request_id=record_id, error=str(exc))


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    description=(
        "Routes the question to the cheapest model likely to answer it correctly, "
        "verifies the answer with a model from a different provider, and escalates "
        "if the check fails. The full routing trace is returned alongside the answer."
    ),
)
async def ask(
    payload: AskRequest,
    request: Request,
    state: AppState = Depends(get_state),
) -> AskResponse:
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    log = logger.bind(request_id=request_id)

    key = cache_key(payload.question, payload.subject, payload.level)
    hit = state.cache.get(key)
    if hit is not None:
        log.info("cache_hit")
        # Not re-persisted: a cache hit is the same answer already on record, and
        # writing it again would double-count it in the cost metrics.
        return _to_response(hit, request_id, cached=True)

    try:
        result = await state.ask_service.ask(
            payload.question, subject=payload.subject, request_id=request_id
        )
    except CascadeError:
        raise  # handled centrally so every error has the same shape

    state.cache.put(key, result)
    _persist(request_id, payload, result, cached=False)
    return _to_response(result, request_id, cached=False)
