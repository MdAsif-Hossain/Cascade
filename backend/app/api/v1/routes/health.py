"""GET /health — liveness for the keep-alive ping.

Deliberately cheap and dependency-free: it makes no provider or database call, so
a rate-limited provider can never make the service look dead and get it
restarted. It reports degraded subsystems as fields rather than as a failure
status, because Cascade answers questions with or without a loaded classifier.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AppState, get_state
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness")
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    return HealthResponse(
        environment=state.settings.environment,
        classifier_loaded=state.classifier.is_loaded,
        providers_configured=state.settings.configured_providers(),
    )
