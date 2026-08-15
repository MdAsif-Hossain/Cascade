"""FastAPI application entrypoint.

Wires routes, error handling, CORS, and the background catalog poller.

Every error leaves here in one shape — ``{error_code, message, request_id}`` —
and no stack trace ever reaches a client (§10). The handlers below are the only
place that decides HTTP status, so a new error type gets consistent treatment by
inheriting from ``CascadeError`` rather than by remembering to catch it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import get_state, set_state
from app.api.v1.routes import ask as ask_routes
from app.api.v1.routes import health as health_routes
from app.api.v1.routes import metrics as metrics_routes
from app.core.config import get_settings
from app.core.errors import CascadeError, ProviderError
from app.core.logging import configure_logging
from app.db.session import create_tables
from app.services.ask import AnswerUnavailableError

logger = structlog.get_logger(__name__)

API_PREFIX = "/api/v1"

# Status codes by error class. Provider failures are 503, not 500: the service is
# working, an upstream dependency is not, and a client should retry rather than
# treat the request as malformed.
_STATUS_BY_ERROR: dict[type[CascadeError], int] = {
    AnswerUnavailableError: 503,
    ProviderError: 503,
}


def _status_for(exc: CascadeError) -> int:
    for error_type, status in _STATUS_BY_ERROR.items():
        if isinstance(exc, error_type):
            return status
    return 500


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.environment != "development")

    create_tables()
    # get_state builds on first access but keeps anything already installed, so a
    # test (or a future embedding of the app) can inject its own wiring without
    # having it silently replaced at startup.
    state = get_state()

    # An immediate first poll means /catalog has real data from the start rather
    # than an empty table until the first hour elapses.
    await state.poller.poll_once()
    state.poller.start()

    logger.info(
        "cascade_started",
        environment=settings.environment,
        providers=settings.configured_providers(),
        classifier_loaded=state.classifier.is_loaded,
    )
    try:
        yield
    finally:
        await state.aclose()
        set_state(None)
        logger.info("cascade_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cascade",
        version="0.5.0",
        summary="Routes student questions to the cheapest model that can answer them correctly.",
        description=(
            "Cascade predicts how hard a question is, routes it to the cheapest "
            "capable model, verifies the answer with a model from a different "
            "provider, and escalates only when that check fails.\n\n"
            "**All cost figures are counterfactual.** Every call runs on a free "
            "provider tier; costs are computed from published per-token list "
            "prices to show what the same work would have cost."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # The frontend is served from Vercel on a domain fixed at deploy time.
        # Left permissive here because the API is public and read-mostly, with no
        # cookies or credentials to protect from a cross-origin caller.
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def attach_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Give every request an id and bind it to the logging context.

        Bound once here so classifier, router, provider, and verifier logs for one
        question can be reassembled without threading the id through every call.
        """
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(CascadeError)
    async def handle_cascade_error(request: Request, exc: CascadeError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        logger.warning("request_failed", error_code=exc.error_code, message=exc.message)
        return JSONResponse(
            status_code=_status_for(exc),
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "validation_error",
                "message": f"{field}: {first.get('msg', 'is invalid')}",
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": f"http_{exc.status_code}",
                "message": str(exc.detail),
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. The detail is logged, never returned."""
        logger.exception("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "message": "Something went wrong on our side. Please try again.",
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    app.include_router(health_routes.router)
    app.include_router(ask_routes.router, prefix=API_PREFIX)
    app.include_router(metrics_routes.router, prefix=API_PREFIX)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "cascade", "docs": "/docs", "health": "/health"}

    return app


app = create_app()


__all__ = ["app", "create_app", "get_state"]
