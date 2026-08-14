"""Shared application state and FastAPI dependencies.

Built once at startup and reused, because each of these owns something expensive:
adapters own HTTP connection pools, the classifier owns a deserialised model, the
embedder and answer cache own warm entries. Rebuilding them per request would add
a TLS handshake and a model load to every question.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.catalog.poller import CatalogPoller
from app.core.cache import AnswerCache
from app.core.config import Settings, get_settings
from app.providers.embeddings import Embedder
from app.routing.classifier import DifficultyClassifier
from app.routing.router import PROVIDER_CLASSES, Router
from app.services.ask import AskResult, AskService
from app.verification.verifier import Verifier


@dataclass
class AppState:
    """Everything the API needs, wired together."""

    settings: Settings
    router: Router
    classifier: DifficultyClassifier
    verifier: Verifier
    ask_service: AskService
    embedder: Embedder
    cache: AnswerCache[AskResult]
    poller: CatalogPoller

    async def aclose(self) -> None:
        await self.poller.stop()
        await self.router.aclose()
        await self.embedder.aclose()


def build_state(settings: Settings | None = None) -> AppState:
    settings = settings or get_settings()

    router = Router(settings)
    embedder = Embedder(settings.google_ai_studio_api_key)
    classifier = DifficultyClassifier(embedder)
    verifier = Verifier(router)
    ask_service = AskService(router, classifier, verifier)

    # The poller gets its own adapter instances so a catalog request can never
    # queue behind a student's question on the same connection pool.
    poller_providers = {
        name: PROVIDER_CLASSES[name](settings.key_for(name))
        for name in settings.configured_providers()
    }

    return AppState(
        settings=settings,
        router=router,
        classifier=classifier,
        verifier=verifier,
        ask_service=ask_service,
        embedder=embedder,
        cache=AnswerCache[AskResult](),
        poller=CatalogPoller(poller_providers),
    )


_state: AppState | None = None


def set_state(state: AppState | None) -> None:
    global _state
    _state = state


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = build_state()
    return _state
