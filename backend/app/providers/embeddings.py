"""Question embedding via the Gemini embedding API.

Uses the same Google AI Studio key as generation, which is what keeps the
embedder free (ADR-0001). ``gemini-embedding-001`` returns 3,072 dimensions
natively; Cascade requests 256 via Matryoshka truncation because the classifier
was trained on 256 and because a 3,072-wide vector on ~1,000 training rows
invites overfitting.

Failure returns ``None`` rather than raising. The classifier treats a missing
embedding as "no prediction available" and falls back to a default tier, so an
embedding outage costs routing quality rather than the whole request.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import httpx
import structlog

logger = structlog.get_logger(__name__)

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"

# Must match ml/features.py::EMBED_DIMENSIONS. A mismatch produces a vector the
# model cannot consume, which surfaces as a shape error on every request.
EMBED_DIMENSIONS = 256

EMBED_TIMEOUT_SECONDS = 20.0

# Small in-process cache. Render's free tier is one container with no Redis, and
# repeated questions are common enough in a study assistant that even a modest
# cache removes a network round trip from the hot path.
CACHE_SIZE = 512


class Embedder:
    """Embeds question text, with a bounded in-memory cache."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        cache_size: int = CACHE_SIZE,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECONDS)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _remember(self, key: str, vector: list[float]) -> None:
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def embed(self, text: str) -> list[float] | None:
        """Return the embedding for ``text``, or None if it cannot be produced."""
        if not self._api_key:
            return None

        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        try:
            response = await self.client.post(
                EMBED_URL,
                headers={"x-goog-api-key": self._api_key},
                json={
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": EMBED_DIMENSIONS,
                },
            )
        except httpx.HTTPError as exc:
            logger.warning("embedding_request_failed", error=str(exc))
            return None

        if not response.is_success:
            logger.warning("embedding_rejected", status=response.status_code)
            return None

        try:
            vector = [float(v) for v in response.json()["embedding"]["values"]]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("embedding_unparseable", error=str(exc))
            return None

        self._remember(key, vector)
        return vector
