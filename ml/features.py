"""Feature extraction for the difficulty classifier.

Two families of features, kept separable because the ablation study's whole job
is to measure what each contributes:

**Handcrafted** — cheap, interpretable signals computable without a network call:
length, digit density, arithmetic operators, question-word type, subject.

**Embedding** — a hosted sentence embedding of the question text, capturing
meaning that surface statistics miss.

The split is not cosmetic. If embeddings alone match the combination, the
handcrafted features are decoration and the report should say so. If handcrafted
alone comes close, the embedding API call on every request is hard to justify at
all. Either finding is worth more than an unexamined "we used both".

Embeddings come from ``gemini-embedding-001`` on the same key that serves
generation, and are cached to disk keyed by text hash — the same question must
never be paid for twice, and cached vectors keep training reproducible.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import sys

import httpx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

# Imported, never redefined: training and inference must compute identical
# features or the persisted model silently becomes meaningless.
from app.routing.features import (  # noqa: E402
    FEATURE_NAMES as HANDCRAFTED_NAMES,
)
from app.routing.features import (  # noqa: E402
    QUESTION_WORDS,
    SUBJECTS,
    Sample,
    handcrafted_features,
)

__all__ = [
    "EMBED_DIMENSIONS",
    "EMBED_MODEL",
    "HANDCRAFTED_NAMES",
    "QUESTION_WORDS",
    "SUBJECTS",
    "EmbeddingClient",
    "Sample",
    "build_handcrafted_matrix",
    "handcrafted_features",
]

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"

# gemini-embedding-001 returns 3,072 dimensions natively. With roughly a thousand
# training rows that is far more features than samples, and logistic regression
# would fit noise happily. The model supports Matryoshka truncation, so a shorter
# vector is requested rather than trained on and regretted.
EMBED_DIMENSIONS = 256

CACHE_PATH = pathlib.Path(__file__).parent / "data" / "raw" / "embeddings.jsonl"

MAX_EMBED_RETRIES = 12
RETRY_BASE_SECONDS = 20.0
MAX_RETRY_WAIT_SECONDS = 120.0

# Spacing between uncached calls. The free embedding tier throttles on sustained
# request rate, and the throttle outlasts a short backoff: a run at ~69 req/min
# sailed through 874 embeddings and then hit a 429 that persisted past seven
# minutes of retries. One request per second stays under it, and the retry ladder
# below is deliberately patient enough to outlast the throttle if it trips again.
REQUEST_SPACING_SECONDS = 1.0


def _retry_after(response: httpx.Response) -> float | None:
    """Honour the provider's own backoff hint when it sends one."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class EmbeddingClient:
    """Fetches question embeddings, caching every vector to disk.

    Cached because embeddings are deterministic, the free tier is finite, and a
    training run that silently re-embeds is a training run whose results cannot
    be reproduced.
    """

    def __init__(self, api_key: str, cache_path: pathlib.Path = CACHE_PATH) -> None:
        self._api_key = api_key
        self._cache_path = cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        self._load()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def has_cached(self, text: str) -> bool:
        """True when this text can be embedded without a network call."""
        return self._key(text) in self._cache

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._cache[record["key"]] = record["vector"]

    def _store(self, key: str, vector: list[float]) -> None:
        self._cache[key] = vector
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "vector": vector}) + "\n")

    def embed(self, text: str, client: httpx.Client) -> list[float]:
        """Embed one text, waiting out rate limits rather than failing the run.

        The embedding endpoint meters separately from generation and throttles
        well before a thousand requests. Without backoff a single 429 aborts
        training after paying for every embedding computed up to that point —
        which is exactly what happened on the first attempt.
        """
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]

        for attempt in range(1, MAX_EMBED_RETRIES + 1):
            response = client.post(
                EMBED_URL,
                headers={"x-goog-api-key": self._api_key},
                json={
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": EMBED_DIMENSIONS,
                },
            )

            if response.status_code == 429:
                if attempt == MAX_EMBED_RETRIES:
                    response.raise_for_status()
                wait = _retry_after(response) or min(
                    RETRY_BASE_SECONDS * attempt, MAX_RETRY_WAIT_SECONDS
                )
                print(f"    rate limited, waiting {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            vector = [float(v) for v in response.json()["embedding"]["values"]]
            self._store(key, vector)
            return vector

        raise RuntimeError("unreachable: retries exhausted without raising")

    def embed_many(self, texts: list[str], *, progress_every: int = 100) -> np.ndarray:
        """Embed a list of texts, reporting progress on the uncached ones.

        Paced with a small delay between uncached calls. Every vector is written
        to disk as it arrives, so an interrupted run resumes rather than
        restarting — the same property that made the labelling run survivable.
        """
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            print(f"  embedding {len(missing)} new texts ({len(texts) - len(missing)} cached)")
        done = 0
        with httpx.Client(timeout=90.0) as client:
            for text in texts:
                was_cached = self._key(text) in self._cache
                self.embed(text, client)
                if not was_cached:
                    done += 1
                    if done % progress_every == 0:
                        print(f"    {done}/{len(missing)}")
                    time.sleep(REQUEST_SPACING_SECONDS)
        return np.array([self._cache[self._key(t)] for t in texts], dtype=np.float64)


def build_handcrafted_matrix(samples: list[Sample]) -> np.ndarray:
    return np.array([handcrafted_features(s) for s in samples], dtype=np.float64)
