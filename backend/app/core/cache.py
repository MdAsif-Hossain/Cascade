"""Answer cache keyed by the normalised question.

The cheapest possible answer is one that costs no model call at all. Students ask
overlapping questions, and on a free tier every avoided call is quota preserved
for a question that actually needs it.

Normalisation is deliberately conservative — case, whitespace, and trailing
punctuation only. Aggressive normalisation (stripping stopwords, stemming) would
collide questions that differ in meaning, and serving a confidently wrong cached
answer is far worse than missing a cache hit.

In-process rather than Redis: Render's free tier runs one container with no
add-ons. The cache is therefore lost on restart, which is acceptable — it is an
optimisation, not a store of record. The database holds the durable trace.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_MAX_ENTRIES = 500
DEFAULT_TTL_SECONDS = 60 * 60 * 24

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[?!.\s]+$")


def normalise_question(question: str) -> str:
    """Fold away differences that never change the answer."""
    collapsed = _WHITESPACE.sub(" ", question).strip().lower()
    return _TRAILING_PUNCTUATION.sub("", collapsed)


def cache_key(question: str, subject: str = "general", level: str = "school") -> str:
    """Hash the normalised question together with the parameters that shape the answer.

    Subject and level are part of the key because they change how the question is
    routed and answered — the same words at school and undergraduate level should
    not share a cached answer.
    """
    material = f"{normalise_question(question)}|{subject}|{level}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class _Entry(Generic[T]):
    value: T
    stored_at: float


class AnswerCache(Generic[T]):
    """A bounded, TTL'd, least-recently-used cache."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: object = time.monotonic,
    ) -> None:
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._hits = 0
        self._misses = 0

    def _now(self) -> float:
        return float(self._clock())  # type: ignore[operator]

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self._now() - entry.stored_at > self._ttl:
            del self._entries[key]
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.value

    def put(self, key: str, value: T) -> None:
        self._entries[key] = _Entry(value=value, stored_at=self._now())
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def hit_rate(self) -> float:
        """Reported on the metrics dashboard; 0.0 before any lookup happens."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def stats(self) -> dict[str, float | int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._entries),
            "hit_rate": round(self.hit_rate, 4),
        }
