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
import math
import pathlib
import re
from dataclasses import dataclass

import httpx
import numpy as np

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"

# gemini-embedding-001 returns 3,072 dimensions natively. With roughly a thousand
# training rows that is far more features than samples, and logistic regression
# would fit noise happily. The model supports Matryoshka truncation, so a shorter
# vector is requested rather than trained on and regretted.
EMBED_DIMENSIONS = 256

CACHE_PATH = pathlib.Path(__file__).parent / "data" / "raw" / "embeddings.jsonl"

SUBJECTS = ("math", "science", "history", "language", "general")
QUESTION_WORDS = ("what", "why", "how", "which", "who", "when", "where")

_MATH_OPERATORS = re.compile(r"[+\-*/=<>%^]|\b(?:times|divided|per cent|percent)\b")
_NUMBER = re.compile(r"\d")


HANDCRAFTED_NAMES: tuple[str, ...] = (
    "char_count",
    "word_count",
    "mean_word_length",
    "digit_density",
    "number_count",
    "math_operator_count",
    "has_math_operator",
    "sentence_count",
    "comma_count",
    "num_choices",
    *(f"qword_{w}" for w in QUESTION_WORDS),
    "qword_other",
    *(f"subject_{s}" for s in SUBJECTS),
)


@dataclass(frozen=True, slots=True)
class Sample:
    """One labelled question, as the feature builder needs it."""

    text: str
    subject: str
    num_choices: int


def handcrafted_features(sample: Sample) -> list[float]:
    """Surface statistics of a question.

    Counts are log-scaled because raw length is heavy-tailed — a single
    500-word word problem would otherwise dominate a linear model's view of every
    other feature.
    """
    text = sample.text
    words = text.split()
    digits = len(_NUMBER.findall(text))
    operators = len(_MATH_OPERATORS.findall(text.lower()))
    lowered = text.strip().lower()

    first_word = lowered.split()[0] if words else ""
    qword_flags = [1.0 if first_word.startswith(w) else 0.0 for w in QUESTION_WORDS]
    qword_flags.append(0.0 if any(qword_flags) else 1.0)

    subject_flags = [1.0 if sample.subject == s else 0.0 for s in SUBJECTS]

    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        digits / len(text) if text else 0.0,
        math.log1p(digits),
        math.log1p(operators),
        1.0 if operators else 0.0,
        math.log1p(text.count(".") + text.count("?") + text.count("!")),
        math.log1p(text.count(",")),
        float(sample.num_choices),
        *qword_flags,
        *subject_flags,
    ]


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
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]

        response = client.post(
            EMBED_URL,
            headers={"x-goog-api-key": self._api_key},
            json={
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": EMBED_DIMENSIONS,
            },
        )
        response.raise_for_status()
        vector = [float(v) for v in response.json()["embedding"]["values"]]
        self._store(key, vector)
        return vector

    def embed_many(self, texts: list[str], *, progress_every: int = 100) -> np.ndarray:
        """Embed a list of texts, reporting progress on the uncached ones."""
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
        return np.array([self._cache[self._key(t)] for t in texts], dtype=np.float64)


def build_handcrafted_matrix(samples: list[Sample]) -> np.ndarray:
    return np.array([handcrafted_features(s) for s in samples], dtype=np.float64)
