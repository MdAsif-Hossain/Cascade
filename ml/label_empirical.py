"""Generate empirical difficulty labels by measuring what models actually get right.

The label *is* the routing decision, which is what makes it worth the API calls:

* a T1 model answers correctly           -> ``T1``
* T1 fails, a T2 model answers correctly -> ``T2``
* both fail                              -> ``T3``

Proxy labels (question length, dataset tags) would be free but indefensible —
they describe what a question looks like, not what it costs to answer. This
measures the thing being predicted.

Two properties matter more than speed here:

**Every response is cached to disk before anything else happens.** A run that
dies halfway, a rate limit, a laptop closing — none of them should cost calls
already made, and re-running must produce identical labels rather than fresh
samples from a non-deterministic model.

**The two labelling models sit on different providers.** Groq and Google meter
separately, so T1 and T2 grading do not compete for the same quota, and the run
finishes in roughly half the wall time it otherwise would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.core.errors import ProviderError, ProviderRateLimitError  # noqa: E402
from app.providers.base import Message, Provider  # noqa: E402
from app.providers.gemini import GeminiProvider  # noqa: E402
from app.providers.groq import GroqProvider  # noqa: E402

from datasets import Question, load_pool  # noqa: E402
from grading import grade  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "raw" / "responses.jsonl"
LABELS_PATH = DATA_DIR / "labeled" / "labels.jsonl"

T1_PROVIDER, T1_MODEL = "groq", "llama-3.1-8b-instant"
T2_PROVIDER, T2_MODEL = "gemini", "gemini-3.1-flash-lite"

SYSTEM = Message(
    role="system",
    content="You are answering exam questions. Follow the answer format exactly. Be brief.",
)

# Numeric questions need room to work through the arithmetic before stating the
# answer; multiple choice needs a letter. Sizing these separately keeps the
# maths questions from being truncated mid-derivation, which would show up as
# difficulty rather than as the budget error it is.
MAX_TOKENS_MCQ = 256
MAX_TOKENS_NUMERIC = 768

CONCURRENCY = 4
MAX_RATE_LIMIT_RETRIES = 6


@dataclass
class Attempt:
    """One graded model attempt at one question."""

    question_id: str
    model: str
    answer: str
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    error: str | None = None


class ResponseCache:
    """Append-only JSONL cache keyed by (model, question).

    Append-only rather than rewritten so a crash mid-write cannot corrupt work
    already done — the worst case is one truncated final line, which is skipped
    on load.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[tuple[str, str], Attempt] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail from an interrupted run
            attempt = Attempt(**record)
            self._entries[(attempt.model, attempt.question_id)] = attempt

    def get(self, model: str, question_id: str) -> Attempt | None:
        return self._entries.get((model, question_id))

    def put(self, attempt: Attempt) -> None:
        self._entries[(attempt.model, attempt.question_id)] = attempt
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(attempt)) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


async def ask(
    provider: Provider, model: str, question: Question, cache: ResponseCache
) -> Attempt:
    """Ask one model one question, grading and caching the result.

    Rate limits are waited out rather than treated as failures: a 429 says
    nothing about whether the model can answer, and recording it as a wrong
    answer would push an easy question into a higher tier.
    """
    cached = cache.get(model, question.id)
    if cached is not None:
        return cached

    max_tokens = MAX_TOKENS_MCQ if question.is_multiple_choice else MAX_TOKENS_NUMERIC
    messages = [SYSTEM, Message(role="user", content=question.prompt())]

    for attempt_number in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            completion = await provider.complete(messages, model, max_tokens=max_tokens)
        except ProviderRateLimitError as exc:
            if attempt_number == MAX_RATE_LIMIT_RETRIES:
                return Attempt(question.id, model, "", False, 0, 0, 0, error=str(exc))
            wait = exc.retry_after or min(2.0 * attempt_number, 30.0)
            await asyncio.sleep(wait)
            continue
        except ProviderError as exc:
            # A genuine provider failure is recorded but NOT cached: caching it
            # would bake a transient outage into the labels permanently.
            return Attempt(question.id, model, "", False, 0, 0, 0, error=str(exc))

        result = Attempt(
            question_id=question.id,
            model=model,
            answer=completion.text,
            correct=grade(completion.text, choices=question.choices, correct=question.answer),
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=completion.latency_ms,
        )
        cache.put(result)
        return result

    return Attempt(question.id, model, "", False, 0, 0, 0, error="retries exhausted")


async def run_stage(
    provider: Provider,
    model: str,
    questions: list[Question],
    cache: ResponseCache,
    label: str,
) -> dict[str, Attempt]:
    """Run one model across a set of questions with bounded concurrency."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: dict[str, Attempt] = {}
    done = 0
    started = time.perf_counter()

    async def worker(question: Question) -> None:
        nonlocal done
        async with semaphore:
            results[question.id] = await ask(provider, model, question, cache)
            done += 1
            if done % 25 == 0 or done == len(questions):
                elapsed = time.perf_counter() - started
                correct = sum(a.correct for a in results.values())
                errors = sum(a.error is not None for a in results.values())
                print(
                    f"  [{label}] {done}/{len(questions)}  correct={correct}  "
                    f"errors={errors}  {elapsed:.0f}s",
                    flush=True,
                )

    await asyncio.gather(*(worker(q) for q in questions))
    return results


def load_keys() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


async def main(limit: int | None) -> None:
    keys = load_keys()
    pool = load_pool()
    if limit:
        # Take an even slice across sources rather than the first N, which would
        # otherwise be entirely ARC-Easy and make a pilot look deceptively easy.
        step = max(1, len(pool) // limit)
        pool = pool[::step][:limit]

    print(f"pool: {len(pool)} questions", flush=True)
    cache = ResponseCache(CACHE_PATH)
    print(f"cache: {len(cache)} responses already on disk", flush=True)

    groq = GroqProvider(keys["GROQ_API_KEY"])
    gemini = GeminiProvider(keys["GOOGLE_AI_STUDIO_API_KEY"])

    try:
        print(f"\nstage 1 — {T1_PROVIDER}/{T1_MODEL}", flush=True)
        t1 = await run_stage(groq, T1_MODEL, pool, cache, "T1")

        # Only questions T1 missed need a T2 attempt. This is most of the run's
        # savings: an easy pool means very few T2 calls.
        needs_t2 = [q for q in pool if not t1[q.id].correct]
        print(f"\nstage 2 — {T2_PROVIDER}/{T2_MODEL} on {len(needs_t2)} T1 failures", flush=True)
        t2 = await run_stage(gemini, T2_MODEL, needs_t2, cache, "T2") if needs_t2 else {}
    finally:
        await groq.aclose()
        await gemini.aclose()

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {"T1": 0, "T2": 0, "T3": 0}
    skipped = 0

    with LABELS_PATH.open("w", encoding="utf-8") as handle:
        for question in pool:
            a1 = t1[question.id]
            a2 = t2.get(question.id)

            if a1.error is not None or (a2 is not None and a2.error is not None):
                # An unanswered question has no measured difficulty. Labelling it
                # T3 would quietly turn provider outages into training signal.
                skipped += 1
                continue

            if a1.correct:
                label = "T1"
            elif a2 is not None and a2.correct:
                label = "T2"
            else:
                label = "T3"
            counts[label] += 1

            handle.write(
                json.dumps(
                    {
                        "id": question.id,
                        "source": question.source,
                        "subject": question.subject,
                        "text": question.text,
                        "answer": question.answer,
                        "num_choices": len(question.choices),
                        "t1_correct": a1.correct,
                        "t2_correct": a2.correct if a2 is not None else None,
                        "label": label,
                    }
                )
                + "\n"
            )

    total = sum(counts.values())
    print(f"\nwrote {total} labels to {LABELS_PATH}", flush=True)
    for tier, count in counts.items():
        share = 100 * count / total if total else 0
        print(f"  {tier}: {count:5d}  ({share:.1f}%)", flush=True)
    if skipped:
        print(f"  skipped (provider error): {skipped}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="label only N questions (pilot run)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit))
