"""Cheap answer checks that run before any judge model is called.

These exist for cost, not accuracy. An empty answer, a refusal, or a reply that
stops mid-sentence needs no second model to recognise, and spending a judge call
to discover it wastes the free-tier budget the whole project runs on.

They are deliberately conservative. A heuristic that wrongly rejects a good
answer forces a needless escalation, which costs more than the judge call it
saved — so every rule here fires only on evidence that is hard to argue with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Only genuinely empty answers are rejected on length. An earlier version used a
# two-character floor and rejected "4" — a complete, correct answer to "what is
# 2+2". Any floor above 1 punishes the terse answers that cheap models are best
# at, which is exactly the wrong direction for a system built on routing to them.
MIN_ANSWER_CHARS = 1

# Above this, a "concise" answer has clearly run away with itself. Generous, so
# that a genuinely long derivation is not punished.
MAX_ANSWER_CHARS = 12_000

_REFUSAL_PATTERNS = (
    re.compile(r"\bi (?:can(?:no|')t|am unable to|won't) (?:help|assist|answer)", re.I),
    re.compile(r"\bas an ai\b.{0,40}\b(?:cannot|can't|unable)", re.I),
    re.compile(r"\bi'?m sorry\b.{0,40}\b(?:cannot|can't|unable)", re.I),
    re.compile(r"\bi (?:do not|don't) have (?:enough )?(?:information|context)\b", re.I),
)

# A sentence that ends without terminal punctuation, a closing brace, or a digit
# is a strong truncation signal — but only when the provider also said it ran out
# of room, which is why finish_reason is required alongside.
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    """Outcome of the cheap checks."""

    passed: bool
    reason: str | None = None

    @property
    def failed(self) -> bool:
        return not self.passed


def check(answer: str, *, finish_reason: str = "stop") -> HeuristicResult:
    """Screen an answer for obvious, model-free failure modes.

    ``finish_reason`` is normalised to lower case by the provider layer, so this
    does not need to know which provider produced it.
    """
    stripped = answer.strip()

    if not stripped:
        return HeuristicResult(False, "empty answer")

    if len(stripped) < MIN_ANSWER_CHARS:
        return HeuristicResult(False, f"answer shorter than {MIN_ANSWER_CHARS} characters")

    if len(stripped) > MAX_ANSWER_CHARS:
        return HeuristicResult(False, f"answer longer than {MAX_ANSWER_CHARS} characters")

    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(stripped):
            return HeuristicResult(False, "answer is a refusal")

    if finish_reason in _TRUNCATION_FINISH_REASONS:
        return HeuristicResult(False, f"answer truncated (finish_reason={finish_reason})")

    return HeuristicResult(True)
