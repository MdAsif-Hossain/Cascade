"""Answer verification: cheap heuristics, then an LLM judge on a different provider.

Verification is what makes cheap routing safe. Without it, sending a question to
the smallest model is a gamble the student pays for; with it, a bad cheap answer
is caught and retried higher up, and the cost saving stops being a quality
trade.

**The judge must not share a provider with the answerer.** Models systematically
prefer their own outputs, and a family judging itself produces scores that look
like quality measurement but are not. This is enforced structurally — the judge
provider is excluded before routing, not checked afterwards — so it cannot be
bypassed by a config change or a tier reshuffle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import ProviderError
from app.providers.base import Message
from app.routing.router import Router
from app.routing.tiers import Tier
from app.verification import heuristics

# Below this the answer is not trusted and the question escalates (CLAUDE.md §8).
PASS_THRESHOLD = 0.7

# The judge is a grader, not a tutor: it needs few tokens, and a large budget
# just invites it to write an essay instead of a score.
JUDGE_MAX_TOKENS = 200

# Judging is cheap work, so it runs on the cheapest band. Using a strong model to
# grade a weak one would cost more than simply answering with the strong model.
JUDGE_TIER: Tier = "T1"

_JUDGE_SYSTEM = Message(
    role="system",
    content=(
        "You grade answers to student questions. Reply with only a decimal "
        "between 0 and 1 on the first line, then one short sentence of "
        "justification. 1.0 means correct and complete; 0.0 means wrong or "
        "missing. Judge correctness first, completeness second."
    ),
)

_SCORE = re.compile(r"(\d*\.?\d+)")


@dataclass(frozen=True, slots=True)
class Verdict:
    """The result of verifying one answer."""

    score: float
    passed: bool
    stage: str
    reason: str
    judge_provider: str | None = None
    judge_model: str | None = None

    @property
    def should_escalate(self) -> bool:
        return not self.passed


def parse_score(text: str) -> float | None:
    """Pull the judge's score out of its reply.

    Only the first number is considered, and anything outside 0–1 is rejected
    rather than clamped: a judge answering "85" meant 85%, but guessing that is
    how a verifier silently starts scoring on a different scale than intended.
    """
    match = _SCORE.search(text.strip())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def build_judge_prompt(question: str, answer: str) -> list[Message]:
    return [
        _JUDGE_SYSTEM,
        Message(
            role="user",
            content=(
                f"Question:\n{question}\n\n"
                f"Answer to grade:\n{answer}\n\n"
                "Score (0-1) then one sentence:"
            ),
        ),
    ]


class Verifier:
    """Two-stage answer verification with a cross-provider judge."""

    def __init__(self, router: Router, *, pass_threshold: float = PASS_THRESHOLD) -> None:
        self._router = router
        self._threshold = pass_threshold

    async def verify(
        self,
        question: str,
        answer: str,
        *,
        answered_by_provider: str,
        finish_reason: str = "stop",
    ) -> Verdict:
        """Verify an answer, escalating cheap checks before paying for a judge."""
        cheap = heuristics.check(answer, finish_reason=finish_reason)
        if cheap.failed:
            return Verdict(
                score=0.0,
                passed=False,
                stage="heuristic",
                reason=cheap.reason or "failed heuristic check",
            )

        try:
            result = await self._router.complete_in_tier(
                build_judge_prompt(question, answer),
                JUDGE_TIER,
                # The structural guarantee: the answerer's provider is removed
                # from the candidate list before a judge is ever chosen.
                exclude_providers=frozenset({answered_by_provider}),
            )
        except ProviderError as exc:
            # No judge available. Accepting is the right failure mode: the answer
            # already passed the cheap checks, and escalating every request during
            # a provider outage would multiply load exactly when capacity is short.
            return Verdict(
                score=self._threshold,
                passed=True,
                stage="judge_unavailable",
                reason=f"no cross-provider judge available ({exc.error_code}); accepted unverified",
            )

        score = parse_score(result.completion.text)
        if score is None:
            return Verdict(
                score=self._threshold,
                passed=True,
                stage="judge_unparseable",
                reason="judge reply had no usable score; accepted unverified",
                judge_provider=result.spec.provider,
                judge_model=result.spec.model,
            )

        passed = score >= self._threshold
        return Verdict(
            score=score,
            passed=passed,
            stage="judge",
            reason=(
                f"judged {score:.2f} (threshold {self._threshold:.2f})"
                if passed
                else f"judged {score:.2f}, below threshold {self._threshold:.2f}"
            ),
            judge_provider=result.spec.provider,
            judge_model=result.spec.model,
        )
