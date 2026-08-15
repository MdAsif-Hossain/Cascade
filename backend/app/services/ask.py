"""The escalation loop: classify, answer, verify, escalate.

This is the request lifecycle from CLAUDE.md §3 steps 3-7, and the place where
the three pieces that justify the project meet. Kept out of the route handler so
the HTTP layer only validates and serialises (§13).

The loop is deliberately bounded. Two escalations maximum, because a question
that fails verification at T3 will not be rescued by trying again — at that point
the honest move is to return the best answer obtained and let the trace show what
happened, rather than burn free-tier quota hiding the failure.

Every attempt is recorded, including rejected ones. The trace is a product
surface: a student seeing "answered cheaply, checked, accepted" is being told
something true about how their answer was produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from app.core.errors import CascadeError, ProviderError
from app.providers.base import Message
from app.routing.classifier import DifficultyClassifier, Prediction
from app.routing.router import Router
from app.routing.tiers import ModelSpec, Tier, baseline_cost_usd, next_tier
from app.verification.verifier import Verdict, Verifier

logger = structlog.get_logger(__name__)

MAX_ESCALATIONS = 2

_SYSTEM = Message(
    role="system",
    content=(
        "You are a study assistant helping a student. Answer accurately and "
        "show the key reasoning steps briefly. Do not pad the answer."
    ),
)


class AnswerUnavailableError(CascadeError):
    """No tier produced an answer, so there is nothing to return."""

    error_code = "answer_unavailable"


@dataclass(frozen=True, slots=True)
class Step:
    """One tier attempt: which model answered, how it was judged, what followed."""

    tier: Tier
    provider: str
    model: str
    latency_ms: int
    verifier_score: float | None
    outcome: str
    detail: str | None = None


@dataclass
class AskResult:
    """A finished answer plus everything needed to explain how it was reached."""

    answer: str
    steps: list[Step]
    predicted_tier: Tier
    final_tier: Tier
    prediction_confidence: float
    prediction_source: str
    estimated_cost_usd: float
    baseline_cost_usd: float
    total_latency_ms: int
    escalated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_saving_usd(self) -> float:
        """Counterfactual saving against always using the top tier. Never real money."""
        return max(0.0, self.baseline_cost_usd - self.estimated_cost_usd)


class AskService:
    """Runs the full classify → answer → verify → escalate lifecycle."""

    def __init__(
        self,
        router: Router,
        classifier: DifficultyClassifier,
        verifier: Verifier,
        *,
        max_escalations: int = MAX_ESCALATIONS,
    ) -> None:
        self._router = router
        self._classifier = classifier
        self._verifier = verifier
        self._max_escalations = max_escalations

    async def ask(
        self,
        question: str,
        *,
        subject: str = "general",
        request_id: str = "",
    ) -> AskResult:
        started = time.perf_counter()
        log = logger.bind(request_id=request_id)

        prediction: Prediction = await self._classifier.predict(question, subject)
        log.info(
            "tier_predicted",
            tier=prediction.tier,
            confidence=round(prediction.confidence, 3),
            source=prediction.source,
        )

        messages = [_SYSTEM, Message(role="user", content=question)]
        steps: list[Step] = []
        warnings: list[str] = []

        tier: Tier | None = prediction.tier
        escalations = 0
        best: tuple[str, Tier, ModelSpec, int] | None = None

        while tier is not None:
            try:
                result = await self._router.complete_in_tier(messages, tier)
            except ProviderError as exc:
                # The whole tier is unreachable. Escalating is the only remaining
                # move, and it is recorded so the trace does not imply the tier
                # answered badly when it never answered at all.
                steps.append(
                    Step(
                        tier=tier,
                        provider="-",
                        model="-",
                        latency_ms=0,
                        verifier_score=None,
                        outcome="provider_error",
                        detail=exc.message,
                    )
                )
                log.warning("tier_unavailable", tier=tier, error=exc.error_code)
                tier = next_tier(tier)
                escalations += 1
                if escalations > self._max_escalations:
                    break
                continue

            completion = result.completion
            verdict: Verdict = await self._verifier.verify(
                question,
                completion.text,
                answered_by_provider=completion.provider,
                finish_reason=completion.finish_reason,
            )

            if best is None:
                # Remember the first real answer. If every later tier fails, this
                # is returned rather than an error — a checked-but-imperfect
                # answer still helps a student more than a 500 does.
                best = (completion.text, tier, result.spec, completion.latency_ms)

            accepted = verdict.passed or escalations >= self._max_escalations
            steps.append(
                Step(
                    tier=tier,
                    provider=completion.provider,
                    model=completion.model,
                    latency_ms=completion.latency_ms,
                    verifier_score=verdict.score,
                    outcome="accepted" if accepted else "escalated",
                    detail=verdict.reason,
                )
            )

            if verdict.stage in ("judge_unavailable", "judge_unparseable"):
                warnings.append(verdict.reason)

            if accepted:
                elapsed = int((time.perf_counter() - started) * 1000)
                estimated = result.spec.estimated_cost_usd(
                    completion.prompt_tokens, completion.completion_tokens
                )
                baseline = baseline_cost_usd(completion.prompt_tokens, completion.completion_tokens)
                log.info(
                    "answer_accepted",
                    tier=tier,
                    provider=completion.provider,
                    escalations=escalations,
                    score=round(verdict.score, 3),
                )
                return AskResult(
                    answer=completion.text,
                    steps=steps,
                    predicted_tier=prediction.tier,
                    final_tier=tier,
                    prediction_confidence=prediction.confidence,
                    prediction_source=prediction.source,
                    estimated_cost_usd=estimated,
                    baseline_cost_usd=baseline,
                    total_latency_ms=elapsed,
                    escalated=escalations > 0,
                    warnings=warnings,
                )

            log.info("escalating", from_tier=tier, reason=verdict.reason)
            tier = next_tier(tier)
            escalations += 1
            if escalations > self._max_escalations:
                break

        if best is not None:
            answer, answered_tier, spec, latency = best
            warnings.append("Returned the best available answer after exhausting escalations.")
            return AskResult(
                answer=answer,
                steps=steps,
                predicted_tier=prediction.tier,
                final_tier=answered_tier,
                prediction_confidence=prediction.confidence,
                prediction_source=prediction.source,
                estimated_cost_usd=spec.estimated_cost_usd(0, 0),
                baseline_cost_usd=0.0,
                total_latency_ms=int((time.perf_counter() - started) * 1000),
                escalated=True,
                warnings=warnings,
            )

        log.error("all_tiers_failed", attempts=len(steps))
        raise AnswerUnavailableError(
            "Every model tier failed to answer. This usually means the free-tier "
            "providers are rate-limited; try again in a minute."
        )
