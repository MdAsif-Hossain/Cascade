"""Tier resolution: turn a tier into an answer, or explain why it could not.

The router's job in Phase 1 is narrow and worth stating precisely, because the
temptation is to let it grow into the whole system. It picks a healthy model from
a tier, calls it, fails over within the tier when that call fails, and records
what happened. It does **not** decide which tier to use (that is the classifier)
and it does **not** decide whether the answer was good enough (that is the
verifier). Escalation between tiers is driven from outside, by the verifier.

Every attempt becomes a ``RoutingStep``, including the failed ones. Those steps
are shown to the student, not just logged — the trace is the product surface that
makes the routing decision legible rather than something the system hides.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.core.errors import ProviderError
from app.providers.base import Completion, Message, Provider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider
from app.routing.breaker import CircuitBreaker
from app.routing.tiers import ModelSpec, Tier, candidates

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    GroqProvider.name: GroqProvider,
    GeminiProvider.name: GeminiProvider,
    OpenRouterProvider.name: OpenRouterProvider,
}

# One retry only. A second attempt absorbs a transient blip; a third mostly
# spends the free tier's budget re-learning that the provider is unwell, when
# failing over to a different provider is both faster and more likely to work.
MAX_ATTEMPTS_PER_MODEL = 2
RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRY_WAIT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class RoutingStep:
    """One attempt against one model, successful or not."""

    tier: Tier
    provider: str
    model: str
    latency_ms: int
    outcome: str
    error: str | None = None


class NoProviderAvailableError(ProviderError):
    """Every candidate in the tier was unavailable, unhealthy, or unconfigured."""

    error_code = "no_provider_available"
    retryable = True


@dataclass
class TierResult:
    """The outcome of resolving one tier, with the full attempt history."""

    completion: Completion
    spec: ModelSpec
    steps: list[RoutingStep] = field(default_factory=list)


class Router:
    """Resolves a tier to an answer, failing over across providers within the tier."""

    def __init__(
        self,
        settings: Settings | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._breaker = breaker or CircuitBreaker()
        self._providers: dict[str, Provider] = {}

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def _provider_for(self, name: str) -> Provider:
        """Reuse one adapter instance per provider so its connection pool survives."""
        if name not in self._providers:
            self._providers[name] = PROVIDER_CLASSES[name](self._settings.key_for(name))
        return self._providers[name]

    async def aclose(self) -> None:
        await asyncio.gather(*(p.aclose() for p in self._providers.values()))
        self._providers.clear()

    @staticmethod
    def _target(spec: ModelSpec) -> str:
        return f"{spec.provider}/{spec.model}"

    async def complete_in_tier(
        self,
        messages: list[Message],
        tier: Tier,
        *,
        exclude_providers: frozenset[str] = frozenset(),
    ) -> TierResult:
        """Answer using the first healthy candidate in ``tier``.

        ``exclude_providers`` exists for the verifier's cross-provider rule: the
        judge must not run on the provider that produced the answer, so the
        caller can rule that provider out before resolution starts rather than
        discovering the conflict afterwards.
        """
        steps: list[RoutingStep] = []
        skipped: list[str] = []

        for spec in candidates(tier):
            target = self._target(spec)

            if spec.provider in exclude_providers:
                skipped.append(f"{target} (excluded)")
                continue
            if not self._settings.key_for(spec.provider):
                skipped.append(f"{target} (no API key)")
                continue
            if self._breaker.is_open(target):
                skipped.append(f"{target} (circuit open)")
                continue

            step, completion = await self._try_model(messages, tier, spec)
            steps.append(step)
            if completion is not None:
                return TierResult(completion=completion, spec=spec, steps=steps)

        raise NoProviderAvailableError(
            provider=tier,
            message=(
                f"no candidate in {tier} produced an answer "
                f"(attempted {len(steps)}, skipped {len(skipped)}: {'; '.join(skipped) or 'none'})"
            ),
        )

    async def _try_model(
        self, messages: list[Message], tier: Tier, spec: ModelSpec
    ) -> tuple[RoutingStep, Completion | None]:
        """Call one model, retrying only errors that a retry could plausibly fix."""
        target = self._target(spec)
        provider = self._provider_for(spec.provider)
        last_error: ProviderError | None = None
        latency_ms = 0

        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                completion = await provider.complete(messages, spec.model)
            except ProviderError as exc:
                last_error = exc
                self._breaker.record_failure(target)
                if not exc.retryable or attempt == MAX_ATTEMPTS_PER_MODEL:
                    break
                await asyncio.sleep(self._backoff_for(exc, attempt))
                continue

            self._breaker.record_success(target)
            return (
                RoutingStep(
                    tier=tier,
                    provider=spec.provider,
                    model=spec.model,
                    latency_ms=completion.latency_ms,
                    outcome="accepted",
                ),
                completion,
            )

        assert last_error is not None
        return (
            RoutingStep(
                tier=tier,
                provider=spec.provider,
                model=spec.model,
                latency_ms=latency_ms,
                outcome="provider_error",
                error=f"{type(last_error).__name__}: {last_error.message}",
            ),
            None,
        )

    @staticmethod
    def _backoff_for(exc: ProviderError, attempt: int) -> float:
        """Prefer the provider's own Retry-After hint over our exponential guess."""
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, int | float) and retry_after > 0:
            return min(float(retry_after), MAX_RETRY_WAIT_SECONDS)
        return min(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_WAIT_SECONDS)
