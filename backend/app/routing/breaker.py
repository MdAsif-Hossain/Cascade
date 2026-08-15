"""Circuit breaker for provider health.

Free tiers fail in bursts. When a provider starts refusing, every further request
spends latency to learn the same thing, and on a throttled account it actively
deepens the throttle. The breaker makes the first few failures cheap to observe
and the rest free to skip.

Health is tracked per provider *and* per model, because they fail independently:
during Phase 0 verification, ``google/gemma-4-31b-it:free`` was rate-limited
upstream in the same second that three other OpenRouter free models answered
normally. Tripping the whole provider on one bad model would discard capacity
that was working.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# CLAUDE.md section 9: three consecutive failures, then a fifteen-minute cooldown.
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 15 * 60


@dataclass
class _State:
    consecutive_failures: int = 0
    opened_at: float | None = None


@dataclass
class CircuitBreaker:
    """Tracks consecutive failures per target and skips targets that are failing.

    ``clock`` is injectable so cooldown expiry can be tested without sleeping.
    """

    threshold: int = FAILURE_THRESHOLD
    cooldown_seconds: float = COOLDOWN_SECONDS
    clock: Callable[[], float] = time.monotonic
    _states: dict[str, _State] = field(default_factory=dict, init=False)

    def _now(self) -> float:
        return self.clock()

    def _state(self, target: str) -> _State:
        return self._states.setdefault(target, _State())

    def is_open(self, target: str) -> bool:
        """True when the target should be skipped entirely.

        A breaker whose cooldown has elapsed closes itself here rather than
        needing a separate tick, so callers only ever ask this one question.
        """
        state = self._state(target)
        if state.opened_at is None:
            return False
        if self._now() - state.opened_at >= self.cooldown_seconds:
            # Cooldown served. Reset fully so the target gets a clean allowance
            # of attempts rather than tripping again on its very next failure.
            state.opened_at = None
            state.consecutive_failures = 0
            return False
        return True

    def record_success(self, target: str) -> None:
        """Clear the failure streak. One success is enough — the streak must be consecutive."""
        self._states[target] = _State()

    def record_failure(self, target: str) -> None:
        state = self._state(target)
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.threshold and state.opened_at is None:
            state.opened_at = self._now()

    def failure_count(self, target: str) -> int:
        return self._state(target).consecutive_failures

    def open_targets(self) -> list[str]:
        """Currently skipped targets, for the metrics dashboard's provider-health panel."""
        return [target for target in self._states if self.is_open(target)]
