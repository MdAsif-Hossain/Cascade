"""Circuit breaker behaviour."""

from __future__ import annotations

from app.routing.breaker import COOLDOWN_SECONDS, FAILURE_THRESHOLD, CircuitBreaker


class FakeClock:
    """Controllable clock, so cooldown expiry is tested without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTripping:
    def test_a_fresh_target_is_closed(self):
        assert CircuitBreaker().is_open("groq/llama") is False

    def test_failures_below_the_threshold_do_not_trip_it(self):
        breaker = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD - 1):
            breaker.record_failure("groq/llama")
        assert breaker.is_open("groq/llama") is False

    def test_the_threshold_failure_trips_it(self):
        breaker = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("groq/llama")
        assert breaker.is_open("groq/llama") is True

    def test_a_success_clears_the_streak(self):
        """The threshold counts consecutive failures, so a success resets it."""
        breaker = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD - 1):
            breaker.record_failure("groq/llama")
        breaker.record_success("groq/llama")
        breaker.record_failure("groq/llama")
        assert breaker.is_open("groq/llama") is False
        assert breaker.failure_count("groq/llama") == 1


class TestIsolation:
    def test_models_fail_independently(self):
        """One rate-limited free model must not disable its healthy neighbours."""
        breaker = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("openrouter/gemma:free")
        assert breaker.is_open("openrouter/gemma:free") is True
        assert breaker.is_open("openrouter/gpt-oss-20b:free") is False


class TestCooldown:
    def test_it_stays_open_during_the_cooldown(self):
        clock = FakeClock()
        breaker = CircuitBreaker(clock=clock)
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("groq/llama")
        clock.advance(COOLDOWN_SECONDS - 1)
        assert breaker.is_open("groq/llama") is True

    def test_it_closes_once_the_cooldown_elapses(self):
        clock = FakeClock()
        breaker = CircuitBreaker(clock=clock)
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("groq/llama")
        clock.advance(COOLDOWN_SECONDS)
        assert breaker.is_open("groq/llama") is False

    def test_reopening_requires_a_fresh_streak_after_cooldown(self):
        """Otherwise one failure after cooldown would immediately re-trip it."""
        clock = FakeClock()
        breaker = CircuitBreaker(clock=clock)
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("groq/llama")
        clock.advance(COOLDOWN_SECONDS)
        assert breaker.is_open("groq/llama") is False
        breaker.record_failure("groq/llama")
        assert breaker.is_open("groq/llama") is False


class TestReporting:
    def test_open_targets_are_listed_for_the_dashboard(self):
        breaker = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure("groq/llama")
        breaker.record_failure("gemini/flash")
        assert breaker.open_targets() == ["groq/llama"]
