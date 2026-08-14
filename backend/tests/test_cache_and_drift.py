"""Answer cache normalisation/eviction and catalog drift rules."""

from __future__ import annotations

import pytest

from app.catalog.drift import affected_tier_models, diff_models, diff_reachability
from app.catalog.poller import CatalogPoller
from app.core.cache import AnswerCache, cache_key, normalise_question
from app.core.errors import ProviderUnavailableError
from app.providers.base import Completion, Message, Provider


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestNormalisation:
    @pytest.mark.parametrize(
        "variant",
        [
            "Why does ice float?",
            "why does ice float",
            "  Why  does   ice   float?  ",
            "WHY DOES ICE FLOAT!",
            "Why does ice float.",
        ],
    )
    def test_harmless_variations_share_a_key(self, variant):
        assert cache_key(variant) == cache_key("Why does ice float?")

    def test_different_questions_do_not_collide(self):
        assert cache_key("Why does ice float?") != cache_key("Why does ice melt?")

    def test_negation_is_not_folded_away(self):
        """Aggressive normalisation here would serve a confidently wrong answer."""
        assert cache_key("Is water a compound?") != cache_key("Is water not a compound?")

    def test_subject_is_part_of_the_key(self):
        assert cache_key("What is a wave?", "math") != cache_key("What is a wave?", "science")

    def test_level_is_part_of_the_key(self):
        """The same words deserve different answers at school and undergraduate level."""
        assert cache_key("What is a derivative?", "math", "school") != cache_key(
            "What is a derivative?", "math", "undergrad"
        )

    def test_internal_whitespace_is_collapsed(self):
        assert normalise_question("a   b\n\tc") == "a b c"


class TestCacheBehaviour:
    def test_a_miss_returns_none(self):
        assert AnswerCache[str]().get("absent") is None

    def test_a_stored_value_is_returned(self):
        cache = AnswerCache[str]()
        cache.put("k", "v")
        assert cache.get("k") == "v"

    def test_entries_expire(self):
        clock = FakeClock()
        cache = AnswerCache[str](ttl_seconds=60, clock=clock)
        cache.put("k", "v")
        clock.advance(61)
        assert cache.get("k") is None

    def test_the_oldest_entry_is_evicted_first(self):
        cache = AnswerCache[str](max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        assert cache.get("a") is None
        assert cache.get("c") == "3"

    def test_reading_an_entry_protects_it_from_eviction(self):
        cache = AnswerCache[str](max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")
        cache.put("c", "3")
        assert cache.get("a") == "1"
        assert cache.get("b") is None

    def test_hit_rate_is_tracked_for_the_dashboard(self):
        cache = AnswerCache[str]()
        cache.put("k", "v")
        cache.get("k")
        cache.get("absent")
        assert cache.hit_rate == pytest.approx(0.5)

    def test_hit_rate_is_zero_before_any_lookup(self):
        assert AnswerCache[str]().hit_rate == 0.0


class TestDriftRules:
    def test_the_first_snapshot_is_not_drift(self):
        """Otherwise every model looks newly added the moment the service starts."""
        assert diff_models("groq", [], ["a", "b"]) == []

    def test_a_removed_model_is_reported(self):
        drifts = diff_models("groq", ["a", "b"], ["a"])
        assert [d.kind for d in drifts] == ["model_removed"]
        assert drifts[0].model == "b"

    def test_a_removed_model_is_breaking(self):
        assert diff_models("groq", ["a"], [])[0].is_breaking is True

    def test_an_added_model_is_reported_but_not_breaking(self):
        """New models are logged for review and never auto-adopted (CLAUDE.md section 9)."""
        drifts = diff_models("groq", ["a"], ["a", "b"])
        assert drifts[0].kind == "model_added"
        assert drifts[0].is_breaking is False

    def test_no_change_produces_no_drift(self):
        assert diff_models("groq", ["a", "b"], ["b", "a"]) == []

    def test_a_provider_going_down_is_reported(self):
        drifts = diff_reachability("groq", True, False)
        assert drifts[0].kind == "provider_unreachable"
        assert drifts[0].is_breaking is True

    def test_a_provider_recovering_is_reported(self):
        assert diff_reachability("groq", False, True)[0].kind == "provider_recovered"

    def test_a_stable_provider_produces_no_drift(self):
        assert diff_reachability("groq", True, True) == []


class TestDriftImpact:
    def test_only_models_we_route_to_are_flagged(self):
        """A provider deleting a model nobody uses is noise, not an incident."""
        drifts = diff_models("groq", ["used", "unused"], ["used"])
        affected = affected_tier_models(drifts, {"groq": ["used"]})
        assert affected == []

    def test_a_removed_tier_model_is_flagged(self):
        drifts = diff_models("groq", ["used"], [])
        affected = affected_tier_models(drifts, {"groq": ["used"]})
        assert affected == ["groq/used"]

    def test_an_unreachable_provider_flags_all_its_tier_models(self):
        drifts = diff_reachability("groq", True, False)
        affected = affected_tier_models(drifts, {"groq": ["a", "b"], "gemini": ["c"]})
        assert affected == ["groq/a", "groq/b"]


class StubProvider(Provider):
    name = "stub"

    def __init__(self, models: list[str] | None = None, fail: bool = False) -> None:
        super().__init__("key")
        self._models = models or []
        self._fail = fail

    async def list_models(self) -> list[str]:
        if self._fail:
            raise ProviderUnavailableError(self.name, "down")
        return list(self._models)

    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> Completion:
        raise NotImplementedError


class TestPoller:
    async def test_the_first_poll_records_the_catalog_without_drift(self):
        poller = CatalogPoller({"stub": StubProvider(["a", "b"])})
        drifts = await poller.poll_once()
        assert drifts == []
        assert poller.health["stub"].models == ["a", "b"]

    async def test_a_second_poll_detects_a_removal(self):
        provider = StubProvider(["a", "b"])
        poller = CatalogPoller({"stub": provider})
        await poller.poll_once()
        provider._models = ["a"]  # noqa: SLF001
        drifts = await poller.poll_once()
        assert [d.kind for d in drifts] == ["model_removed"]

    async def test_an_unreachable_provider_is_marked_unhealthy(self):
        poller = CatalogPoller({"stub": StubProvider(fail=True)})
        await poller.poll_once()
        assert poller.health["stub"].reachable is False
        assert poller.health["stub"].last_error

    async def test_three_consecutive_failures_trip_the_circuit(self):
        poller = CatalogPoller({"stub": StubProvider(fail=True)})
        for _ in range(3):
            await poller.poll_once()
        assert poller.health["stub"].is_circuit_broken is True

    async def test_recovery_clears_the_failure_streak(self):
        provider = StubProvider(fail=True)
        poller = CatalogPoller({"stub": provider})
        await poller.poll_once()
        provider._fail = False  # noqa: SLF001
        await poller.poll_once()
        assert poller.health["stub"].reachable is True
        assert poller.health["stub"].consecutive_failures == 0
