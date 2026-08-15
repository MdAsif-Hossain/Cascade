"""Tier definitions and counterfactual cost arithmetic."""

from __future__ import annotations

import pytest

from app.routing.tiers import (
    TIER_ORDER,
    TIERS,
    ModelSpec,
    baseline_cost_usd,
    baseline_spec,
    candidates,
    next_tier,
)


class TestTierStructure:
    def test_every_tier_is_defined(self):
        assert set(TIERS) == {"T1", "T2", "T3"}

    @pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
    def test_every_tier_has_more_than_one_provider(self, tier):
        """A tier served by a single provider cannot fail over when that provider dies."""
        providers = {spec.provider for spec in candidates(tier)}
        assert len(providers) > 1, f"{tier} would be unroutable if {providers} went down"

    @pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
    def test_no_model_is_listed_twice_within_a_tier(self, tier):
        models = [(s.provider, s.model) for s in candidates(tier)]
        assert len(models) == len(set(models))

    def test_no_model_appears_in_two_tiers(self):
        """A model in two bands makes an escalation potentially a no-op."""
        seen: dict[tuple[str, str], str] = {}
        for tier, specs in TIERS.items():
            for spec in specs:
                key = (spec.provider, spec.model)
                assert key not in seen, f"{key} is in both {seen.get(key)} and {tier}"
                seen[key] = tier

    @pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
    def test_every_model_has_a_positive_price(self, tier):
        """A zero price would silently erase that call from the cost comparison."""
        for spec in candidates(tier):
            assert spec.prompt_usd_per_mtok > 0
            assert spec.completion_usd_per_mtok > 0

    @pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
    def test_every_model_records_its_price_reference(self, tier):
        """Each price must name the model it came from, so a reader can check it."""
        for spec in candidates(tier):
            assert spec.price_reference


class TestEscalation:
    def test_tiers_escalate_upward(self):
        assert next_tier("T1") == "T2"
        assert next_tier("T2") == "T3"

    def test_top_tier_has_nowhere_to_escalate(self):
        assert next_tier("T3") is None

    def test_tier_order_matches_the_tier_table(self):
        assert set(TIER_ORDER) == set(TIERS)


class TestCost:
    def test_cost_is_per_million_tokens(self):
        spec = ModelSpec("test", "m", 1.0, 2.0, "ref")
        assert spec.estimated_cost_usd(1_000_000, 0) == pytest.approx(1.0)
        assert spec.estimated_cost_usd(0, 1_000_000) == pytest.approx(2.0)

    def test_prompt_and_completion_are_priced_separately(self):
        spec = ModelSpec("test", "m", 1.0, 2.0, "ref")
        assert spec.estimated_cost_usd(1_000_000, 1_000_000) == pytest.approx(3.0)

    def test_a_free_call_still_has_a_counterfactual_cost(self):
        """The whole point: no money moves, but the avoided cost is quantified."""
        assert baseline_cost_usd(100, 200) > 0

    def test_baseline_is_the_first_model_in_the_top_tier(self):
        assert baseline_spec() == candidates("T3")[0]

    def test_baseline_is_the_most_expensive_choice(self):
        """Routing must be compared against the priciest option, or the saving is overstated."""
        baseline = baseline_spec()
        cheapest_t1 = min(candidates("T1"), key=lambda s: s.estimated_cost_usd(1000, 1000))
        assert baseline.estimated_cost_usd(1000, 1000) > cheapest_t1.estimated_cost_usd(1000, 1000)

    def test_zero_tokens_cost_nothing(self):
        assert baseline_cost_usd(0, 0) == 0.0
