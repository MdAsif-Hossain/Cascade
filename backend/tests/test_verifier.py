"""Verifier: cheap heuristics, judge scoring, and the cross-provider constraint."""

from __future__ import annotations

import pytest
import respx

from app.core.config import Settings
from app.routing.router import Router
from app.routing.tiers import candidates
from app.verification import heuristics
from app.verification.verifier import (
    PASS_THRESHOLD,
    Verifier,
    build_judge_prompt,
    parse_score,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def settings(**overrides: str) -> Settings:
    defaults = {
        "groq_api_key": "g-key",
        "google_ai_studio_api_key": "m-key",
        "openrouter_api_key": "o-key",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[arg-type]


def openai_ok(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def gemini_ok(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
    }


class TestHeuristics:
    def test_a_normal_answer_passes(self):
        assert heuristics.check("The derivative of x squared is 2x.").passed

    def test_an_empty_answer_fails(self):
        assert heuristics.check("").failed

    def test_a_whitespace_answer_fails(self):
        assert heuristics.check("   \n  ").failed

    @pytest.mark.parametrize(
        "answer",
        [
            "I can't help with that.",
            "I cannot assist with this request.",
            "As an AI language model, I cannot answer that.",
            "I'm sorry, but I am unable to help.",
            "I don't have enough information to answer.",
        ],
    )
    def test_refusals_fail(self, answer):
        assert heuristics.check(answer).failed

    def test_a_truncated_answer_fails(self):
        """The provider told us it ran out of room, so the answer is incomplete."""
        result = heuristics.check("The first step is to", finish_reason="length")
        assert result.failed
        assert "truncated" in (result.reason or "")

    def test_an_absurdly_long_answer_fails(self):
        assert heuristics.check("word " * 5000).failed

    def test_a_very_short_but_valid_answer_passes(self):
        """'4' is a complete answer to 'what is 2+2' and must not be rejected."""
        assert heuristics.check("4").passed

    def test_the_word_cannot_alone_is_not_a_refusal(self):
        """Otherwise any answer explaining why something cannot happen is rejected."""
        answer = "A triangle cannot have two right angles, because the angles sum to 180 degrees."
        assert heuristics.check(answer).passed


class TestScoreParsing:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("0.9", 0.9),
            ("0.85\nThe answer is correct.", 0.85),
            ("1.0 — fully correct", 1.0),
            ("0", 0.0),
            ("1", 1.0),
        ],
    )
    def test_valid_scores_parse(self, reply, expected):
        assert parse_score(reply) == pytest.approx(expected)

    def test_a_reply_without_a_number_is_unparseable(self):
        assert parse_score("This answer looks good to me.") is None

    def test_a_score_outside_the_range_is_rejected_not_clamped(self):
        """A judge answering '85' meant percent; guessing would silently rescale."""
        assert parse_score("85") is None


class TestJudgePrompt:
    def test_the_prompt_carries_both_question_and_answer(self):
        messages = build_judge_prompt("What is 2+2?", "It is 4.")
        body = " ".join(m.content for m in messages)
        assert "What is 2+2?" in body
        assert "It is 4." in body

    def test_the_prompt_opens_with_a_system_instruction(self):
        assert build_judge_prompt("q", "a")[0].role == "system"


class TestCrossProviderConstraint:
    """CLAUDE.md section 8: the judge must never share a provider with the answerer."""

    @respx.mock
    async def test_the_judge_does_not_run_on_the_answering_provider(self):
        groq = respx.post(GROQ_URL).respond(200, json=openai_ok("0.9"))
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.9"))

        verdict = await Verifier(Router(settings())).verify(
            "What is 2+2?", "It is 4.", answered_by_provider="groq"
        )

        assert not groq.called, "judge ran on the same provider that produced the answer"
        assert verdict.judge_provider != "groq"

    @respx.mock
    async def test_a_gemini_answer_is_judged_elsewhere(self):
        gemini = respx.post(gemini_url(candidates("T1")[1].model)).respond(
            200, json=gemini_ok("0.9")
        )
        respx.post(GROQ_URL).respond(200, json=openai_ok("0.9"))

        verdict = await Verifier(Router(settings())).verify(
            "What is 2+2?", "It is 4.", answered_by_provider="gemini"
        )

        assert not gemini.called
        assert verdict.judge_provider != "gemini"

    @respx.mock
    async def test_no_alternative_provider_means_accepted_unverified(self):
        """Failing closed here would escalate everything during an outage."""
        verifier = Verifier(Router(settings(google_ai_studio_api_key="", openrouter_api_key="")))
        verdict = await verifier.verify("q", "a plausible answer", answered_by_provider="groq")
        assert verdict.passed is True
        assert verdict.stage == "judge_unavailable"


class TestVerdicts:
    @respx.mock
    async def test_a_high_score_passes(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.95"))
        verdict = await Verifier(Router(settings())).verify(
            "q", "a good answer", answered_by_provider="groq"
        )
        assert verdict.passed is True
        assert verdict.should_escalate is False

    @respx.mock
    async def test_a_low_score_triggers_escalation(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.3"))
        verdict = await Verifier(Router(settings())).verify(
            "q", "a poor answer", answered_by_provider="groq"
        )
        assert verdict.passed is False
        assert verdict.should_escalate is True

    @respx.mock
    async def test_the_threshold_boundary_passes(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(
            200, json=gemini_ok(str(PASS_THRESHOLD))
        )
        verdict = await Verifier(Router(settings())).verify(
            "q", "a borderline answer", answered_by_provider="groq"
        )
        assert verdict.passed is True

    async def test_a_refusal_never_reaches_the_judge(self):
        """The cheap stage exists to avoid paying for a judge call it can decide alone."""
        verdict = await Verifier(Router(settings())).verify(
            "q", "I can't help with that.", answered_by_provider="groq"
        )
        assert verdict.stage == "heuristic"
        assert verdict.should_escalate is True

    @respx.mock
    async def test_an_unparseable_judge_reply_accepts_rather_than_escalates(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(
            200, json=gemini_ok("Looks fine to me.")
        )
        verdict = await Verifier(Router(settings())).verify(
            "q", "an answer", answered_by_provider="groq"
        )
        assert verdict.stage == "judge_unparseable"
        assert verdict.passed is True

    @respx.mock
    async def test_the_verdict_records_which_model_judged(self):
        respx.post(gemini_url(candidates("T1")[1].model)).respond(200, json=gemini_ok("0.9"))
        verdict = await Verifier(Router(settings())).verify(
            "q", "an answer", answered_by_provider="groq"
        )
        assert verdict.judge_model
        assert verdict.judge_provider
