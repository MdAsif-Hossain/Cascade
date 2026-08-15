"""Grading rules for empirical labelling.

Grading accuracy caps label quality, so these cases are drawn from answer shapes
models actually produce rather than from what a well-behaved model would produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ml/ sits outside the backend package but its grading rules are what the labels
# depend on, so they are tested alongside everything else rather than untested.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))

from grading import (  # noqa: E402
    extract_final_number,
    grade,
    grade_multiple_choice,
    grade_numeric,
)

CHOICES = ("Sunlight", "Water", "Nitrogen", "Iron")
CORRECT = "Sunlight"


class TestMultipleChoiceLetters:
    @pytest.mark.parametrize(
        "answer",
        ["A", "A.", "(A)", " a ", "**A**", "A)", "Answer: A", "The answer is A."],
    )
    def test_letter_forms_models_actually_emit(self, answer):
        assert grade_multiple_choice(answer, CHOICES, CORRECT) is True

    @pytest.mark.parametrize("answer", ["B", "(C)", "Answer: D"])
    def test_wrong_letters_are_wrong(self, answer):
        assert grade_multiple_choice(answer, CHOICES, CORRECT) is False

    def test_a_letter_beyond_the_option_count_is_not_credited(self):
        assert grade_multiple_choice("E", CHOICES, CORRECT) is False


class TestMultipleChoiceText:
    def test_naming_the_correct_option_counts(self):
        """Ignoring the letter-only instruction but answering correctly is still correct."""
        assert grade_multiple_choice("The answer is sunlight.", CHOICES, CORRECT) is True

    def test_listing_every_option_is_not_credited(self):
        """Otherwise a model that hedges across all options scores full marks."""
        answer = "It could be Sunlight, Water, Nitrogen or Iron."
        assert grade_multiple_choice(answer, CHOICES, CORRECT) is False

    def test_naming_only_a_wrong_option_is_wrong(self):
        assert grade_multiple_choice("Definitely water.", CHOICES, CORRECT) is False

    def test_an_empty_answer_is_wrong(self):
        assert grade_multiple_choice("", CHOICES, CORRECT) is False

    def test_an_unrelated_answer_is_wrong(self):
        assert grade_multiple_choice("I cannot help with that.", CHOICES, CORRECT) is False


class TestNumericExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("18", 18.0),
            ("The answer is 18", 18.0),
            ("18.00", 18.0),
            ("1,234", 1234.0),
            ("$72", 72.0),
            ("**42**", 42.0),
            ("First 10, then 5, so 15", 15.0),
            ("-7", -7.0),
        ],
    )
    def test_the_last_number_is_taken_as_the_answer(self, text, expected):
        assert extract_final_number(text) == pytest.approx(expected)

    def test_text_without_a_number_yields_none(self):
        assert extract_final_number("I do not know.") is None


class TestNumericGrading:
    def test_formatting_variants_all_match(self):
        for answer in ("18", "18.0", "18.00", "The answer is 18."):
            assert grade_numeric(answer, "18") is True

    def test_a_different_number_is_wrong(self):
        assert grade_numeric("19", "18") is False

    def test_working_shown_before_the_final_answer_still_grades(self):
        assert grade_numeric("16 - 3 - 4 = 9, then 9 * 2 = 18", "18") is True

    def test_a_missing_number_is_wrong(self):
        assert grade_numeric("I cannot solve this.", "18") is False


class TestDispatch:
    def test_choices_present_means_multiple_choice(self):
        assert grade("A", choices=CHOICES, correct=CORRECT) is True

    def test_no_choices_means_numeric(self):
        assert grade("18", choices=(), correct="18") is True
