"""Dataset normalisation.

These parsers decide what counts as the correct answer for every training
question. A bug here does not crash anything — it silently mislabels ground
truth, and the classifier then learns from it. Fixtures are inline rather than
fetched so the suite runs offline (CLAUDE.md section 13).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))

from datasets import Question, _from_arc, _from_gsm8k, _from_sciq  # noqa: E402

ARC_ROW = {
    "id": "Mercury_417466",
    "question": "Why is photosynthesis the foundation of most food webs?",
    "choices": {
        "text": [
            "Sunlight is the energy source.",
            "Most ecosystems are on land.",
            "CO2 is common.",
        ],
        "label": ["A", "B", "C"],
    },
    "answerKey": "A",
}

OPENBOOK_ROW = {
    "id": "8-343",
    "question_stem": "A person wants to save money. They should",
    "choices": {"text": ["spend more", "budget", "borrow"], "label": ["A", "B", "C"]},
    "answerKey": "B",
}

SCIQ_ROW = {
    "question": "Compounds capable of accepting electrons are called what?",
    "correct_answer": "oxidants",
    "distractor1": "antioxidants",
    "distractor2": "Oxygen",
    "distractor3": "residues",
}

GSM8K_ROW = {
    "question": "Janet has 16 eggs and uses 7. She sells the rest at $2 each. What does she earn?",
    "answer": "She has 16 - 7 = 9 left.\nShe makes 9 * 2 = 18 dollars.\n#### 18",
}


class TestArcAndOpenBook:
    def test_the_answer_key_maps_to_the_right_option_text(self):
        question = _from_arc(ARC_ROW, "arc-easy")
        assert question is not None
        assert question.answer == "Sunlight is the energy source."

    def test_a_non_first_answer_key_maps_correctly(self):
        """Guards against an off-by-one that would only show on some rows."""
        question = _from_arc(OPENBOOK_ROW, "openbookqa")
        assert question is not None
        assert question.answer == "budget"

    def test_every_option_is_kept(self):
        question = _from_arc(ARC_ROW, "arc-easy")
        assert question is not None
        assert len(question.choices) == 3

    def test_openbookqa_uses_question_stem(self):
        question = _from_arc(OPENBOOK_ROW, "openbookqa")
        assert question is not None
        assert question.text.startswith("A person wants")

    def test_a_malformed_answer_key_is_dropped_not_guessed(self):
        """A wrong ground truth would poison a label; dropping the row is safer."""
        broken = {**ARC_ROW, "answerKey": "Z"}
        assert _from_arc(broken, "arc-easy") is None

    def test_the_correct_answer_is_among_the_choices(self):
        question = _from_arc(ARC_ROW, "arc-easy")
        assert question is not None
        assert question.answer in question.choices


class TestSciQ:
    def test_the_correct_answer_is_preserved(self):
        question = _from_sciq(SCIQ_ROW, 0)
        assert question is not None
        assert question.answer == "oxidants"

    def test_distractors_become_options(self):
        question = _from_sciq(SCIQ_ROW, 0)
        assert question is not None
        assert len(question.choices) == 4
        assert "antioxidants" in question.choices

    def test_option_order_is_deterministic(self):
        """A re-run must produce identical prompts or labels stop being reproducible."""
        first = _from_sciq(SCIQ_ROW, 0)
        second = _from_sciq(SCIQ_ROW, 0)
        assert first is not None and second is not None
        assert first.choices == second.choices

    def test_the_correct_answer_is_not_always_first(self):
        """Sorted options mean a model cannot score by always picking A."""
        question = _from_sciq(SCIQ_ROW, 0)
        assert question is not None
        assert question.choices[0] != question.answer

    def test_a_row_whose_distractors_duplicate_the_answer_is_dropped(self):
        degenerate = {
            "question": "q",
            "correct_answer": "same",
            "distractor1": "same",
            "distractor2": "same",
            "distractor3": "same",
        }
        assert _from_sciq(degenerate, 0) is None


class TestGsm8k:
    def test_the_final_answer_is_extracted_from_the_marker(self):
        question = _from_gsm8k(GSM8K_ROW, 0)
        assert question is not None
        assert question.answer == "18"

    def test_thousands_separators_are_stripped(self):
        row = {"question": "q", "answer": "working\n#### 1,234"}
        question = _from_gsm8k(row, 0)
        assert question is not None
        assert question.answer == "1234"

    def test_a_row_without_the_marker_is_dropped(self):
        assert _from_gsm8k({"question": "q", "answer": "no marker here"}, 0) is None

    def test_gsm8k_questions_are_free_form_not_multiple_choice(self):
        question = _from_gsm8k(GSM8K_ROW, 0)
        assert question is not None
        assert question.is_multiple_choice is False

    def test_gsm8k_is_labelled_maths(self):
        question = _from_gsm8k(GSM8K_ROW, 0)
        assert question is not None
        assert question.subject == "math"


class TestPrompts:
    def test_multiple_choice_prompts_list_lettered_options(self):
        prompt = Question(
            id="x",
            source="s",
            subject="science",
            text="Pick one.",
            answer="A thing",
            choices=("A thing", "Another thing"),
        ).prompt()
        assert "A. A thing" in prompt
        assert "B. Another thing" in prompt

    def test_multiple_choice_prompts_ask_for_a_single_letter(self):
        prompt = Question(
            id="x",
            source="s",
            subject="science",
            text="Pick one.",
            answer="a",
            choices=("a", "b"),
        ).prompt()
        assert "letter" in prompt.lower()

    def test_numeric_prompts_ask_for_the_final_number_only(self):
        prompt = Question(
            id="x", source="gsm8k", subject="math", text="How many?", answer="18"
        ).prompt()
        assert "numeric" in prompt.lower()
        assert "A." not in prompt

    @pytest.mark.parametrize("count", [2, 3, 4, 5])
    def test_option_letters_stay_in_sequence(self, count):
        choices = tuple(f"option {i}" for i in range(count))
        prompt = Question(
            id="x", source="s", subject="science", text="q", answer=choices[0], choices=choices
        ).prompt()
        for i in range(count):
            assert f"{chr(65 + i)}. option {i}" in prompt
