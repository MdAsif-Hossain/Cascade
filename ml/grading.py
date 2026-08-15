"""Automatic grading of model answers against known-correct answers.

Grading accuracy sets a ceiling on label quality. A grader that marks a correct
answer wrong pushes an easy question up a tier, and the classifier then learns
that noise as if it were difficulty. So the rules here are deliberately strict
about ambiguity: when it is not clear what the model chose, the answer is graded
wrong rather than guessed at. Being wrong in a consistent direction is
recoverable; being wrong at random is not.
"""

from __future__ import annotations

import re

# "A", "A.", "(A)", "**A**", "Answer: A" — the shapes models actually emit when
# asked for a single letter.
_LETTER_PATTERNS = (
    re.compile(r"^\s*[*_(\[]*([A-Ea-e])[*_)\].:,]*\s*$"),
    re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*[*_(\[]*([A-Ea-e])\b", re.IGNORECASE),
    re.compile(r"^\s*[*_(\[]*([A-Ea-e])[*_)\].:,]\s", re.MULTILINE),
)

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_choice_letter(text: str, num_choices: int) -> int | None:
    """Return the zero-based option index the model chose, or None if unclear."""
    stripped = text.strip()
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            index = ord(match.group(1).upper()) - ord("A")
            if 0 <= index < num_choices:
                return index
    return None


def grade_multiple_choice(answer_text: str, choices: tuple[str, ...], correct: str) -> bool:
    """True when the model picked the correct option.

    Two ways to be right: naming the letter, or stating the option text. The
    second is accepted because a model ignoring the "letter only" instruction and
    answering correctly in prose has still answered the question correctly, and
    counting that as a failure would inflate the difficulty label.
    """
    if not answer_text.strip():
        return False

    index = extract_choice_letter(answer_text, len(choices))
    if index is not None:
        return choices[index].strip().lower() == correct.strip().lower()

    normalised = " ".join(answer_text.lower().split())
    correct_normalised = " ".join(correct.lower().split())
    if correct_normalised not in normalised:
        return False

    # Only credit a text match when no other option also appears, otherwise a
    # model listing every option would be scored correct.
    others = [
        c for c in choices if c.strip().lower() != correct.strip().lower()
    ]
    return not any(" ".join(o.lower().split()) in normalised for o in others)


def extract_final_number(text: str) -> float | None:
    """Return the last number in the text, which is where a final answer lands."""
    matches = _NUMBER.findall(text.replace("$", "").replace("**", ""))
    for raw in reversed(matches):
        cleaned = raw.replace(",", "").rstrip(".")
        if not cleaned or cleaned == "-":
            continue
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def grade_numeric(answer_text: str, correct: str) -> bool:
    """True when the model's final number matches the expected one.

    Compared with a small tolerance rather than by string, so that 18, 18.0 and
    18.00 all count. GSM8K answers are integers, so the tolerance only absorbs
    formatting, never a genuinely different answer.
    """
    predicted = extract_final_number(answer_text)
    if predicted is None:
        return False
    try:
        expected = float(correct)
    except ValueError:
        return False
    return abs(predicted - expected) < 1e-4


def grade(answer_text: str, *, choices: tuple[str, ...], correct: str) -> bool:
    """Grade an answer, dispatching on whether the question was multiple choice."""
    if choices:
        return grade_multiple_choice(answer_text, choices, correct)
    return grade_numeric(answer_text, correct)
