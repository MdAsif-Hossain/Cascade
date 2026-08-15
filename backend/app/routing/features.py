"""Handcrafted question features — the single implementation used everywhere.

This lives in the backend rather than in ``ml/`` because both training and
inference must compute features identically. Two copies of this logic would drift
the moment one is edited, and the failure would be silent: the model would keep
returning confident predictions computed from features that no longer mean what
it learned. ``ml/features.py`` imports from here rather than redefining.

Feature order is part of the model contract. Appending is safe; reordering or
removing invalidates every persisted classifier, which is why the names are
exported alongside the values and recorded in the artifact's metadata.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

SUBJECTS: tuple[str, ...] = ("math", "science", "history", "language", "general")
QUESTION_WORDS: tuple[str, ...] = ("what", "why", "how", "which", "who", "when", "where")

_MATH_OPERATORS = re.compile(r"[+\-*/=<>%^]|\b(?:times|divided|per cent|percent)\b")
_DIGIT = re.compile(r"\d")

FEATURE_NAMES: tuple[str, ...] = (
    "log_char_count",
    "log_word_count",
    "mean_word_length",
    "digit_density",
    "log_digit_count",
    "log_math_operator_count",
    "has_math_operator",
    "log_sentence_count",
    "log_comma_count",
    # `num_choices` was here and has been removed. It is a property of the
    # benchmark, not of a question: 76% of training rows carried 4 options while
    # a student's question always has 0. The model learned to read it, so every
    # real question arrived looking like the free-form GSM8K subset and routed to
    # T2 regardless of content. A feature that cannot be computed the same way at
    # training and inference is worse than no feature at all.
    *(f"qword_{w}" for w in QUESTION_WORDS),
    "qword_other",
    *(f"subject_{s}" for s in SUBJECTS),
)


@dataclass(frozen=True, slots=True)
class Sample:
    """A question in the minimal form feature extraction needs.

    ``num_choices`` is retained on the sample because the labelling pipeline
    knows it, but it is deliberately **not** a feature — see FEATURE_NAMES.
    """

    text: str
    subject: str = "general"
    num_choices: int = 0


def handcrafted_features(sample: Sample) -> list[float]:
    """Surface statistics of a question.

    Counts are log-scaled because length is heavy-tailed: one 500-word word
    problem would otherwise dominate a linear model's view of every other
    feature.
    """
    text = sample.text
    words = text.split()
    digits = len(_DIGIT.findall(text))
    operators = len(_MATH_OPERATORS.findall(text.lower()))

    first_word = words[0].lower() if words else ""
    qword_flags = [1.0 if first_word.startswith(w) else 0.0 for w in QUESTION_WORDS]
    qword_flags.append(0.0 if any(qword_flags) else 1.0)

    subject_flags = [1.0 if sample.subject == s else 0.0 for s in SUBJECTS]

    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        digits / len(text) if text else 0.0,
        math.log1p(digits),
        math.log1p(operators),
        1.0 if operators else 0.0,
        math.log1p(text.count(".") + text.count("?") + text.count("!")),
        math.log1p(text.count(",")),
        *qword_flags,
        *subject_flags,
    ]
