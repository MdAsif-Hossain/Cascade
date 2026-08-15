"""Feature extraction — guarding the train/inference contract.

The classifier is trained in ``ml/`` and served in ``app/``. If a feature can be
computed one way during training and a different way at inference, the model
keeps returning confident predictions from inputs that no longer mean what it
learned. Nothing crashes; the routing just quietly goes wrong.

That happened once: ``num_choices`` was a feature, 76% of training rows had 4
options, and every real student question has 0 — so every question looked like
the free-form maths subset and routed to T2 regardless of content. These tests
exist so it cannot happen again unnoticed.
"""

from __future__ import annotations

import pytest

from app.routing.features import FEATURE_NAMES, Sample, handcrafted_features


class TestFeatureContract:
    def test_names_and_values_stay_the_same_length(self):
        """Metadata records the names; a mismatch means the report describes the wrong model."""
        assert len(handcrafted_features(Sample("What is 2+2?"))) == len(FEATURE_NAMES)

    def test_no_feature_depends_on_data_only_the_benchmark_has(self):
        """A student's question has no answer options, no source dataset, no label."""
        forbidden = {"num_choices", "source", "dataset", "answer", "label"}
        assert forbidden.isdisjoint(FEATURE_NAMES)

    def test_answer_options_do_not_change_the_features(self):
        """The regression guard: same question, different num_choices, same vector."""
        text = "Why is the sky blue?"
        as_free_form = handcrafted_features(Sample(text, "science", num_choices=0))
        as_multiple_choice = handcrafted_features(Sample(text, "science", num_choices=4))
        assert as_free_form == as_multiple_choice

    def test_features_are_finite(self):
        """A NaN or inf propagates into the model and produces nonsense silently."""
        for text in ("", "?", "a" * 3000, "123456", "  "):
            values = handcrafted_features(Sample(text))
            assert all(v == v and abs(v) != float("inf") for v in values)


class TestFeatureBehaviour:
    def test_subject_is_one_hot(self):
        values = handcrafted_features(Sample("q", "math"))
        subject_values = [
            values[FEATURE_NAMES.index(n)] for n in FEATURE_NAMES if n.startswith("subject_")
        ]
        assert sum(subject_values) == 1.0

    def test_an_unknown_subject_sets_no_flag(self):
        values = handcrafted_features(Sample("q", "astrology"))
        subject_values = [
            values[FEATURE_NAMES.index(n)] for n in FEATURE_NAMES if n.startswith("subject_")
        ]
        assert sum(subject_values) == 0.0

    def test_question_word_is_one_hot(self):
        values = handcrafted_features(Sample("Why is the sky blue?"))
        qword_values = [
            values[FEATURE_NAMES.index(n)] for n in FEATURE_NAMES if n.startswith("qword_")
        ]
        assert sum(qword_values) == 1.0

    def test_a_statement_falls_back_to_qword_other(self):
        values = handcrafted_features(Sample("Explain photosynthesis."))
        assert values[FEATURE_NAMES.index("qword_other")] == 1.0

    def test_math_operators_are_detected(self):
        values = handcrafted_features(Sample("What is 3 + 4?"))
        assert values[FEATURE_NAMES.index("has_math_operator")] == 1.0

    def test_prose_has_no_math_operator(self):
        values = handcrafted_features(Sample("Who wrote Hamlet?"))
        assert values[FEATURE_NAMES.index("has_math_operator")] == 0.0

    def test_digit_density_reflects_numerals(self):
        numeric = handcrafted_features(Sample("12345 67890"))
        prose = handcrafted_features(Sample("no numerals here at all"))
        index = FEATURE_NAMES.index("digit_density")
        assert numeric[index] > prose[index]

    def test_empty_text_does_not_divide_by_zero(self):
        assert handcrafted_features(Sample(""))[FEATURE_NAMES.index("digit_density")] == 0.0

    @pytest.mark.parametrize("text", ["short", "a much longer question with many more words"])
    def test_length_features_are_log_scaled(self, text):
        """Log scaling stops one long word problem dominating a linear model."""
        values = handcrafted_features(Sample(text))
        assert values[FEATURE_NAMES.index("log_char_count")] < len(text)
