"""Loading and splitting the labelled dataset.

Shared by training, ablation, and final evaluation so that all three see exactly
the same rows in exactly the same splits. If each script built its own split, an
ablation comparison would be measuring split variance as much as feature value,
and the "touch the test set once" rule would be unenforceable.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from features import Sample

LABELS_PATH = pathlib.Path(__file__).parent / "data" / "labeled" / "labels.jsonl"

# Fixed so every script and every re-run produces identical splits.
RANDOM_SEED = 20260814

TIERS = ("T1", "T2", "T3")
TIER_TO_INDEX = {tier: i for i, tier in enumerate(TIERS)}


@dataclass(frozen=True, slots=True)
class LabelledQuestion:
    id: str
    source: str
    subject: str
    text: str
    num_choices: int
    label: str

    def to_sample(self) -> Sample:
        return Sample(text=self.text, subject=self.subject, num_choices=self.num_choices)


def load_labelled(path: pathlib.Path = LABELS_PATH) -> list[LabelledQuestion]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python label_empirical.py` first to generate labels."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            LabelledQuestion(
                id=record["id"],
                source=record["source"],
                subject=record["subject"],
                text=record["text"],
                num_choices=record["num_choices"],
                label=record["label"],
            )
        )
    return rows


@dataclass
class Splits:
    """A 70/15/15 stratified split, held identically across every script."""

    train: list[LabelledQuestion]
    val: list[LabelledQuestion]
    test: list[LabelledQuestion]

    def summary(self) -> str:
        return f"train={len(self.train)} val={len(self.val)} test={len(self.test)}"


def make_splits(rows: list[LabelledQuestion]) -> Splits:
    """Split 70/15/15, stratified by label.

    Stratified because the classes are heavily imbalanced — an unstratified 15%
    test split could easily contain almost no T3 examples, making the metric that
    matters most unmeasurable.
    """
    labels = [r.label for r in rows]
    train, holdout = train_test_split(
        rows, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    holdout_labels = [r.label for r in holdout]
    val, test = train_test_split(
        holdout, test_size=0.50, stratify=holdout_labels, random_state=RANDOM_SEED
    )
    return Splits(train=train, val=val, test=test)


def labels_to_array(rows: list[LabelledQuestion]) -> np.ndarray:
    return np.array([TIER_TO_INDEX[r.label] for r in rows], dtype=np.int64)


def majority_baseline_accuracy(train: list[LabelledQuestion], evaluate_on: list[LabelledQuestion]) -> float:
    """Accuracy of always predicting the most common training label.

    Reported alongside every model score. With a dominant class, a model can look
    strong while having learned nothing, and this is the number that exposes that.
    """
    counts: dict[str, int] = {}
    for row in train:
        counts[row.label] = counts.get(row.label, 0) + 1
    majority = max(counts, key=lambda k: counts[k])
    return sum(r.label == majority for r in evaluate_on) / len(evaluate_on)
