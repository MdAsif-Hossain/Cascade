"""Load and normalise the public question pool used for empirical labelling.

Five datasets with known-correct answers are pulled from the HuggingFace
datasets-server over plain HTTP. That endpoint is used rather than the
``datasets`` library because the library pulls in pyarrow and a large dependency
tree for what is, here, a few thousand rows of JSON — and every dependency added
is one the 512 MB backend might later inherit.

Each dataset has its own shape. They are normalised into a single ``Question``
so the labelling script never branches on where a question came from, and so
grading has exactly two cases (multiple choice, or a numeric answer) instead of
five.

Raw responses are cached under ``ml/data/raw/`` (gitignored) so a re-run costs
nothing and the pool stays identical across runs — the labels would not be
reproducible otherwise.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

RAW_DIR = pathlib.Path(__file__).parent / "data" / "raw"
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

Subject = Literal["math", "science", "history", "language", "general"]


@dataclass(frozen=True, slots=True)
class Question:
    """One question with a known-correct answer.

    ``choices`` is empty for free-form numeric questions (GSM8K). ``answer`` holds
    the correct option text for multiple choice, or the final number otherwise.
    """

    id: str
    source: str
    subject: Subject
    text: str
    answer: str
    choices: tuple[str, ...] = field(default=())

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)

    def prompt(self) -> str:
        """The question as put to a model.

        Multiple choice questions are presented with their options so that a wrong
        answer means the model failed the question, not that it failed to guess
        which options existed. Both forms demand a terse final answer, because a
        model that rambles is harder to grade automatically and that grading noise
        would land in the labels.
        """
        if self.is_multiple_choice:
            options = "\n".join(
                f"{chr(65 + i)}. {choice}" for i, choice in enumerate(self.choices)
            )
            return (
                f"{self.text}\n\n{options}\n\n"
                "Answer with the single letter of the correct option, nothing else."
            )
        return (
            f"{self.text}\n\n"
            "Give only the final numeric answer on the last line, with no units or working."
        )


def _cache_path(dataset: str, config: str, split: str) -> pathlib.Path:
    safe = f"{dataset}__{config}__{split}".replace("/", "_")
    return RAW_DIR / f"{safe}.json"


def fetch_rows(
    dataset: str, config: str, split: str, limit: int, *, client: httpx.Client
) -> list[dict[str, Any]]:
    """Fetch rows, caching to disk so repeat runs never re-download."""
    cached = _cache_path(dataset, config, split)
    if cached.exists():
        rows: list[dict[str, Any]] = json.loads(cached.read_text(encoding="utf-8"))
        if len(rows) >= limit:
            return rows[:limit]

    collected: list[dict[str, Any]] = []
    offset = 0
    while len(collected) < limit:
        response = client.get(
            DATASETS_SERVER,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(PAGE_SIZE, limit - len(collected)),
            },
        )
        response.raise_for_status()
        page = response.json()["rows"]
        if not page:
            break
        collected.extend(entry["row"] for entry in page)
        offset += len(page)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(collected), encoding="utf-8")
    return collected[:limit]


def _from_arc(row: dict[str, Any], source: str) -> Question | None:
    """ARC and OpenBookQA share a shape: choices as parallel text/label lists."""
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    key = row.get("answerKey")
    if key not in labels:
        # A handful of rows carry a malformed answer key. Dropping them is safer
        # than guessing, since a wrong ground truth would poison a label.
        return None
    stem = row.get("question") or row.get("question_stem", "")
    return Question(
        id=str(row.get("id", "")),
        source=source,
        subject="science",
        text=str(stem),
        answer=str(texts[labels.index(key)]),
        choices=tuple(str(t) for t in texts),
    )


def _from_sciq(row: dict[str, Any], index: int) -> Question | None:
    """SciQ stores the correct answer and three distractors as separate fields.

    Options are ordered deterministically rather than shuffled so that a re-run
    produces identical prompts, and so the correct answer is not always first.
    """
    correct = str(row["correct_answer"])
    distractors = [str(row[f"distractor{i}"]) for i in (1, 2, 3)]
    options = sorted({correct, *distractors})
    if len(options) < 2:
        return None
    return Question(
        id=f"sciq-{index}",
        source="sciq",
        subject="science",
        text=str(row["question"]),
        answer=correct,
        choices=tuple(options),
    )


def _from_gsm8k(row: dict[str, Any], index: int) -> Question | None:
    """GSM8K answers end with '#### <number>' after the worked solution."""
    match = re.search(r"####\s*([-\d,\.]+)", str(row["answer"]))
    if not match:
        return None
    return Question(
        id=f"gsm8k-{index}",
        source="gsm8k",
        subject="math",
        text=str(row["question"]),
        answer=match.group(1).replace(",", "").strip(),
    )


# Counts reach roughly 1,200 questions, weighted towards the harder sources.
#
# A 60-question pilot on 2026-08-14 measured how often the T1 model answers each
# source correctly: ARC-Easy 100%, GSM8K 92%, SciQ 83%, ARC-Challenge 73%,
# OpenBookQA 71%. An even split therefore produces a pool that is ~86% T1, which
# leaves almost no T2/T3 examples to learn from. Shifting weight towards
# ARC-Challenge and OpenBookQA buys harder questions without reaching outside
# the five datasets the project specified.
#
# This changes the class balance, not the labels: each question's tier is still
# whatever the models actually achieve on it.
POOL_PLAN: tuple[tuple[str, str, str, str, int], ...] = (
    ("allenai/ai2_arc", "ARC-Easy", "test", "arc-easy", 120),
    ("allenai/ai2_arc", "ARC-Challenge", "test", "arc-challenge", 400),
    ("openai/gsm8k", "main", "test", "gsm8k", 300),
    ("allenai/sciq", "default", "test", "sciq", 130),
    ("allenai/openbookqa", "main", "test", "openbookqa", 300),
)


def load_pool() -> list[Question]:
    """Build the full labelling pool, in a stable order."""
    questions: list[Question] = []
    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        for dataset, config, split, source, count in POOL_PLAN:
            rows = fetch_rows(dataset, config, split, count, client=client)
            for index, row in enumerate(rows):
                question: Question | None
                if source == "gsm8k":
                    question = _from_gsm8k(row, index)
                elif source == "sciq":
                    question = _from_sciq(row, index)
                else:
                    question = _from_arc(row, source)
                if question is not None and question.text.strip():
                    questions.append(question)
    return questions


if __name__ == "__main__":
    pool = load_pool()
    print(f"{len(pool)} questions")
    by_source: dict[str, int] = {}
    for q in pool:
        by_source[q.source] = by_source.get(q.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source:16s} {count:4d}")
    print(f"\nmultiple choice: {sum(q.is_multiple_choice for q in pool)}")
    print(f"numeric:         {sum(not q.is_multiple_choice for q in pool)}")
    print("\n--- sample prompts ---")
    for q in (pool[0], pool[-1]):
        print(f"\n[{q.source}] answer={q.answer!r}\n{q.prompt()[:300]}")
