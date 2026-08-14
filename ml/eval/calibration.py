"""Calibration: does the classifier know when it is guessing?

A classifier that is 90% confident should be right about 90% of the time. One
that is confidently wrong is worse than one that is uncertain and wrong, because
the escalation loop has no way to tell the difference — a confident T1 prediction
and an unsure one produce exactly the same routing decision.

This measures predicted confidence against observed correctness, bucketed. The
gap between them is the calibration error, and it is the number that says whether
confidence could safely be used as a routing input in future work (it is not used
as one today).

Validation split only. The test split belongs to run_eval.py.
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features import EmbeddingClient, build_handcrafted_matrix  # noqa: E402
from train_classifier import load_keys  # noqa: E402
from training_data import (  # noqa: E402
    TIERS,
    labels_to_array,
    load_labelled,
    make_splits,
    restrict_to_embedded,
)

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[2] / "backend" / "artifacts"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

BUCKETS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def build_matrix(rows: list, feature_set: str, embedder: EmbeddingClient) -> np.ndarray:
    handcrafted = build_handcrafted_matrix([r.to_sample() for r in rows])
    if feature_set == "handcrafted":
        return handcrafted
    embeddings = embedder.embed_many([r.text for r in rows])
    if feature_set == "embedding":
        return embeddings
    return np.hstack([handcrafted, embeddings])


def main() -> None:
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    rows = restrict_to_embedded(load_labelled())
    splits = make_splits(rows)
    y_val = labels_to_array(splits.val)

    embedder = EmbeddingClient(load_keys()["GOOGLE_AI_STUDIO_API_KEY"])
    x_val = build_matrix(splits.val, metadata["feature_set"], embedder)

    model = joblib.load(ARTIFACT_DIR / "classifier.joblib")
    probabilities = model.predict_proba(x_val)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_val

    print(f"{'confidence':<16}{'n':>6}{'mean conf':>12}{'accuracy':>11}{'gap':>9}")
    print("-" * 54)

    buckets = []
    for low, high in BUCKETS:
        mask = (confidence >= low) & (confidence < high)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        buckets.append(
            {
                "range": f"{low:.1f}-{min(high, 1.0):.1f}",
                "n": n,
                "mean_confidence": round(mean_confidence, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(accuracy - mean_confidence, 4),
            }
        )
        print(
            f"{low:.1f}-{min(high, 1.0):.1f}{'':<10}{n:>6}{mean_confidence:>12.3f}"
            f"{accuracy:>11.3f}{accuracy - mean_confidence:>+9.3f}"
        )

    # Expected calibration error: the average gap, weighted by bucket size.
    total = sum(b["n"] for b in buckets)
    ece = sum(b["n"] * abs(b["gap"]) for b in buckets) / total if total else 0.0
    print(f"\nexpected calibration error: {ece:.4f}")

    # Under-prediction is what the verifier has to catch, so it is reported
    # separately from raw error — the two have very different consequences.
    under = float((predicted < y_val).mean())
    over = float((predicted > y_val).mean())
    print(f"under-predicted (verifier escalates): {under * 100:.1f}%")
    print(f"over-predicted (paid for more than needed): {over * 100:.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "calibration.json").write_text(
        json.dumps(
            {
                "buckets": buckets,
                "expected_calibration_error": round(ece, 4),
                "under_predicted_rate": round(under, 4),
                "over_predicted_rate": round(over, 4),
                "split": "validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    fig.patch.set_facecolor("#F2F4F1")
    ax.set_facecolor("#F2F4F1")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1, label="perfect")
    ax.plot(
        [b["mean_confidence"] for b in buckets],
        [b["accuracy"] for b in buckets],
        marker="o",
        color="#4B7B6E",
        label="observed",
    )
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Classifier calibration — validation", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"wrote {RESULTS_DIR / 'calibration.json'} and calibration.png")


if __name__ == "__main__":
    main()
