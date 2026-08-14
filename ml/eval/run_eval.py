"""Final evaluation on the held-out test split. Run once.

The test split has been untouched through labelling, training, and ablation —
model selection happened entirely on validation. Running this more than once, or
tuning anything after seeing its output, turns the test set into a second
validation set and makes the headline numbers meaningless.

The script refuses to overwrite its own results for that reason. Deleting the
file to re-run is a deliberate act; doing it by accident is not possible.

Produces:
  * accuracy, macro-F1, per-class precision/recall
  * a confusion matrix, saved as PNG
  * routing quality vs. an always-T3 baseline
  * estimated cost reduction, counterfactual and labelled as such
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # No display on CI or a headless box.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))

from app.routing.tiers import TIERS as TIER_TABLE  # noqa: E402
from features import EmbeddingClient, build_handcrafted_matrix  # noqa: E402
from train_classifier import load_keys  # noqa: E402
from training_data import (  # noqa: E402
    TIERS,
    labels_to_array,
    load_labelled,
    majority_baseline_accuracy,
    make_splits,
)

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[2] / "backend" / "artifacts"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
TEST_RESULTS = RESULTS_DIR / "test_results.json"
CONFUSION_PNG = RESULTS_DIR / "confusion_matrix.png"

# Tier colours from the design system, so every chart in the report agrees with
# every badge in the UI (CLAUDE.md §11).
TIER_COLOURS = {"T1": "#4B7B6E", "T2": "#C08A2E", "T3": "#8B4A6B"}

# Typical study question, used to price tiers on equal footing.
TYPICAL_PROMPT_TOKENS = 60
TYPICAL_COMPLETION_TOKENS = 220


def build_test_matrix(rows: list, feature_set: str, embedder: EmbeddingClient) -> np.ndarray:
    handcrafted = build_handcrafted_matrix([r.to_sample() for r in rows])
    if feature_set == "handcrafted":
        return handcrafted
    embeddings = embedder.embed_many([r.text for r in rows])
    if feature_set == "embedding":
        return embeddings
    return np.hstack([handcrafted, embeddings])


def save_confusion_matrix(matrix: np.ndarray, path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    fig.patch.set_facecolor("#F2F4F1")
    ax.set_facecolor("#F2F4F1")

    ax.imshow(matrix, cmap="BuGn", vmin=0)

    ax.set_xticks(range(len(TIERS)), TIERS)
    ax.set_yticks(range(len(TIERS)), TIERS)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    ax.set_title("Difficulty classifier — test split", pad=12)

    largest = matrix.max() if matrix.size else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="#14171A" if value < largest * 0.6 else "#F2F4F1",
                fontsize=11,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def tier_cost(tier: str) -> float:
    """Counterfactual cost of a typical question at this tier's preferred model."""
    spec = TIER_TABLE[tier][0]  # type: ignore[index]
    return spec.estimated_cost_usd(TYPICAL_PROMPT_TOKENS, TYPICAL_COMPLETION_TOKENS)


def main(force: bool) -> None:
    if TEST_RESULTS.exists() and not force:
        print(f"{TEST_RESULTS} already exists — the test split has been used.")
        print("Re-running would turn it into a second validation set. Pass --force")
        print("only if you intend to discard the previous result deliberately.")
        sys.exit(1)

    model_path = ARTIFACT_DIR / "classifier.joblib"
    if not model_path.exists():
        print(f"{model_path} not found — run ml/train_classifier.py first.")
        sys.exit(1)

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    feature_set = metadata["feature_set"]

    rows = load_labelled()
    splits = make_splits(rows)
    y_test = labels_to_array(splits.test)

    print(f"evaluating on the held-out test split: {len(splits.test)} questions")
    print(f"model: {metadata['model']} / {feature_set}\n")

    embedder = EmbeddingClient(load_keys()["GOOGLE_AI_STUDIO_API_KEY"])
    x_test = build_test_matrix(splits.test, feature_set, embedder)

    model = joblib.load(model_path)
    predicted = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predicted))
    macro_f1 = float(f1_score(y_test, predicted, average="macro"))
    baseline = majority_baseline_accuracy(splits.train, splits.test)

    print(f"accuracy           {accuracy:.4f}")
    print(f"macro-F1           {macro_f1:.4f}")
    print(f"majority baseline  {baseline:.4f}   ({accuracy - baseline:+.4f})\n")
    print(classification_report(y_test, predicted, target_names=list(TIERS), zero_division=0))

    matrix = confusion_matrix(y_test, predicted, labels=list(range(len(TIERS))))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(matrix, CONFUSION_PNG)
    print(f"confusion matrix saved to {CONFUSION_PNG}")

    # Counterfactual cost: what routing by prediction would cost against sending
    # everything to the top tier. No money is spent either way.
    top_tier_cost = tier_cost("T3")
    routed = sum(tier_cost(TIERS[int(p)]) for p in predicted)
    always_top = top_tier_cost * len(predicted)
    reduction = 100 * (1 - routed / always_top) if always_top else 0.0

    # Quality retention: how often the predicted tier is at least as strong as the
    # tier the question actually needed. A lower prediction is where the verifier
    # has to catch the shortfall and escalate.
    sufficient = int(np.sum(predicted >= y_test))
    retention = 100 * sufficient / len(y_test)

    print(f"\nestimated cost reduction vs always-T3: {reduction:.1f}%  (counterfactual)")
    print(f"predicted tier sufficient for the question: {retention:.1f}%")
    print(f"under-predicted (verifier must escalate):   {100 - retention:.1f}%")

    payload = {
        "evaluated_at": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
        "model": metadata["model"],
        "feature_set": feature_set,
        "test_size": len(splits.test),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "majority_baseline_accuracy": round(baseline, 4),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(TIERS),
        "classification_report": classification_report(
            y_test, predicted, target_names=list(TIERS), zero_division=0, output_dict=True
        ),
        "counterfactual_cost": {
            "note": (
                "Estimated from published per-token list prices. Every call ran on "
                "a free tier; no money was spent."
            ),
            "typical_prompt_tokens": TYPICAL_PROMPT_TOKENS,
            "typical_completion_tokens": TYPICAL_COMPLETION_TOKENS,
            "routed_usd": round(routed, 8),
            "always_top_tier_usd": round(always_top, 8),
            "reduction_percent": round(reduction, 2),
        },
        "quality": {
            "predicted_tier_sufficient_percent": round(retention, 2),
            "under_predicted_percent": round(100 - retention, 2),
        },
    }
    TEST_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {TEST_RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even though the test split has already been used",
    )
    main(parser.parse_args().force)
