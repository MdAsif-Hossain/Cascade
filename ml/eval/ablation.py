"""Ablation study: what does each family of features actually contribute?

This is the single most important artifact in the project (CLAUDE.md §7). It
answers a question the rest of the work cannot: is the classifier learning from
the embedding, from the handcrafted signals, or genuinely from both?

The answer has a direct engineering consequence. If handcrafted features alone
come close, the embedding API call on every request is hard to justify — it costs
a network round trip on the hot path and makes the backend depend on Google for
routing as well as generation. If embeddings dominate, the handcrafted features
are decoration and should be described as such rather than listed as a
contribution.

Reported against the majority-class baseline throughout, because the label
distribution is heavily skewed and raw accuracy would flatter a model that has
learned nothing.

Validation split only. The test split is touched once, by ``run_eval.py``.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features import EmbeddingClient, build_handcrafted_matrix  # noqa: E402
from train_classifier import load_keys, make_boosting, make_logistic  # noqa: E402
from training_data import (  # noqa: E402
    TIERS,
    labels_to_array,
    load_labelled,
    majority_baseline_accuracy,
    make_splits,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

FEATURE_SETS = ("handcrafted", "embedding", "both")
MODELS = (("logistic", make_logistic), ("boosting", make_boosting))


def build_matrices(
    rows_train: list, rows_val: list, embedder: EmbeddingClient
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    hand_train = build_handcrafted_matrix([r.to_sample() for r in rows_train])
    hand_val = build_handcrafted_matrix([r.to_sample() for r in rows_val])

    print("  embedding training questions")
    emb_train = embedder.embed_many([r.text for r in rows_train])
    print("  embedding validation questions")
    emb_val = embedder.embed_many([r.text for r in rows_val])

    return {
        "handcrafted": (hand_train, hand_val),
        "embedding": (emb_train, emb_val),
        "both": (
            np.hstack([hand_train, emb_train]),
            np.hstack([hand_val, emb_val]),
        ),
    }


def main() -> None:
    rows = load_labelled()
    splits = make_splits(rows)
    y_train = labels_to_array(splits.train)
    y_val = labels_to_array(splits.val)

    print(f"rows={len(rows)}  {splits.summary()}")
    baseline = majority_baseline_accuracy(splits.train, splits.val)
    print(f"majority-class baseline accuracy (val): {baseline:.3f}\n")

    embedder = EmbeddingClient(load_keys()["GOOGLE_AI_STUDIO_API_KEY"])
    matrices = build_matrices(splits.train, splits.val, embedder)

    results = []
    print(f"\n{'model':<12}{'features':<14}{'n_feat':>8}{'acc':>8}{'macro-F1':>10}{'vs base':>9}")
    print("-" * 61)

    for model_name, factory in MODELS:
        for feature_set in FEATURE_SETS:
            x_train, x_val = matrices[feature_set]
            model = factory()
            model.fit(x_train, y_train)
            predicted = model.predict(x_val)

            accuracy = float(accuracy_score(y_val, predicted))
            macro_f1 = float(f1_score(y_val, predicted, average="macro"))
            precision, recall, f1, support = precision_recall_fscore_support(
                y_val, predicted, labels=list(range(len(TIERS))), zero_division=0
            )

            results.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": int(x_train.shape[1]),
                    "accuracy": round(accuracy, 4),
                    "macro_f1": round(macro_f1, 4),
                    "accuracy_over_baseline": round(accuracy - baseline, 4),
                    "per_class": {
                        tier: {
                            "precision": round(float(precision[i]), 4),
                            "recall": round(float(recall[i]), 4),
                            "f1": round(float(f1[i]), 4),
                            "support": int(support[i]),
                        }
                        for i, tier in enumerate(TIERS)
                    },
                }
            )

            print(
                f"{model_name:<12}{feature_set:<14}{x_train.shape[1]:>8}"
                f"{accuracy:>8.3f}{macro_f1:>10.3f}{accuracy - baseline:>+9.3f}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "majority_baseline_accuracy": round(baseline, 4),
        "split_sizes": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "results": results,
    }
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown so the table can be pasted straight into the report without
    # being retyped — retyping is how numbers drift from their source.
    lines = [
        "| Model | Features | Dimensions | Accuracy | Macro-F1 | vs. majority baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['feature_set']} | {r['n_features']} | "
            f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['accuracy_over_baseline']:+.3f} |"
        )
    lines.append("")
    lines.append(f"Majority-class baseline accuracy: {baseline:.3f}")
    (RESULTS_DIR / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nbest: {best['model']} / {best['feature_set']} (macro-F1 {best['macro_f1']:.3f})")
    print(f"wrote {RESULTS_DIR / 'ablation.json'} and ablation.md")


if __name__ == "__main__":
    main()
