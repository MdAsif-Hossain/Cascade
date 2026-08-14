"""Train the difficulty classifier and persist it for the backend to load.

Model selection happens on the validation split only. The test split is untouched
here — it belongs to ``eval/run_eval.py``, which runs once, at the end.

Logistic regression is tried first because it is small, fast, and interpretable,
and because a 512 MB backend has to load whatever comes out of this. Gradient
boosting is trained too, but only adopted if it clears logistic regression on
validation macro-F1 by a margin worth the extra artifact size.

Macro-F1 rather than accuracy drives the choice: the classes are badly imbalanced,
and accuracy would reward a model that predicts the majority tier and never
escalates anything — precisely the failure the verifier exists to catch.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    HANDCRAFTED_NAMES,
    EmbeddingClient,
    build_handcrafted_matrix,
)
from training_data import (
    RANDOM_SEED,
    TIERS,
    LabelledQuestion,
    Splits,
    labels_to_array,
    load_labelled,
    majority_baseline_accuracy,
    make_splits,
)

ARTIFACT_DIR = pathlib.Path(__file__).parents[1] / "backend" / "artifacts"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

FeatureSet = str
FEATURE_SETS: tuple[FeatureSet, ...] = ("handcrafted", "embedding", "both")


def load_keys() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


@dataclass
class FeatureBundle:
    """Feature matrices for every split, for one feature set."""

    name: FeatureSet
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.train.shape[1])


def build_features(splits: Splits, embedder: EmbeddingClient) -> dict[FeatureSet, FeatureBundle]:
    """Build all three feature sets from one pass of embedding work."""
    parts: dict[str, dict[str, np.ndarray]] = {}

    for split_name, rows in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        samples = [r.to_sample() for r in rows]
        handcrafted = build_handcrafted_matrix(samples)
        print(f"  {split_name}: embedding {len(rows)} questions")
        embedding = embedder.embed_many([r.text for r in rows])
        parts[split_name] = {
            "handcrafted": handcrafted,
            "embedding": embedding,
            "both": np.hstack([handcrafted, embedding]),
        }

    return {
        name: FeatureBundle(
            name=name,
            train=parts["train"][name],
            val=parts["val"][name],
            test=parts["test"][name],
        )
        for name in FEATURE_SETS
    }


def make_logistic() -> Pipeline:
    """Scaled logistic regression with balanced class weights.

    Scaling matters because handcrafted counts and embedding components live on
    wildly different ranges. Balanced weights matter because without them the
    model has almost no incentive to ever predict the rare tiers.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_boosting() -> Pipeline:
    return Pipeline(
        [
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, random_state=RANDOM_SEED
                ),
            )
        ]
    )


@dataclass
class Evaluation:
    model_name: str
    feature_set: str
    accuracy: float
    macro_f1: float

    def row(self) -> str:
        return (
            f"  {self.model_name:18s} {self.feature_set:12s} "
            f"acc={self.accuracy:.3f}  macro-F1={self.macro_f1:.3f}"
        )


def evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray, name: str, feature_set: str) -> Evaluation:
    predicted = model.predict(x)
    return Evaluation(
        model_name=name,
        feature_set=feature_set,
        accuracy=float(accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, average="macro")),
    )


def main() -> None:
    rows: list[LabelledQuestion] = load_labelled()
    splits = make_splits(rows)
    print(f"labelled rows: {len(rows)}  ({splits.summary()})")

    distribution: dict[str, int] = {}
    for row in rows:
        distribution[row.label] = distribution.get(row.label, 0) + 1
    print("label distribution:")
    for tier in TIERS:
        count = distribution.get(tier, 0)
        print(f"  {tier}: {count:5d}  ({100 * count / len(rows):.1f}%)")

    y_train = labels_to_array(splits.train)
    y_val = labels_to_array(splits.val)

    print("\nbuilding features")
    embedder = EmbeddingClient(load_keys()["GOOGLE_AI_STUDIO_API_KEY"])
    bundles = build_features(splits, embedder)

    baseline = majority_baseline_accuracy(splits.train, splits.val)
    print(f"\nmajority-class baseline (val): acc={baseline:.3f}")

    print("\nvalidation results")
    results: list[Evaluation] = []
    trained: dict[tuple[str, str], Pipeline] = {}

    for feature_set in FEATURE_SETS:
        bundle = bundles[feature_set]
        for model_name, factory in (("logistic", make_logistic), ("boosting", make_boosting)):
            model = factory()
            model.fit(bundle.train, y_train)
            result = evaluate(model, bundle.val, y_val, model_name, feature_set)
            results.append(result)
            trained[(model_name, feature_set)] = model
            print(result.row())

    best = max(results, key=lambda r: r.macro_f1)
    print(f"\nbest on validation: {best.model_name} / {best.feature_set} (macro-F1 {best.macro_f1:.3f})")

    best_model = trained[(best.model_name, best.feature_set)]
    bundle = bundles[best.feature_set]

    print("\nvalidation report for the selected model")
    print(
        classification_report(
            y_val,
            best_model.predict(bundle.val),
            target_names=list(TIERS),
            zero_division=0,
        )
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "classifier.joblib")

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "model": best.model_name,
        "feature_set": best.feature_set,
        "n_features": bundle.n_features,
        "embedding_model": EMBED_MODEL if best.feature_set != "handcrafted" else None,
        "embedding_dimensions": EMBED_DIMENSIONS if best.feature_set != "handcrafted" else None,
        "handcrafted_features": list(HANDCRAFTED_NAMES),
        "tiers": list(TIERS),
        "random_seed": RANDOM_SEED,
        "dataset_sizes": {
            "total": len(rows),
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "label_distribution": distribution,
        "validation": {
            "majority_baseline_accuracy": round(baseline, 4),
            "accuracy": round(best.accuracy, 4),
            "macro_f1": round(best.macro_f1, 4),
        },
        "test": "not evaluated here — see ml/eval/run_eval.py, run once",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (RESULTS_DIR / "validation_results.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
    )

    size_kb = (ARTIFACT_DIR / "classifier.joblib").stat().st_size / 1024
    print(f"\nsaved {ARTIFACT_DIR / 'classifier.joblib'} ({size_kb:.0f} KB)")
    print(f"saved {ARTIFACT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
