"""Difficulty classifier: predict which tier a question needs.

Loads the scikit-learn pipeline trained by ``ml/train_classifier.py``. The
artifact is kilobytes, which is the whole reason a linear model was chosen — the
backend has 512 MB and no room for a transformer.

**Degrades rather than fails.** If the artifact is missing, or the embedding API
is unreachable, the classifier falls back to a default tier instead of raising.
A study assistant that answers every question from the mid tier is worse than one
that routes well; a study assistant that returns 500 because a model file is
absent is worse than both.

The predicted tier is a starting point, not a commitment. The verifier is what
catches an under-prediction, so the cost of being wrong here is one escalation,
not a wrong answer.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import structlog

from app.providers.embeddings import Embedder
from app.routing.features import Sample, handcrafted_features
from app.routing.tiers import Tier

logger = structlog.get_logger(__name__)

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "classifier.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

TIERS: tuple[Tier, ...] = ("T1", "T2", "T3")

# Used when no model is available. T2 rather than T1: without a prediction, the
# safer error is a slightly more expensive answer than a wrong one, and the
# verifier will still escalate if T2 proves insufficient.
FALLBACK_TIER: Tier = "T2"


@dataclass(frozen=True, slots=True)
class Prediction:
    """A tier prediction with the confidence behind it.

    ``confidence`` feeds the calibration analysis — predicted confidence is
    compared against observed escalation rate, which is how we learn whether the
    classifier knows when it is guessing.
    """

    tier: Tier
    confidence: float
    source: str

    @property
    def is_fallback(self) -> bool:
        return self.source != "model"


class DifficultyClassifier:
    """Predicts the tier a question needs, degrading to a default when it cannot."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        model_path: pathlib.Path = MODEL_PATH,
        metadata_path: pathlib.Path = METADATA_PATH,
    ) -> None:
        self._embedder = embedder
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self._load(model_path, metadata_path)

    def _load(self, model_path: pathlib.Path, metadata_path: pathlib.Path) -> None:
        if not model_path.exists():
            logger.warning(
                "classifier_artifact_missing", path=str(model_path), fallback=FALLBACK_TIER
            )
            return
        try:
            import joblib

            self._model = joblib.load(model_path)
        except (OSError, ValueError, ImportError) as exc:
            logger.error("classifier_load_failed", error=str(exc), fallback=FALLBACK_TIER)
            self._model = None
            return

        if metadata_path.exists():
            try:
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("classifier_metadata_unreadable", error=str(exc))

        logger.info(
            "classifier_loaded",
            model=self._metadata.get("model"),
            feature_set=self._metadata.get("feature_set"),
            n_features=self._metadata.get("n_features"),
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def _feature_set(self) -> str:
        return str(self._metadata.get("feature_set", "both"))

    @property
    def needs_embedding(self) -> bool:
        return self._feature_set in ("embedding", "both")

    async def _build_vector(self, sample: Sample) -> list[list[float]] | None:
        """Build the one-row feature matrix the pipeline expects.

        Plain lists rather than numpy arrays: scikit-learn converts them itself,
        and keeping numpy out of the backend's direct imports means one less
        dependency in a 512 MB container.
        """
        handcrafted = handcrafted_features(sample)

        if not self.needs_embedding:
            return [handcrafted]

        if self._embedder is None:
            return None
        embedding = await self._embedder.embed(sample.text)
        if embedding is None:
            return None

        if self._feature_set == "embedding":
            return [list(embedding)]
        return [handcrafted + list(embedding)]

    async def predict(self, question: str, subject: str = "general") -> Prediction:
        """Predict the tier for a question.

        Never raises: every failure path returns a fallback prediction, because a
        routing hint that is unavailable should cost accuracy, not availability.
        """
        if self._model is None:
            return Prediction(FALLBACK_TIER, 0.0, "no_model")

        sample = Sample(text=question, subject=subject, num_choices=0)
        try:
            vector = await self._build_vector(sample)
        except Exception as exc:  # noqa: BLE001 - availability outranks precision here
            logger.warning("classifier_features_failed", error=str(exc))
            return Prediction(FALLBACK_TIER, 0.0, "feature_error")

        if vector is None:
            return Prediction(FALLBACK_TIER, 0.0, "no_embedding")

        try:
            probabilities = [float(p) for p in self._model.predict_proba(vector)[0]]
        except Exception as exc:  # noqa: BLE001 - see above
            logger.warning("classifier_predict_failed", error=str(exc))
            return Prediction(FALLBACK_TIER, 0.0, "predict_error")

        if len(probabilities) != len(TIERS):
            logger.error("classifier_class_count_mismatch", got=len(probabilities))
            return Prediction(FALLBACK_TIER, 0.0, "shape_mismatch")

        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(tier=TIERS[index], confidence=probabilities[index], source="model")
