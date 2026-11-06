from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stegoguard_lm.detection.metrics import threshold_for_best_f1
from stegoguard_lm.detection.modeling import (
    FeatureStore,
    _fold_context,
    _inner_oof_scores,
    _make_model,
    _matrix,
    _row_features,
    _sample_weights,
    variant_feature_names,
)


@dataclass
class FoldDetector:
    fold_id: str
    context: dict[str, Any]
    model: Any
    threshold_f1: float


def fit_step8_fold_ensemble(
    *,
    static_store: FeatureStore,
    split_definition: dict[str, Any],
    population_ids: set[str],
    variant: str,
    c: float,
    max_iter: int,
    seed: int,
) -> list[FoldDetector]:
    detectors: list[FoldDetector] = []
    for fold_index, fold in enumerate(split_definition["folds"]):
        train_ids = [sid for sid in fold["train_sample_ids"] if sid in population_ids]
        test_ids = [sid for sid in fold["test_sample_ids"] if sid in population_ids]
        if not test_ids:
            continue
        context = _fold_context(static_store, train_ids)
        x_train, y_train = _matrix(static_store, train_ids, context, variant)
        inner_y, inner_scores = _inner_oof_scores(
            store=static_store,
            outer_train_ids=train_ids,
            variant=variant,
            c=c,
            max_iter=max_iter,
            seed=seed + 1000 * fold_index,
        )
        threshold = threshold_for_best_f1(inner_y, inner_scores)
        model = _make_model(c, max_iter, seed + fold_index)
        model.fit(x_train, y_train, logreg__sample_weight=_sample_weights(y_train))
        detectors.append(
            FoldDetector(
                fold_id=str(fold["fold_id"]),
                context=context,
                model=model,
                threshold_f1=float(threshold),
            )
        )
    if not detectors:
        raise ValueError("No Step 8 fold detectors were reconstructed.")
    return detectors


def score_new_sample(
    *,
    new_store: FeatureStore,
    sample_id: str,
    detectors: list[FoldDetector],
    variant: str,
) -> dict[str, Any]:
    names = variant_feature_names(variant)
    fold_rows: list[dict[str, Any]] = []
    for detector in detectors:
        values = _row_features(new_store, sample_id, detector.context)
        x = np.asarray([[float(values[name]) for name in names]], dtype=float)
        score = float(detector.model.predict_proba(x)[0, 1])
        flagged = int(score >= detector.threshold_f1)
        fold_rows.append(
            {
                "fold_id": detector.fold_id,
                "score": score,
                "threshold_f1": detector.threshold_f1,
                "flagged": flagged,
            }
        )
    votes = sum(int(row["flagged"]) for row in fold_rows)
    n = len(fold_rows)
    return {
        "ensemble_mean_score": float(np.mean([row["score"] for row in fold_rows])),
        "ensemble_median_score": float(np.median([row["score"] for row in fold_rows])),
        "flagged_fold_count": int(votes),
        "fold_count": int(n),
        "detected_f1": bool(votes * 2 >= n),
        "fold_results": fold_rows,
    }
