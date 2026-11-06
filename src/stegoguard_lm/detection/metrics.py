from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_for_empirical_fpr(clean_scores: np.ndarray, target_fpr: float) -> float:
    scores = np.asarray(clean_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("No clean scores available for FPR threshold calibration.")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between 0 and 1.")
    # Select the largest score that can be crossed while keeping empirical FPR
    # at or below target. If even one false positive would exceed the target,
    # choose a threshold just above the clean maximum.
    max_allowed_fp = int(np.floor(target_fpr * scores.size))
    ordered = np.sort(scores)[::-1]
    if max_allowed_fp <= 0:
        return float(np.nextafter(ordered[0], np.inf))
    index = min(max_allowed_fp - 1, ordered.size - 1)
    return float(ordered[index])


def threshold_for_best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(s)
    if candidates.size == 0:
        return 0.5
    best_threshold = float(candidates[0])
    best_value = -1.0
    for threshold in candidates:
        pred = (s >= threshold).astype(int)
        value = f1_score(y, pred, zero_division=0)
        if value > best_value or (
            value == best_value and threshold > best_threshold
        ):
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    pred_f1: np.ndarray | None = None,
    pred_fpr_05: np.ndarray | None = None,
    pred_fpr_01: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    result: dict[str, Any] = {
        "n": int(y.size),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if y.sum() > 0 else float("nan"),
    }
    if pred_f1 is not None:
        p = np.asarray(pred_f1, dtype=int)
        result.update(
            {
                "precision": float(precision_score(y, p, zero_division=0)),
                "recall": float(recall_score(y, p, zero_division=0)),
                "f1": float(f1_score(y, p, zero_division=0)),
                "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
            }
        )
    for name, pred in (("fpr_05", pred_fpr_05), ("fpr_01", pred_fpr_01)):
        if pred is None:
            continue
        p = np.asarray(pred, dtype=int)
        positives = y == 1
        negatives = y == 0
        result[f"tpr_at_{name}"] = float(p[positives].mean()) if positives.any() else float("nan")
        result[f"empirical_{name}"] = float(p[negatives].mean()) if negatives.any() else float("nan")
    return result


def cluster_bootstrap_ci(
    *,
    rows: list[dict[str, Any]],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    replicates: int,
    seed: int,
    cluster_field: str = "clean_parent_id",
) -> dict[str, float]:
    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        clusters.setdefault(str(row[cluster_field]), []).append(index)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        return {"lower": float("nan"), "upper": float("nan")}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    y_all = np.asarray([int(row["y"]) for row in rows], dtype=int)
    s_all = np.asarray([float(row["score"]) for row in rows], dtype=float)
    for _ in range(int(replicates)):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        indices: list[int] = []
        for cluster_id in sampled:
            indices.extend(clusters[str(cluster_id)])
        y = y_all[indices]
        s = s_all[indices]
        if len(np.unique(y)) < 2:
            continue
        try:
            values.append(float(metric_fn(y, s)))
        except ValueError:
            continue
    if not values:
        return {"lower": float("nan"), "upper": float("nan")}
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }
