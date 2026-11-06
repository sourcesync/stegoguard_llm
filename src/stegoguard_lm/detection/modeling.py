from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import clean_tail_threshold, fold_cross_model_features, tail_fraction
from .metrics import threshold_for_best_f1, threshold_for_empirical_fpr

VARIANTS = (
    "distributional",
    "cross_model",
    "semantic",
    "distributional_cross_model",
    "distributional_semantic",
    "cross_model_semantic",
    "full_fusion",
)


@dataclass
class FeatureStore:
    dataset_by_id: dict[str, dict[str, Any]]
    evaluator_a_by_id: dict[str, dict[str, Any]]
    evaluator_b_by_id: dict[str, dict[str, Any]]
    semantic_by_id: dict[str, dict[str, Any]]
    unicode_by_id: dict[str, dict[str, Any]]


def _stats(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot compute clean feature statistics from no values.")
    return float(array.mean()), float(array.std(ddof=0))


def _fold_context(
    store: FeatureStore,
    train_ids: list[str],
) -> dict[str, Any]:
    clean_train_ids = {
        sample_id
        for sample_id in train_ids
        if int(store.dataset_by_id[sample_id]["y"]) == 0
    }
    threshold_a = clean_tail_threshold(
        list(store.evaluator_a_by_id.values()), clean_train_ids
    )
    threshold_b = clean_tail_threshold(
        list(store.evaluator_b_by_id.values()), clean_train_ids
    )

    metrics_a: dict[str, list[float]] = defaultdict(list)
    metrics_b: dict[str, list[float]] = defaultdict(list)
    for sample_id in clean_train_ids:
        row_a = store.evaluator_a_by_id[sample_id]
        row_b = store.evaluator_b_by_id[sample_id]
        tail_a = tail_fraction(row_a["token_logprobs"], threshold_a)
        tail_b = tail_fraction(row_b["token_logprobs"], threshold_b)
        for metric, value in (
            ("mean_logp", row_a["mean_logp"]),
            ("ppl", row_a["ppl"]),
            ("tail_05_frac", tail_a),
        ):
            metrics_a[metric].append(float(value))
        for metric, value in (
            ("mean_logp", row_b["mean_logp"]),
            ("ppl", row_b["ppl"]),
            ("tail_05_frac", tail_b),
        ):
            metrics_b[metric].append(float(value))

    clean_stats = {
        "a": {metric: _stats(values) for metric, values in metrics_a.items()},
        "b": {metric: _stats(values) for metric, values in metrics_b.items()},
    }
    return {
        "tail_threshold_a": threshold_a,
        "tail_threshold_b": threshold_b,
        "clean_cross_stats": clean_stats,
    }


def _row_features(
    store: FeatureStore,
    sample_id: str,
    context: dict[str, Any],
) -> dict[str, float]:
    row_a = store.evaluator_a_by_id[sample_id]
    row_b = store.evaluator_b_by_id[sample_id]
    semantic = store.semantic_by_id[sample_id]
    tail_a = tail_fraction(row_a["token_logprobs"], context["tail_threshold_a"])
    tail_b = tail_fraction(row_b["token_logprobs"], context["tail_threshold_b"])

    features = {
        "dist_a_mean_logp": float(row_a["mean_logp"]),
        "dist_a_ppl": float(row_a["ppl"]),
        "dist_a_logp_var": float(row_a["logp_var"]),
        "dist_a_tail_05_frac": float(tail_a),
        "dist_a_sentence_logp_var": float(row_a["sentence_logp_var"]),
        "dist_b_mean_logp": float(row_b["mean_logp"]),
        "dist_b_ppl": float(row_b["ppl"]),
        "dist_b_logp_var": float(row_b["logp_var"]),
        "dist_b_tail_05_frac": float(tail_b),
        "dist_b_sentence_logp_var": float(row_b["sentence_logp_var"]),
        "sem_paraphrase_embedding_similarity": float(
            semantic["paraphrase_embedding_similarity"]
        ),
        "sem_visible_instruction_retention": float(
            semantic["visible_instruction_retention"]
        ),
    }
    features.update(
        fold_cross_model_features(
            row_a=row_a,
            row_b=row_b,
            tail_a=tail_a,
            tail_b=tail_b,
            clean_stats=context["clean_cross_stats"],
        )
    )
    return features


def variant_feature_names(variant: str) -> list[str]:
    distributional = [
        "dist_a_mean_logp",
        "dist_a_ppl",
        "dist_a_logp_var",
        "dist_a_tail_05_frac",
        "dist_a_sentence_logp_var",
        "dist_b_mean_logp",
        "dist_b_ppl",
        "dist_b_logp_var",
        "dist_b_tail_05_frac",
        "dist_b_sentence_logp_var",
    ]
    cross = [
        "delta_mean_logp",
        "delta_ppl",
        "delta_tail_05_frac",
        "abs_delta_mean_logp",
        "abs_delta_ppl",
    ]
    semantic = [
        "sem_paraphrase_embedding_similarity",
        "sem_visible_instruction_retention",
    ]
    mapping = {
        "distributional": distributional,
        "cross_model": cross,
        "semantic": semantic,
        "distributional_cross_model": distributional + cross,
        "distributional_semantic": distributional + semantic,
        "cross_model_semantic": cross + semantic,
        "full_fusion": distributional + cross + semantic,
    }
    if variant not in mapping:
        raise ValueError(f"Unknown detector variant: {variant}")
    return mapping[variant]


def _matrix(
    store: FeatureStore,
    sample_ids: list[str],
    context: dict[str, Any],
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    names = variant_feature_names(variant)
    matrix: list[list[float]] = []
    labels: list[int] = []
    for sample_id in sample_ids:
        values = _row_features(store, sample_id, context)
        matrix.append([float(values[name]) for name in names])
        labels.append(int(store.dataset_by_id[sample_id]["y"]))
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    total = y.size
    weights = np.ones(total, dtype=float)
    weights[y == 1] = total / (2.0 * positives)
    weights[y == 0] = total / (2.0 * negatives)
    return weights


def _make_model(c: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    max_iter=int(max_iter),
                    solver="liblinear",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _inner_oof_scores(
    *,
    store: FeatureStore,
    outer_train_ids: list[str],
    variant: str,
    c: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    topics = sorted({str(store.dataset_by_id[sample_id]["topic_id"]) for sample_id in outer_train_ids})
    score_by_id: dict[str, float] = {}
    for inner_index, test_topic in enumerate(topics):
        inner_test = [
            sample_id
            for sample_id in outer_train_ids
            if str(store.dataset_by_id[sample_id]["topic_id"]) == test_topic
        ]
        inner_train = [sample_id for sample_id in outer_train_ids if sample_id not in set(inner_test)]
        if not inner_test or not inner_train:
            continue
        context = _fold_context(store, inner_train)
        x_train, y_train = _matrix(store, inner_train, context, variant)
        x_test, _ = _matrix(store, inner_test, context, variant)
        if len(np.unique(y_train)) < 2:
            raise ValueError("Inner training fold contains only one class.")
        model = _make_model(c, max_iter, seed + inner_index)
        model.fit(x_train, y_train, logreg__sample_weight=_sample_weights(y_train))
        scores = model.predict_proba(x_test)[:, 1]
        score_by_id.update(zip(inner_test, map(float, scores)))

    ordered_ids = [sample_id for sample_id in outer_train_ids if sample_id in score_by_id]
    if len(ordered_ids) != len(outer_train_ids):
        missing = sorted(set(outer_train_ids) - set(ordered_ids))
        raise ValueError(f"Missing inner OOF scores for {missing[:5]}")
    y = np.asarray([int(store.dataset_by_id[sample_id]["y"]) for sample_id in ordered_ids], dtype=int)
    scores = np.asarray([score_by_id[sample_id] for sample_id in ordered_ids], dtype=float)
    return y, scores


def run_topic_disjoint_cv(
    *,
    store: FeatureStore,
    split_definition: dict[str, Any],
    population_ids: set[str],
    variant: str,
    c: float,
    max_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    predictions: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(split_definition["folds"]):
        train_ids = [
            sample_id
            for sample_id in fold["train_sample_ids"]
            if sample_id in population_ids
        ]
        test_ids = [
            sample_id
            for sample_id in fold["test_sample_ids"]
            if sample_id in population_ids
        ]
        if not test_ids:
            continue
        context = _fold_context(store, train_ids)
        x_train, y_train = _matrix(store, train_ids, context, variant)
        x_test, y_test = _matrix(store, test_ids, context, variant)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise ValueError(
                f"Fold {fold['fold_id']} lacks both classes for variant {variant}."
            )

        inner_y, inner_scores = _inner_oof_scores(
            store=store,
            outer_train_ids=train_ids,
            variant=variant,
            c=c,
            max_iter=max_iter,
            seed=seed + 1000 * fold_index,
        )
        threshold_f1 = threshold_for_best_f1(inner_y, inner_scores)
        threshold_05 = threshold_for_empirical_fpr(inner_scores[inner_y == 0], 0.05)
        threshold_01 = threshold_for_empirical_fpr(inner_scores[inner_y == 0], 0.01)

        model = _make_model(c, max_iter, seed + fold_index)
        model.fit(x_train, y_train, logreg__sample_weight=_sample_weights(y_train))
        test_scores = model.predict_proba(x_test)[:, 1]
        for sample_id, y, score in zip(test_ids, y_test, test_scores):
            base = store.dataset_by_id[sample_id]
            predictions.append(
                {
                    "sample_id": sample_id,
                    "clean_parent_id": base["clean_parent_id"],
                    "pair_id": base["pair_id"],
                    "topic_id": base["topic_id"],
                    "source_model": base["source_model"],
                    "domain": base["domain"],
                    "nominal_target_length": base["nominal_target_length"],
                    "attack_name": base["attack_name"],
                    "label": base["label"],
                    "y": int(y),
                    "payload_bit_length": base["payload_bit_length"],
                    "payload_rate_bpt": base["payload_rate_bpt"],
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "score": float(score),
                    "threshold_f1": float(threshold_f1),
                    "pred_f1": int(score >= threshold_f1),
                    "threshold_fpr_05": float(threshold_05),
                    "pred_fpr_05": int(score >= threshold_05),
                    "threshold_fpr_01": float(threshold_01),
                    "pred_fpr_01": int(score >= threshold_01),
                }
            )
    return predictions
