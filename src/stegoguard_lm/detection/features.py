from __future__ import annotations

import math
from typing import Any

import numpy as np

BASE_DISTRIBUTIONAL = (
    "mean_logp",
    "ppl",
    "logp_var",
    "sentence_logp_var",
)


def tail_fraction(token_logprobs: list[float], threshold: float) -> float:
    if not token_logprobs:
        raise ValueError("tail_fraction requires at least one token log probability")
    values = np.asarray(token_logprobs, dtype=float)
    return float(np.mean(values < float(threshold)))


def clean_tail_threshold(
    evaluator_records: list[dict[str, Any]],
    clean_train_ids: set[str],
    quantile: float = 0.05,
) -> float:
    values: list[float] = []
    for row in evaluator_records:
        if str(row["sample_id"]) in clean_train_ids:
            values.extend(map(float, row["token_logprobs"]))
    if not values:
        raise ValueError("No clean training token log probabilities available.")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def safe_zscore(value: float, mean: float, std: float) -> float:
    if not math.isfinite(std) or std <= 1e-12:
        return 0.0
    return float((value - mean) / std)


def fold_cross_model_features(
    *,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    tail_a: float,
    tail_b: float,
    clean_stats: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, float]:
    values_a = {
        "mean_logp": float(row_a["mean_logp"]),
        "ppl": float(row_a["ppl"]),
        "tail_05_frac": float(tail_a),
    }
    values_b = {
        "mean_logp": float(row_b["mean_logp"]),
        "ppl": float(row_b["ppl"]),
        "tail_05_frac": float(tail_b),
    }
    deltas: dict[str, float] = {}
    for metric in ("mean_logp", "ppl", "tail_05_frac"):
        mean_a, std_a = clean_stats["a"][metric]
        mean_b, std_b = clean_stats["b"][metric]
        z_a = safe_zscore(values_a[metric], mean_a, std_a)
        z_b = safe_zscore(values_b[metric], mean_b, std_b)
        deltas[f"delta_{metric}"] = z_a - z_b
    deltas["abs_delta_mean_logp"] = abs(deltas["delta_mean_logp"])
    deltas["abs_delta_ppl"] = abs(deltas["delta_ppl"])
    return deltas
