from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def parent_means(
    rows: list[dict[str, Any]],
    value_field: str,
    *,
    cluster_field: str = "clean_parent_id",
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        grouped[str(row[cluster_field])].append(float(value))
    return {parent: float(np.mean(values)) for parent, values in grouped.items() if values}


def paired_parent_values(
    test_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    value_field: str,
    *,
    cluster_field: str = "clean_parent_id",
) -> tuple[list[str], np.ndarray]:
    test = parent_means(test_rows, value_field, cluster_field=cluster_field)
    reference = parent_means(reference_rows, value_field, cluster_field=cluster_field)
    common = sorted(set(test) & set(reference))
    diffs = np.asarray([test[parent] - reference[parent] for parent in common], dtype=float)
    return common, diffs


def paired_parent_difference(
    test_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    value_field: str,
    *,
    replicates: int,
    seed: int,
    cluster_field: str = "clean_parent_id",
) -> dict[str, Any]:
    common, diffs = paired_parent_values(
        test_rows, reference_rows, value_field, cluster_field=cluster_field
    )
    if diffs.size == 0:
        return {
            "difference": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n_pairs": 0,
        }
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        boot[i] = float(np.mean(rng.choice(diffs, size=diffs.size, replace=True)))
    return {
        "difference": float(np.mean(diffs)),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "n_pairs": int(len(common)),
    }


def paired_sign_flip_test(
    diffs: Iterable[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Two-sided paired randomization test using deterministic Monte Carlo sign flips."""
    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return {"observed_mean": None, "p_value": None, "n_pairs": 0}
    observed = float(np.mean(arr))
    if np.allclose(arr, 0.0):
        return {"observed_mean": observed, "p_value": 1.0, "n_pairs": int(arr.size)}
    rng = np.random.default_rng(int(seed))
    extreme = 0
    # Chunk to keep memory bounded on local machines.
    remaining = int(replicates)
    while remaining > 0:
        batch = min(5000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, arr.size), replace=True)
        means = np.mean(signs * arr[None, :], axis=1)
        extreme += int(np.sum(np.abs(means) >= abs(observed) - 1e-15))
        remaining -= batch
    p = float((extreme + 1) / (int(replicates) + 1))
    return {"observed_mean": observed, "p_value": p, "n_pairs": int(arr.size)}


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [None for _ in p_values]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted_sorted = [0.0] * m
    running = 1.0
    for rank_rev in range(m - 1, -1, -1):
        _, p = indexed[rank_rev]
        rank = rank_rev + 1
        candidate = min(1.0, p * m / rank)
        running = min(running, candidate)
        adjusted_sorted[rank_rev] = running
    result: list[float | None] = [None for _ in p_values]
    for (original_index, _), q in zip(indexed, adjusted_sorted):
        result[original_index] = float(q)
    return result


def summarize_effects(values: list[float], *, full_effect: float) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min_effect": None,
            "max_effect": None,
            "mean_effect": None,
            "sign_agreement_rate": None,
            "max_abs_deviation_from_full": None,
        }
    arr = np.asarray(values, dtype=float)
    if full_effect > 0:
        agreement = np.mean(arr > 0)
    elif full_effect < 0:
        agreement = np.mean(arr < 0)
    else:
        agreement = np.mean(np.isclose(arr, 0.0))
    return {
        "n": int(arr.size),
        "min_effect": float(np.min(arr)),
        "max_effect": float(np.max(arr)),
        "mean_effect": float(np.mean(arr)),
        "sign_agreement_rate": float(agreement),
        "max_abs_deviation_from_full": float(np.max(np.abs(arr - float(full_effect)))),
    }
