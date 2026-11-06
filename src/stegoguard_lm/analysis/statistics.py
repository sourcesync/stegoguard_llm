from __future__ import annotations

from collections import defaultdict
from typing import Any

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


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "ci95_lower": None, "ci95_upper": None, "n": 0}
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        boot[i] = float(np.mean(rng.choice(array, size=array.size, replace=True)))
    return {
        "mean": float(np.mean(array)),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "n": int(array.size),
    }


def parent_mean_ci(
    rows: list[dict[str, Any]],
    value_field: str,
    *,
    replicates: int,
    seed: int,
    cluster_field: str = "clean_parent_id",
) -> dict[str, Any]:
    values = list(parent_means(rows, value_field, cluster_field=cluster_field).values())
    return bootstrap_mean_ci(values, replicates=replicates, seed=seed)


def paired_parent_difference(
    test_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    value_field: str,
    *,
    replicates: int,
    seed: int,
    cluster_field: str = "clean_parent_id",
) -> dict[str, Any]:
    test = parent_means(test_rows, value_field, cluster_field=cluster_field)
    reference = parent_means(reference_rows, value_field, cluster_field=cluster_field)
    common = sorted(set(test) & set(reference))
    if not common:
        return {
            "difference": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n_pairs": 0,
        }
    diffs = np.asarray([test[parent] - reference[parent] for parent in common], dtype=float)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        boot[i] = float(np.mean(rng.choice(diffs, size=diffs.size, replace=True)))
    return {
        "difference": float(np.mean(diffs)),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "n_pairs": int(diffs.size),
    }
