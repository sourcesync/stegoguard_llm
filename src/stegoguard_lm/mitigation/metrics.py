from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def cluster_bootstrap_mean_ci(
    *,
    rows: list[dict[str, Any]],
    value_fn: Callable[[dict[str, Any]], float],
    cluster_field: str,
    replicates: int,
    seed: int,
) -> dict[str, float | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_field])].append(row)
    keys = sorted(clusters)
    if not keys or replicates <= 0:
        return {"lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(replicates)):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        for key in sampled_keys:
            sampled_rows.extend(clusters[str(key)])
        estimates.append(float(np.mean([value_fn(row) for row in sampled_rows])))
    return {
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
    }


def summarize_payload_results(
    rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["attack_name"]))].append(row)
    for (condition, attack), subset in sorted(groups.items()):
        exact = [float(bool(row["exact_payload_recovery"])) for row in subset]
        decoder = [float(bool(row["decoder_success"])) for row in subset]
        bitacc = [float(row["effective_bit_accuracy"]) for row in subset]
        residual = [float(row["residual_capacity_fraction"]) for row in subset]
        key = f"{condition}::{attack}"
        result[key] = {
            "n": len(subset),
            "exact_payload_recovery_rate": float(np.mean(exact)),
            "payload_disruption_rate": float(1.0 - np.mean(exact)),
            "decoder_success_rate": float(np.mean(decoder)),
            "effective_bit_accuracy": _summary(bitacc),
            "residual_capacity_fraction": _summary(residual),
            "exact_recovery_ci95_parent_cluster": cluster_bootstrap_mean_ci(
                rows=subset,
                value_fn=lambda row: float(bool(row["exact_payload_recovery"])),
                cluster_field="clean_parent_id",
                replicates=bootstrap_replicates,
                seed=seed + sum(ord(ch) for ch in key),
            ),
        }
    return result


def summarize_randomized_survival(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "randomized_rewrite":
            by_sample[(str(row["original_sample_id"]), str(row["attack_name"]))].append(row)
    sample_rows: list[dict[str, Any]] = []
    for (sample_id, attack), subset in sorted(by_sample.items()):
        exact = [bool(row["exact_payload_recovery"]) for row in subset]
        sample_rows.append(
            {
                "original_sample_id": sample_id,
                "attack_name": attack,
                "seed_count": len(subset),
                "payload_survival_frequency": sum(exact) / len(exact),
                "any_seed_survival": any(exact),
                "mean_effective_bit_accuracy": float(
                    np.mean([float(row["effective_bit_accuracy"]) for row in subset])
                ),
            }
        )
    result: dict[str, Any] = {"per_sample": sample_rows, "by_attack": {}}
    for attack in sorted({row["attack_name"] for row in sample_rows}):
        subset = [row for row in sample_rows if row["attack_name"] == attack]
        result["by_attack"][attack] = {
            "n_samples": len(subset),
            "mean_payload_survival_frequency": float(
                np.mean([row["payload_survival_frequency"] for row in subset])
            ),
            "any_seed_survival_rate": float(
                np.mean([float(row["any_seed_survival"]) for row in subset])
            ),
            "mean_effective_bit_accuracy": float(
                np.mean([row["mean_effective_bit_accuracy"] for row in subset])
            ),
        }
    return result


def summarize_utility_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["label"]), str(row["attack_name"]))].append(row)
    for key_tuple, subset in sorted(groups.items()):
        condition, label, attack = key_tuple
        result[f"{condition}::{label}::{attack}"] = {
            "n": len(subset),
            "semantic_similarity": _summary(
                float(row["semantic_similarity_original_to_mitigated"]) for row in subset
            ),
            "instruction_retention_delta": _summary(
                float(row["visible_instruction_retention_delta"]) for row in subset
            ),
            "mean_logp_delta": _summary(float(row["mean_logp_delta"]) for row in subset),
            "length_ratio": _summary(float(row["length_ratio"]) for row in subset),
        }
    return result
