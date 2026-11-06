from __future__ import annotations

from typing import Any


def index_primary_full_fusion_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in predictions
        if row.get("experiment") == "primary_common_all"
        and row.get("variant") == "full_fusion"
    ]
    index = {str(row["sample_id"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate primary full-fusion prediction rows.")
    return index


def intervention_decision(
    *,
    sample_id: str,
    prediction_row: dict[str, Any],
    unicode_feature_row: dict[str, Any],
    operating_point: str,
) -> dict[str, Any]:
    if operating_point not in {"f1", "fpr_01"}:
        raise ValueError(f"Unsupported operating point: {operating_point}")
    if bool(unicode_feature_row.get("has_suspicious_codepoint")):
        return {
            "intervene": True,
            "route": "unicode_sanitization",
            "decision_reason": "unicode_hygiene_override",
        }
    pred_field = "pred_f1" if operating_point == "f1" else "pred_fpr_01"
    flagged = bool(prediction_row[pred_field])
    return {
        "intervene": flagged,
        "route": "linguistic_rewrite" if flagged else "pass_through",
        "decision_reason": pred_field if flagged else f"{pred_field}_not_flagged",
    }


def selected_oracle_task_id(
    *,
    sample_id: str,
    route: str,
    rewrite_policy: str,
    random_seed: int | None,
) -> str | None:
    if route == "pass_through":
        return None
    if route == "unicode_sanitization":
        return f"unicode_sanitization::fixed::{sample_id}"
    if route != "linguistic_rewrite":
        raise ValueError(f"Unknown route: {route}")
    if rewrite_policy == "deterministic_rewrite":
        return f"deterministic_rewrite::fixed::{sample_id}"
    if rewrite_policy == "randomized_rewrite":
        if random_seed is None:
            raise ValueError("randomized rewrite policy requires a seed")
        return f"randomized_rewrite::seed_{int(random_seed)}::{sample_id}"
    raise ValueError(f"Unknown rewrite policy: {rewrite_policy}")
