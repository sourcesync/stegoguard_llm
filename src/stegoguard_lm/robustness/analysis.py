from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .statistics import (
    benjamini_hochberg,
    paired_parent_difference,
    paired_parent_values,
    paired_sign_flip_test,
    summarize_effects,
)


def split_roles(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stego = [r for r in rows if r.get("record_role") == "stego_trajectory"]
    clean = [r for r in rows if r.get("record_role") == "clean_control"]
    return stego, clean


def _filter(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = rows
    for key, value in filters.items():
        if value is None:
            continue
        out = [r for r in out if r.get(key) == value]
    return out


def _r3_contrast_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    round_id = int(cfg["endpoint_round"])
    specs: list[dict[str, Any]] = []
    for defender in cfg["defender_policies"]:
        for test, ref in cfg["condition_contrasts"]:
            specs.append({
                "contrast_type": "attacker_condition",
                "round_id": round_id,
                "defender_policy": defender,
                "attacker_condition": None,
                "test": test,
                "reference": ref,
            })
    test_def, ref_def = cfg["defender_contrast"]
    for attacker in cfg["attacker_conditions"]:
        specs.append({
            "contrast_type": "defender_policy",
            "round_id": round_id,
            "defender_policy": None,
            "attacker_condition": attacker,
            "test": test_def,
            "reference": ref_def,
        })
    return specs


def _rows_for_spec(stego: list[dict[str, Any]], spec: dict[str, Any], *, extra: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = {"round_id": int(spec["round_id"])}
    if extra:
        base.update(extra)
    if spec["contrast_type"] == "attacker_condition":
        test = [r for r in stego if int(r["round_id"]) == int(base["round_id"])
                and r["defender_policy"] == spec["defender_policy"]
                and r["attacker_condition"] == spec["test"]]
        ref = [r for r in stego if int(r["round_id"]) == int(base["round_id"])
               and r["defender_policy"] == spec["defender_policy"]
               and r["attacker_condition"] == spec["reference"]]
    else:
        test = [r for r in stego if int(r["round_id"]) == int(base["round_id"])
                and r["attacker_condition"] == spec["attacker_condition"]
                and r["defender_policy"] == spec["test"]]
        ref = [r for r in stego if int(r["round_id"]) == int(base["round_id"])
               and r["attacker_condition"] == spec["attacker_condition"]
               and r["defender_policy"] == spec["reference"]]
    if extra:
        for key, value in extra.items():
            if key == "round_id":
                continue
            test = [r for r in test if r.get(key) == value]
            ref = [r for r in ref if r.get(key) == value]
    return test, ref


def _contrast_id(spec: dict[str, Any]) -> str:
    if spec["contrast_type"] == "attacker_condition":
        return f"{spec['defender_policy']}::{spec['test']}-vs-{spec['reference']}"
    return f"{spec['attacker_condition']}::{spec['test']}-vs-{spec['reference']}"


def compute_randomization_tests(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-test all 27 Step 11 primary contrasts with paired sign-flip randomization."""
    cfg = config["step12_robustness"]
    seed0 = int(config["seed"])
    reps = int(cfg["randomization_replicates"])
    rows: list[dict[str, Any]] = []
    index = 0
    # The exact 27 contrast universe used in Step 11.
    for defender in cfg["defender_policies"]:
        for round_id in cfg["primary_comparison_rounds"]:
            for test, ref in cfg["condition_contrasts"]:
                spec = {
                    "contrast_type": "attacker_condition", "round_id": int(round_id),
                    "defender_policy": defender, "attacker_condition": None,
                    "test": test, "reference": ref,
                }
                test_rows, ref_rows = _rows_for_spec(stego, spec)
                for outcome in cfg["primary_outcomes"]:
                    index += 1
                    _, diffs = paired_parent_values(test_rows, ref_rows, outcome)
                    test_result = paired_sign_flip_test(diffs, replicates=reps, seed=seed0 + 37 * index)
                    rows.append({**spec, "contrast_id": _contrast_id(spec), "outcome": outcome, **test_result})
    test_def, ref_def = cfg["defender_contrast"]
    for attacker in cfg["attacker_conditions"]:
        for round_id in cfg["primary_comparison_rounds"]:
            spec = {
                "contrast_type": "defender_policy", "round_id": int(round_id),
                "defender_policy": None, "attacker_condition": attacker,
                "test": test_def, "reference": ref_def,
            }
            test_rows, ref_rows = _rows_for_spec(stego, spec)
            for outcome in cfg["primary_outcomes"]:
                index += 1
                _, diffs = paired_parent_values(test_rows, ref_rows, outcome)
                test_result = paired_sign_flip_test(diffs, replicates=reps, seed=seed0 + 37 * index)
                rows.append({**spec, "contrast_id": _contrast_id(spec), "outcome": outcome, **test_result})

    # Multiplicity correction is performed within each outcome family, not pooled across unlike outcomes.
    for outcome in cfg["primary_outcomes"]:
        idx = [i for i, row in enumerate(rows) if row["outcome"] == outcome]
        qvals = benjamini_hochberg([rows[i]["p_value"] for i in idx])
        for i, q in zip(idx, qvals):
            rows[i]["q_value_bh"] = q
            rows[i]["fdr_reject_05"] = bool(q is not None and q < float(cfg["fdr_alpha"]))
    return rows


def compute_influence(stego: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = _r3_contrast_specs(config)
    parent_ids = sorted({str(r["clean_parent_id"]) for r in stego})
    cells = sorted({(str(r["source_model"]), str(r["domain"]), int(r["nominal_target_length"])) for r in stego})
    parent_details: list[dict[str, Any]] = []
    cell_details: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for spec in specs:
        test_rows, ref_rows = _rows_for_spec(stego, spec)
        full_effects: dict[str, float] = {}
        for outcome in ["successful_covert_bits", "exact_payload_recovery"]:
            _, diffs = paired_parent_values(test_rows, ref_rows, outcome)
            full_effects[outcome] = float(np.mean(diffs)) if diffs.size else float("nan")

        collected_parent: dict[str, list[float]] = {k: [] for k in full_effects}
        collected_cell: dict[str, list[float]] = {k: [] for k in full_effects}

        for parent in parent_ids:
            t = [r for r in test_rows if str(r["clean_parent_id"]) != parent]
            rr = [r for r in ref_rows if str(r["clean_parent_id"]) != parent]
            row = {**spec, "contrast_id": _contrast_id(spec), "excluded_parent": parent}
            for outcome in full_effects:
                _, diffs = paired_parent_values(t, rr, outcome)
                effect = float(np.mean(diffs)) if diffs.size else float("nan")
                row[f"{outcome}_difference"] = effect
                collected_parent[outcome].append(effect)
            parent_details.append(row)

        for source_model, domain, length in cells:
            t = [r for r in test_rows if not (
                str(r["source_model"]) == source_model and str(r["domain"]) == domain and int(r["nominal_target_length"]) == length
            )]
            rr = [r for r in ref_rows if not (
                str(r["source_model"]) == source_model and str(r["domain"]) == domain and int(r["nominal_target_length"]) == length
            )]
            row = {
                **spec, "contrast_id": _contrast_id(spec),
                "excluded_source_model": source_model, "excluded_domain": domain,
                "excluded_nominal_target_length": length,
            }
            for outcome in full_effects:
                _, diffs = paired_parent_values(t, rr, outcome)
                effect = float(np.mean(diffs)) if diffs.size else float("nan")
                row[f"{outcome}_difference"] = effect
                collected_cell[outcome].append(effect)
            cell_details.append(row)

        row = {**spec, "contrast_id": _contrast_id(spec)}
        for outcome, full_effect in full_effects.items():
            row[f"full_{outcome}_difference"] = full_effect
            for prefix, values in [("loo_parent", collected_parent[outcome]), ("loo_cell", collected_cell[outcome])]:
                stats = summarize_effects(values, full_effect=full_effect)
                for key, value in stats.items():
                    row[f"{outcome}_{prefix}_{key}"] = value
        summary.append(row)
    return summary, parent_details, cell_details


def compute_alternative_leakage(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    specs = _r3_contrast_specs(config)
    reps = int(cfg["bootstrap_replicates"])
    seed0 = int(config["seed"]) + 5000
    output: list[dict[str, Any]] = []
    index = 0
    for spec in specs:
        test_rows, ref_rows = _rows_for_spec(stego, spec)
        for outcome in cfg["alternative_leakage_outcomes"]:
            index += 1
            result = paired_parent_difference(test_rows, ref_rows, outcome, replicates=reps, seed=seed0 + index * 19)
            output.append({
                **spec, "contrast_id": _contrast_id(spec), "outcome": outcome,
                "difference": result["difference"], "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"], "n_pairs": result["n_pairs"],
            })
    return output


def _trajectory_window_rows(stego: list[dict[str, Any]], rounds: set[int]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stego:
        if int(row["round_id"]) in rounds:
            grouped[str(row["trajectory_id"])].append(row)
    output: list[dict[str, Any]] = []
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda r: int(r["round_id"]))
        output.append({
            "trajectory_id": trajectory_id,
            "clean_parent_id": rows[0]["clean_parent_id"],
            "attacker_condition": rows[0]["attacker_condition"],
            "defender_policy": rows[0]["defender_policy"],
            "initial_attack_name": rows[0]["initial_attack_name"],
            "source_model": rows[0]["source_model"],
            "domain": rows[0]["domain"],
            "nominal_target_length": rows[0]["nominal_target_length"],
            "adaptive_window_successful_bits": int(sum(int(r["successful_covert_bits"]) for r in rows)),
            "adaptive_window_exact_survival_count": int(sum(bool(r["exact_payload_recovery"]) for r in rows)),
            "adaptive_window_mean_residual_capacity_bits": float(np.mean([float(r["residual_capacity_bits"]) for r in rows])),
            "adaptive_window_migration_count": int(sum(bool(r["channel_migrated_from_prior_round"]) for r in rows)),
        })
    return output


def compute_adaptive_window_contrasts(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    window = _trajectory_window_rows(stego, set(int(x) for x in cfg["adaptive_rounds"]))
    reps = int(cfg["bootstrap_replicates"])
    seed0 = int(config["seed"]) + 9000
    specs = _r3_contrast_specs(config)
    output: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if spec["contrast_type"] == "attacker_condition":
            test_rows = [r for r in window if r["defender_policy"] == spec["defender_policy"] and r["attacker_condition"] == spec["test"]]
            ref_rows = [r for r in window if r["defender_policy"] == spec["defender_policy"] and r["attacker_condition"] == spec["reference"]]
        else:
            test_rows = [r for r in window if r["attacker_condition"] == spec["attacker_condition"] and r["defender_policy"] == spec["test"]]
            ref_rows = [r for r in window if r["attacker_condition"] == spec["attacker_condition"] and r["defender_policy"] == spec["reference"]]
        bits = paired_parent_difference(test_rows, ref_rows, "adaptive_window_successful_bits", replicates=reps, seed=seed0 + index * 31)
        exact = paired_parent_difference(test_rows, ref_rows, "adaptive_window_exact_survival_count", replicates=reps, seed=seed0 + index * 31 + 1)
        output.append({
            **spec, "contrast_id": _contrast_id(spec),
            "adaptive_rounds": ",".join(f"R{x}" for x in cfg["adaptive_rounds"]),
            "successful_bits_difference": bits["difference"],
            "successful_bits_ci95_lower": bits["ci95_lower"],
            "successful_bits_ci95_upper": bits["ci95_upper"],
            "exact_survival_count_difference": exact["difference"],
            "exact_survival_count_ci95_lower": exact["ci95_lower"],
            "exact_survival_count_ci95_upper": exact["ci95_upper"],
            "n_pairs": bits["n_pairs"],
        })
    return output


def compute_history_conditioned(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    specs = _r3_contrast_specs(config)
    reps = int(cfg["bootstrap_replicates"])
    seed0 = int(config["seed"]) + 13000
    values = cfg["history_values"]
    output: list[dict[str, Any]] = []
    idx = 0
    for spec in specs:
        for history in values:
            idx += 1
            test_rows, ref_rows = _rows_for_spec(stego, spec, extra={"initial_attack_name": history})
            bits = paired_parent_difference(test_rows, ref_rows, "successful_covert_bits", replicates=reps, seed=seed0 + idx * 23)
            exact = paired_parent_difference(test_rows, ref_rows, "exact_payload_recovery", replicates=reps, seed=seed0 + idx * 23 + 1)
            output.append({
                **spec, "contrast_id": _contrast_id(spec), "initial_attack_name": history,
                "successful_bits_difference": bits["difference"],
                "successful_bits_ci95_lower": bits["ci95_lower"],
                "successful_bits_ci95_upper": bits["ci95_upper"],
                "exact_survival_difference": exact["difference"],
                "exact_survival_ci95_lower": exact["ci95_lower"],
                "exact_survival_ci95_upper": exact["ci95_upper"],
                "n_pairs": bits["n_pairs"],
            })
    return output


def compute_heterogeneity(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    specs = _r3_contrast_specs(config)
    reps = int(cfg["bootstrap_replicates"])
    seed0 = int(config["seed"]) + 17000
    output: list[dict[str, Any]] = []
    idx = 0
    for spec in specs:
        for field in cfg["heterogeneity_fields"]:
            values = sorted({r[field] for r in stego}, key=lambda x: str(x))
            for value in values:
                idx += 1
                test_rows, ref_rows = _rows_for_spec(stego, spec, extra={field: value})
                bits = paired_parent_difference(test_rows, ref_rows, "successful_covert_bits", replicates=reps, seed=seed0 + idx * 13)
                exact = paired_parent_difference(test_rows, ref_rows, "exact_payload_recovery", replicates=reps, seed=seed0 + idx * 13 + 1)
                output.append({
                    **spec, "contrast_id": _contrast_id(spec), "stratum_field": field, "stratum_value": value,
                    "successful_bits_difference": bits["difference"],
                    "successful_bits_ci95_lower": bits["ci95_lower"],
                    "successful_bits_ci95_upper": bits["ci95_upper"],
                    "exact_survival_difference": exact["difference"],
                    "exact_survival_ci95_lower": exact["ci95_lower"],
                    "exact_survival_ci95_upper": exact["ci95_upper"],
                    "n_pairs": bits["n_pairs"],
                })
    return output


def compute_clean_cell_metrics(clean: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step12_robustness"]
    cells = sorted({(str(r["source_model"]), str(r["domain"]), int(r["nominal_target_length"])) for r in clean})
    output: list[dict[str, Any]] = []
    for defender in cfg["defender_policies"]:
        for round_id in cfg["round_ids"]:
            for source_model, domain, length in cells:
                rows = [r for r in clean if r["defender_policy"] == defender and int(r["round_id"]) == int(round_id)
                        and str(r["source_model"]) == source_model and str(r["domain"]) == domain and int(r["nominal_target_length"]) == length]
                output.append({
                    "defender_policy": defender, "round_id": int(round_id),
                    "source_model": source_model, "domain": domain, "nominal_target_length": length,
                    "n": len(rows),
                    "intervention_rate": float(np.mean([bool(r["detected"]) for r in rows])) if rows else None,
                    "semantic_similarity_mean": float(np.mean([float(r["semantic_similarity_original_to_mitigated"]) for r in rows])) if rows else None,
                    "instruction_retention_delta_mean": float(np.mean([float(r["visible_instruction_retention_delta"]) for r in rows])) if rows else None,
                    "length_ratio_mean": float(np.mean([float(r["length_ratio"]) for r in rows])) if rows else None,
                })
    return output


def compute_static_ablation_reference(step8_metrics: dict[str, Any], step8_summary: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = step8_metrics["experiments"]
    output: list[dict[str, Any]] = []
    for key in step8_summary["ablation_keys"]:
        overall = experiments[key]["overall"]
        output.append({
            "experiment_key": key,
            "roc_auc": overall["roc_auc"],
            "pr_auc": overall["pr_auc"],
            "f1": overall["f1"],
            "tpr_at_fpr_05": overall["tpr_at_fpr_05"],
            "empirical_fpr_05": overall["empirical_fpr_05"],
            "tpr_at_fpr_01": overall["tpr_at_fpr_01"],
            "empirical_fpr_01": overall["empirical_fpr_01"],
        })
    return output


def compute_cross_stage_reference(step9_summary: dict[str, Any], step11_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Small immutable reference table for paper-level consistency checks; no new inference."""
    rows: list[dict[str, Any]] = []
    for key, value in step9_summary["oracle_payload_security_full"].items():
        rows.append({
            "stage": "step9_oracle_mitigation",
            "condition": key,
            "metric": "exact_payload_recovery_rate",
            "value": value.get("exact_payload_recovery_rate"),
            "note": "Frozen Step 9 full feasible-inventory oracle result",
        })
    for row in step11_summary["tables"]["round_metrics"]:
        if int(row["round_id"]) == 3:
            rows.append({
                "stage": "step11_r3_adaptive",
                "condition": f"{row['attacker_condition']}::{row['defender_policy']}",
                "metric": "exact_payload_survival_rate",
                "value": row["exact_payload_survival_rate"],
                "note": "Frozen Step 11 R3 system-level result",
            })
    return rows


def assess_claim_stability(
    randomization: list[dict[str, Any]],
    influence: list[dict[str, Any]],
    alternative: list[dict[str, Any]],
    step11_summary: dict[str, Any],
) -> dict[str, Any]:
    by = {(r["contrast_id"], r["outcome"]): r for r in randomization if int(r["round_id"]) == 3}
    infl = {r["contrast_id"]: r for r in influence}

    def rand(contrast: str, outcome: str) -> dict[str, Any] | None:
        return by.get((contrast, outcome))

    migration_det = "deterministic_rewrite::channel_migration_feedback-vs-frozen_no_feedback"
    migration_rand = "randomized_rewrite::channel_migration_feedback-vs-frozen_no_feedback"
    mig_fixed_det = "deterministic_rewrite::channel_migration_feedback-vs-fixed_channel_feedback"
    mig_fixed_rand = "randomized_rewrite::channel_migration_feedback-vs-fixed_channel_feedback"
    fixed_det = "deterministic_rewrite::fixed_channel_feedback-vs-frozen_no_feedback"
    fixed_rand = "randomized_rewrite::fixed_channel_feedback-vs-frozen_no_feedback"
    rand_migration = "channel_migration_feedback::randomized_rewrite-vs-deterministic_rewrite"

    def positive_and_fdr(cid: str) -> bool:
        r = rand(cid, "successful_covert_bits")
        return bool(r and r["observed_mean"] > 0 and r.get("fdr_reject_05"))

    loo_support = all(
        infl[cid]["successful_covert_bits_loo_cell_sign_agreement_rate"] == 1.0
        for cid in [migration_det, migration_rand]
    )
    c1 = "robustly_supported" if positive_and_fdr(migration_det) and positive_and_fdr(migration_rand) and loo_support else "supported_with_sensitivity_caveats"

    c2 = "directionally_supported_mixed_uncertainty"
    if positive_and_fdr(mig_fixed_det) and positive_and_fdr(mig_fixed_rand):
        c2 = "robustly_supported"

    fixed_any = any(positive_and_fdr(cid) for cid in [fixed_det, fixed_rand])
    c3 = "weak_or_mixed" if not fixed_any else "some_evidence_but_small"

    rr = rand(rand_migration, "successful_covert_bits")
    c4 = "not_supported" if rr and (rr["observed_mean"] <= 0 or not rr.get("fdr_reject_05")) else "mixed"

    rank_rows = [r for r in step11_summary["tables"]["attack_family_metrics"] if r["attack_name"] == "rank_encoding"]
    exact_zero = all(float(r["exact_payload_survival_rate"]) == 0.0 for r in rank_rows)
    residual_positive = any(float(r["mean_residual_capacity_bits"]) > 0.0 for r in rank_rows)
    c5 = "supported" if exact_zero and residual_positive else "review_required"

    alt_migration = [r for r in alternative if r["contrast_id"] in {migration_det, migration_rand}]
    return {
        "protocol_version": "step12_robustness_v1",
        "C12.1_migration_vs_frozen": {
            "assessment": c1,
            "claim": "Migration-enabled feedback produces higher R3 covert throughput than frozen/no-feedback under both tested defender policies.",
            "robustness_basis": "paired sign-flip tests with BH correction plus leave-one-cell/parent influence checks",
        },
        "C12.2_migration_vs_fixed": {
            "assessment": c2,
            "claim": "Migration tends to outperform fixed-channel adaptation, but the strength of that superiority is less uniform than migration-versus-frozen.",
        },
        "C12.3_fixed_vs_frozen": {
            "assessment": c3,
            "claim": "Within-family adaptation yields only small or unstable recovery relative to frozen/no-feedback.",
        },
        "C12.4_randomization_advantage": {
            "assessment": c4,
            "claim": "Randomized rewriting does not show a sustained adaptive resilience advantage over deterministic rewriting in this bounded protocol.",
        },
        "C12.5_rank_residual_leakage": {
            "assessment": c5,
            "claim": "Exact rank-message delivery is suppressed, but residual information can remain; exact failure must not be equated with zero leakage.",
        },
        "alternative_leakage_rows_for_migration_vs_frozen": alt_migration,
    }


def build_step12_analysis(
    feedback_rows: list[dict[str, Any]],
    step8_metrics: dict[str, Any],
    step8_summary: dict[str, Any],
    step9_summary: dict[str, Any],
    step11_summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cfg = config["step12_robustness"]
    stego, clean = split_roles(feedback_rows)
    parents = len({str(r["clean_parent_id"]) for r in stego})
    trajectories = len({str(r["trajectory_id"]) for r in stego})
    if parents != int(cfg["expected_parent_count"]):
        raise ValueError(f"Unexpected parent count: {parents}")
    if trajectories != int(cfg["expected_trajectory_count"]):
        raise ValueError(f"Unexpected trajectory count: {trajectories}")
    if len(stego) != int(cfg["expected_stego_rows"]) or len(clean) != int(cfg["expected_clean_rows"]):
        raise ValueError("Frozen Step 10 row counts do not match Step 12 specification.")

    randomization = compute_randomization_tests(stego, config)
    influence_summary, loo_parent, loo_cell = compute_influence(stego, config)
    alternative = compute_alternative_leakage(stego, config)
    adaptive_window = compute_adaptive_window_contrasts(stego, config)
    history = compute_history_conditioned(stego, config)
    heterogeneity = compute_heterogeneity(stego, config)
    clean_cells = compute_clean_cell_metrics(clean, config)
    static_ablation = compute_static_ablation_reference(step8_metrics, step8_summary)
    cross_stage = compute_cross_stage_reference(step9_summary, step11_summary)
    claims = assess_claim_stability(randomization, influence_summary, alternative, step11_summary)

    tables = {
        "randomization_tests": randomization,
        "r3_influence_summary": influence_summary,
        "r3_leave_one_parent": loo_parent,
        "r3_leave_one_cell": loo_cell,
        "r3_alternative_leakage": alternative,
        "adaptive_window_contrasts": adaptive_window,
        "history_conditioned_r3": history,
        "heterogeneity_r3": heterogeneity,
        "clean_cell_metrics": clean_cells,
        "static_ablation_reference": static_ablation,
        "cross_stage_reference": cross_stage,
    }
    summary = {
        "protocol_version": cfg["protocol_version"],
        "input_step11_protocol_version": cfg["input_protocol_versions"]["step11"],
        "status": "complete",
        "parents": parents,
        "trajectories": trajectories,
        "stego_rows": len(stego),
        "clean_rows": len(clean),
        "randomization_test_rows": len(randomization),
        "r3_contrast_count": len(influence_summary),
        "loo_parent_rows": len(loo_parent),
        "loo_cell_rows": len(loo_cell),
        "alternative_leakage_rows": len(alternative),
        "adaptive_window_rows": len(adaptive_window),
        "history_conditioned_rows": len(history),
        "heterogeneity_rows": len(heterogeneity),
        "clean_cell_rows": len(clean_cells),
        "static_ablation_rows": len(static_ablation),
        "cross_stage_reference_rows": len(cross_stage),
        "interpretation_guardrails": cfg["interpretation_rules"],
    }
    return summary, tables, claims
