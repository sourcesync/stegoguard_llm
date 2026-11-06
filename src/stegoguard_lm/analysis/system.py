from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from .statistics import paired_parent_difference, parent_mean_ci


def _subset(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    result = rows
    for key, value in filters.items():
        result = [row for row in result if row.get(key) == value]
    return result


def split_roles(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stego = [row for row in rows if row.get("record_role") == "stego_trajectory"]
    clean = [row for row in rows if row.get("record_role") == "clean_control"]
    return stego, clean


def compute_round_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    output: list[dict[str, Any]] = []
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            for round_id in cfg["round_ids"]:
                rows = [
                    r for r in stego
                    if r["attacker_condition"] == attacker
                    and r["defender_policy"] == defender
                    and int(r["round_id"]) == int(round_id)
                ]
                families = Counter(str(r["attack_name"]) for r in rows)
                output.append({
                    "attacker_condition": attacker,
                    "defender_policy": defender,
                    "round_id": int(round_id),
                    "n": len(rows),
                    "successful_covert_bits_total": int(sum(int(r["successful_covert_bits"]) for r in rows)),
                    "successful_covert_bits_mean": float(np.mean([float(r["successful_covert_bits"]) for r in rows])) if rows else None,
                    "exact_payload_survival_rate": float(np.mean([bool(r["exact_payload_recovery"]) for r in rows])) if rows else None,
                    "mean_effective_bit_accuracy": float(np.mean([float(r["effective_bit_accuracy"]) for r in rows])) if rows else None,
                    "mean_residual_capacity_bits": float(np.mean([float(r["residual_capacity_bits"]) for r in rows])) if rows else None,
                    "intervention_rate": float(np.mean([bool(r["detected"]) for r in rows])) if rows else None,
                    "strategy_change_rate": float(np.mean([bool(r["strategy_changed_from_prior_round"]) for r in rows])) if rows else None,
                    "channel_migration_rate": float(np.mean([bool(r["channel_migrated_from_prior_round"]) for r in rows])) if rows else None,
                    "rank_n": int(families.get("rank_encoding", 0)),
                    "synonym_n": int(families.get("synonym_slot_codebook", 0)),
                    "synonym_share": float(families.get("synonym_slot_codebook", 0) / len(rows)) if rows else None,
                })
    return output


def compute_trajectory_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stego:
        by_traj[str(row["trajectory_id"])].append(row)
    output: list[dict[str, Any]] = []
    for trajectory_id, rows in sorted(by_traj.items()):
        rows.sort(key=lambda r: int(r["round_id"]))
        if [int(r["round_id"]) for r in rows] != [int(x) for x in cfg["round_ids"]]:
            raise ValueError(f"Incomplete or misordered trajectory: {trajectory_id}")
        bits = [int(r["successful_covert_bits"]) for r in rows]
        exact = [int(bool(r["exact_payload_recovery"])) for r in rows]
        output.append({
            "trajectory_id": trajectory_id,
            "clean_parent_id": str(rows[0]["clean_parent_id"]),
            "attacker_condition": str(rows[0]["attacker_condition"]),
            "defender_policy": str(rows[0]["defender_policy"]),
            "initial_attack_name": str(rows[0]["initial_attack_name"]),
            "source_model": str(rows[0]["source_model"]),
            "domain": str(rows[0]["domain"]),
            "nominal_target_length": int(rows[0]["nominal_target_length"]),
            "r0_bits": bits[0],
            "r1_bits": bits[1],
            "r2_bits": bits[2],
            "r3_bits": bits[3],
            "cumulative_successful_bits": int(sum(bits)),
            "r3_minus_r0_bits": int(bits[3] - bits[0]),
            "r3_minus_r1_bits": int(bits[3] - bits[1]),
            "exact_survival_count": int(sum(exact)),
            "migration_count": int(sum(bool(r["channel_migrated_from_prior_round"]) for r in rows)),
            "strategy_change_count": int(sum(bool(r["strategy_changed_from_prior_round"]) for r in rows)),
            "final_attack_name": str(rows[-1]["attack_name"]),
        })
    return output


def compute_pairwise_comparisons(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    reps = int(cfg["bootstrap_replicates"])
    seed0 = int(config["seed"])
    output: list[dict[str, Any]] = []
    index = 0

    def add_row(*, contrast_type: str, round_id: int, defender_policy: str | None,
                attacker_condition: str | None, test_label: str, ref_label: str,
                test_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> None:
        nonlocal index
        index += 1
        bits = paired_parent_difference(
            test_rows, ref_rows, "successful_covert_bits",
            replicates=reps, seed=seed0 + index * 17,
        )
        exact = paired_parent_difference(
            test_rows, ref_rows, "exact_payload_recovery",
            replicates=reps, seed=seed0 + index * 17 + 1,
        )
        output.append({
            "contrast_type": contrast_type,
            "round_id": int(round_id),
            "defender_policy": defender_policy,
            "attacker_condition": attacker_condition,
            "test": test_label,
            "reference": ref_label,
            "n_pairs": bits["n_pairs"],
            "successful_bits_difference": bits["difference"],
            "successful_bits_ci95_lower": bits["ci95_lower"],
            "successful_bits_ci95_upper": bits["ci95_upper"],
            "exact_survival_difference": exact["difference"],
            "exact_survival_ci95_lower": exact["ci95_lower"],
            "exact_survival_ci95_upper": exact["ci95_upper"],
        })

    for defender in cfg["defender_policies"]:
        for round_id in cfg["primary_comparison_rounds"]:
            for test_condition, ref_condition in cfg["primary_condition_contrasts"]:
                test_rows = [r for r in stego if r["defender_policy"] == defender and r["attacker_condition"] == test_condition and int(r["round_id"]) == int(round_id)]
                ref_rows = [r for r in stego if r["defender_policy"] == defender and r["attacker_condition"] == ref_condition and int(r["round_id"]) == int(round_id)]
                add_row(
                    contrast_type="attacker_condition",
                    round_id=int(round_id),
                    defender_policy=defender,
                    attacker_condition=None,
                    test_label=test_condition,
                    ref_label=ref_condition,
                    test_rows=test_rows,
                    ref_rows=ref_rows,
                )

    test_defender, ref_defender = cfg["defender_contrast"]
    for attacker in cfg["attacker_conditions"]:
        for round_id in cfg["primary_comparison_rounds"]:
            test_rows = [r for r in stego if r["defender_policy"] == test_defender and r["attacker_condition"] == attacker and int(r["round_id"]) == int(round_id)]
            ref_rows = [r for r in stego if r["defender_policy"] == ref_defender and r["attacker_condition"] == attacker and int(r["round_id"]) == int(round_id)]
            add_row(
                contrast_type="defender_policy",
                round_id=int(round_id),
                defender_policy=None,
                attacker_condition=attacker,
                test_label=test_defender,
                ref_label=ref_defender,
                test_rows=test_rows,
                ref_rows=ref_rows,
            )
    return output


def compute_transition_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stego:
        by_traj[str(row["trajectory_id"])].append(row)
    output: list[dict[str, Any]] = []
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            trajectories = [sorted(rows, key=lambda r: int(r["round_id"])) for rows in by_traj.values() if rows[0]["attacker_condition"] == attacker and rows[0]["defender_policy"] == defender]
            for prev_round, cur_round in zip(cfg["round_ids"][:-1], cfg["round_ids"][1:]):
                counter: Counter[str] = Counter()
                gross_switches = 0
                for rows in trajectories:
                    prev = next(r for r in rows if int(r["round_id"]) == int(prev_round))
                    cur = next(r for r in rows if int(r["round_id"]) == int(cur_round))
                    key = f"{prev['attack_name']}->{cur['attack_name']}"
                    counter[key] += 1
                    gross_switches += int(prev["attack_name"] != cur["attack_name"])
                output.append({
                    "attacker_condition": attacker,
                    "defender_policy": defender,
                    "transition": f"R{prev_round}->R{cur_round}",
                    "n_trajectories": len(trajectories),
                    "rank_to_rank": int(counter.get("rank_encoding->rank_encoding", 0)),
                    "rank_to_synonym": int(counter.get("rank_encoding->synonym_slot_codebook", 0)),
                    "synonym_to_rank": int(counter.get("synonym_slot_codebook->rank_encoding", 0)),
                    "synonym_to_synonym": int(counter.get("synonym_slot_codebook->synonym_slot_codebook", 0)),
                    "gross_channel_switches": int(gross_switches),
                    "gross_switch_rate": float(gross_switches / len(trajectories)) if trajectories else None,
                })
    return output


def compute_attack_family_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    output: list[dict[str, Any]] = []
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            for round_id in cfg["round_ids"]:
                for attack in cfg["attack_names"]:
                    rows = [r for r in stego if r["attacker_condition"] == attacker and r["defender_policy"] == defender and int(r["round_id"]) == int(round_id) and r["attack_name"] == attack]
                    output.append({
                        "attacker_condition": attacker,
                        "defender_policy": defender,
                        "round_id": int(round_id),
                        "attack_name": attack,
                        "n": len(rows),
                        "successful_covert_bits_total": int(sum(int(r["successful_covert_bits"]) for r in rows)),
                        "exact_payload_survival_rate": float(np.mean([bool(r["exact_payload_recovery"]) for r in rows])) if rows else None,
                        "mean_residual_capacity_bits": float(np.mean([float(r["residual_capacity_bits"]) for r in rows])) if rows else None,
                        "mean_effective_bit_accuracy": float(np.mean([float(r["effective_bit_accuracy"]) for r in rows])) if rows else None,
                    })
    return output


def compute_rank_residual_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    output: list[dict[str, Any]] = []
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            for round_id in cfg["round_ids"]:
                rows = [r for r in stego if r["attacker_condition"] == attacker and r["defender_policy"] == defender and int(r["round_id"]) == int(round_id) and r["attack_name"] == "rank_encoding"]
                output.append({
                    "attacker_condition": attacker,
                    "defender_policy": defender,
                    "round_id": int(round_id),
                    "n_rank": len(rows),
                    "exact_survival_count": int(sum(bool(r["exact_payload_recovery"]) for r in rows)),
                    "mean_residual_capacity_bits": float(np.mean([float(r["residual_capacity_bits"]) for r in rows])) if rows else None,
                    "median_residual_capacity_bits": float(np.median([float(r["residual_capacity_bits"]) for r in rows])) if rows else None,
                    "nonzero_residual_fraction": float(np.mean([float(r["residual_capacity_bits"]) > 0 for r in rows])) if rows else None,
                    "mean_effective_bit_accuracy": float(np.mean([float(r["effective_bit_accuracy"]) for r in rows])) if rows else None,
                })
    return output


def compute_recovery_and_resilience(round_metrics: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config["step11_system_analysis"]
    recovery: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    by_key = {(r["attacker_condition"], r["defender_policy"], r["round_id"]): r for r in round_metrics}
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            rows = [by_key[(attacker, defender, int(round_id))] for round_id in cfg["round_ids"]]
            bits = [int(r["successful_covert_bits_total"]) for r in rows]
            exact = [float(r["exact_payload_survival_rate"]) for r in rows]
            recovery.append({
                "attacker_condition": attacker,
                "defender_policy": defender,
                "r0_bits": bits[0], "r1_bits": bits[1], "r2_bits": bits[2], "r3_bits": bits[3],
                "r3_minus_r0_bits": bits[3] - bits[0],
                "r3_minus_r1_bits": bits[3] - bits[1],
                "cumulative_bits": sum(bits),
                "r0_exact_survival_rate": exact[0],
                "r3_exact_survival_rate": exact[3],
                "r3_minus_r0_exact_rate": exact[3] - exact[0],
            })
            profiles.append({
                "attacker_condition": attacker,
                "defender_policy": defender,
                "baseline_successful_bits": bits[0],
                "final_successful_bits": bits[3],
                "peak_successful_bits": max(bits),
                "cumulative_successful_bits": sum(bits),
                "baseline_exact_survival_rate": exact[0],
                "final_exact_survival_rate": exact[3],
                "final_mean_residual_capacity_bits": rows[3]["mean_residual_capacity_bits"],
                "final_intervention_rate": rows[3]["intervention_rate"],
                "final_synonym_share": rows[3]["synonym_share"],
                "final_channel_migration_rate": rows[3]["channel_migration_rate"],
            })
    return recovery, profiles


def compute_history_conditioned_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    output: list[dict[str, Any]] = []
    attacker = "channel_migration_feedback"
    for defender in cfg["defender_policies"]:
        for initial in cfg["attack_names"]:
            for round_id in cfg["round_ids"]:
                rows = [r for r in stego if r["attacker_condition"] == attacker and r["defender_policy"] == defender and r["initial_attack_name"] == initial and int(r["round_id"]) == int(round_id)]
                output.append({
                    "defender_policy": defender,
                    "initial_attack_name": initial,
                    "round_id": int(round_id),
                    "n": len(rows),
                    "successful_covert_bits_total": int(sum(int(r["successful_covert_bits"]) for r in rows)),
                    "exact_payload_survival_rate": float(np.mean([bool(r["exact_payload_recovery"]) for r in rows])) if rows else None,
                    "synonym_share": float(np.mean([r["attack_name"] == "synonym_slot_codebook" for r in rows])) if rows else None,
                })
    return output


def compute_stratified_metrics(stego: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    r3 = max(int(x) for x in cfg["round_ids"])
    output: list[dict[str, Any]] = []
    models = sorted({str(r["source_model"]) for r in stego})
    domains = sorted({str(r["domain"]) for r in stego})
    lengths = sorted({int(r["nominal_target_length"]) for r in stego})
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            for model in models:
                for domain in domains:
                    for length in lengths:
                        rows = [r for r in stego if r["attacker_condition"] == attacker and r["defender_policy"] == defender and int(r["round_id"]) == r3 and r["source_model"] == model and r["domain"] == domain and int(r["nominal_target_length"]) == int(length)]
                        output.append({
                            "attacker_condition": attacker,
                            "defender_policy": defender,
                            "round_id": r3,
                            "source_model": model,
                            "domain": domain,
                            "nominal_target_length": int(length),
                            "n": len(rows),
                            "successful_covert_bits_total": int(sum(int(r["successful_covert_bits"]) for r in rows)),
                            "exact_payload_survival_rate": float(np.mean([bool(r["exact_payload_recovery"]) for r in rows])) if rows else None,
                            "synonym_share": float(np.mean([r["attack_name"] == "synonym_slot_codebook" for r in rows])) if rows else None,
                        })
    return output


def compute_clean_metrics(clean: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config["step11_system_analysis"]
    output: list[dict[str, Any]] = []
    for defender in cfg["defender_policies"]:
        for round_id in cfg["round_ids"]:
            rows = [r for r in clean if r["defender_policy"] == defender and int(r["round_id"]) == int(round_id)]
            output.append({
                "defender_policy": defender,
                "round_id": int(round_id),
                "n": len(rows),
                "intervention_rate": float(np.mean([bool(r["detected"]) for r in rows])) if rows else None,
                "mean_semantic_similarity": float(np.mean([float(r["semantic_similarity_original_to_mitigated"]) for r in rows])) if rows else None,
                "mean_instruction_retention_delta": float(np.mean([float(r["visible_instruction_retention_delta"]) for r in rows])) if rows else None,
                "mean_length_ratio": float(np.mean([float(r["length_ratio"]) for r in rows])) if rows else None,
            })
    return output


def _find_pairwise(pairwise: list[dict[str, Any]], *, contrast_type: str, round_id: int, test: str, reference: str, defender_policy: str | None = None, attacker_condition: str | None = None) -> dict[str, Any]:
    matches = [r for r in pairwise if r["contrast_type"] == contrast_type and r["round_id"] == round_id and r["test"] == test and r["reference"] == reference and r.get("defender_policy") == defender_policy and r.get("attacker_condition") == attacker_condition]
    if len(matches) != 1:
        raise ValueError(f"Expected one pairwise row, found {len(matches)} for {test} vs {reference}")
    return matches[0]


def compute_hypothesis_assessment(
    round_metrics: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    attack_family: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    def rm(attacker: str, defender: str, round_id: int) -> dict[str, Any]:
        return next(r for r in round_metrics if r["attacker_condition"] == attacker and r["defender_policy"] == defender and r["round_id"] == round_id)

    rank_exact_total = sum(
        int(round(float(r["exact_payload_survival_rate"] or 0) * int(r["n"])))
        for r in attack_family if r["attack_name"] == "rank_encoding"
    )
    synonym_exact_present = any((r["exact_payload_survival_rate"] or 0) > 0 for r in attack_family if r["attack_name"] == "synonym_slot_codebook")
    migration_share_shift = {
        defender: rm("channel_migration_feedback", defender, 3)["synonym_share"] - rm("channel_migration_feedback", defender, 0)["synonym_share"]
        for defender in config["step11_system_analysis"]["defender_policies"]
    }

    fixed_rows = [
        _find_pairwise(pairwise, contrast_type="attacker_condition", round_id=3, test="fixed_channel_feedback", reference="frozen_no_feedback", defender_policy=defender)
        for defender in config["step11_system_analysis"]["defender_policies"]
    ]
    migration_vs_fixed = [
        _find_pairwise(pairwise, contrast_type="attacker_condition", round_id=3, test="channel_migration_feedback", reference="fixed_channel_feedback", defender_policy=defender)
        for defender in config["step11_system_analysis"]["defender_policies"]
    ]
    randomization = _find_pairwise(
        pairwise,
        contrast_type="defender_policy",
        round_id=3,
        test="randomized_rewrite",
        reference="deterministic_rewrite",
        attacker_condition="channel_migration_feedback",
    )

    h1 = rank_exact_total == 0 and synonym_exact_present
    h2 = all(value > 0 for value in migration_share_shift.values())
    h3 = all((row["successful_bits_ci95_lower"] or 0) > 0 for row in fixed_rows)
    h5_point_direction = all((row["successful_bits_difference"] or 0) > 0 for row in migration_vs_fixed)
    h5_strict = all((row["successful_bits_ci95_lower"] or 0) > 0 for row in migration_vs_fixed)
    h4 = (randomization["successful_bits_ci95_upper"] or 0) < 0

    return {
        "protocol_version": config["step11_system_analysis"]["protocol_version"],
        "H10.1": {
            "assessment": "supported_within_protocol" if h1 else "not_supported",
            "claim": "The defense creates differential selection pressure across attack families.",
            "evidence": {"rank_exact_survival_count_across_family_rows": rank_exact_total, "synonym_has_exact_survivors": synonym_exact_present},
        },
        "H10.2": {
            "assessment": "supported_within_protocol" if h2 else "not_supported",
            "claim": "Migration shifts relative use toward synonym under persistent rank suppression.",
            "evidence": {"r3_minus_r0_synonym_share": migration_share_shift},
        },
        "H10.3": {
            "assessment": "supported" if h3 else "weak_or_not_supported",
            "claim": "Fixed-channel feedback produces reliable recovery over frozen control.",
            "evidence": {"r3_fixed_minus_frozen": fixed_rows},
        },
        "H10.4": {
            "assessment": "supported" if h4 else "not_supported",
            "claim": "Randomized rewriting reduces migration-arm adaptive recovery relative to deterministic rewriting.",
            "evidence": {"r3_randomized_minus_deterministic": randomization},
        },
        "H10.5": {
            "assessment": (
                "supported_within_protocol" if h5_strict
                else "directionally_supported_mixed_ci" if h5_point_direction
                else "not_supported"
            ),
            "claim": "Allowing channel migration preserves more covert throughput than fixed-channel adaptation.",
            "evidence": {
                "r3_migration_minus_fixed": migration_vs_fixed,
                "note": "R3 point estimates favor migration under both defender policies, but parent-paired bootstrap intervals are not uniformly separated from zero.",
            },
        },
    }


def build_system_analysis(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    stego, clean = split_roles(rows)
    round_metrics = compute_round_metrics(stego, config)
    trajectory_metrics = compute_trajectory_metrics(stego, config)
    pairwise = compute_pairwise_comparisons(stego, config)
    transitions = compute_transition_metrics(stego, config)
    family = compute_attack_family_metrics(stego, config)
    rank_residual = compute_rank_residual_metrics(stego, config)
    recovery, resilience = compute_recovery_and_resilience(round_metrics, config)
    history = compute_history_conditioned_metrics(stego, config)
    stratified = compute_stratified_metrics(stego, config)
    clean_metrics = compute_clean_metrics(clean, config)
    hypotheses = compute_hypothesis_assessment(round_metrics, pairwise, family, config)

    summary = {
        "protocol_version": config["step11_system_analysis"]["protocol_version"],
        "input_protocol_version": config["step11_system_analysis"]["input_protocol_version"],
        "status": "complete",
        "parents": len({str(r["clean_parent_id"]) for r in stego}),
        "trajectories": len({str(r["trajectory_id"]) for r in stego}),
        "stego_rows": len(stego),
        "clean_rows": len(clean),
        "round_condition_rows": len(round_metrics),
        "pairwise_comparison_rows": len(pairwise),
        "r3_stratified_rows": len(stratified),
        "figure_count": 5,
        "interpretation_guardrails": config["step11_system_analysis"]["interpretation_rules"],
    }
    tables = {
        "round_metrics": round_metrics,
        "pairwise_comparisons": pairwise,
        "trajectory_metrics": trajectory_metrics,
        "transition_metrics": transitions,
        "attack_family_metrics": family,
        "rank_residual_capacity": rank_residual,
        "recovery_metrics": recovery,
        "resilience_profiles": resilience,
        "history_conditioned_metrics": history,
        "stratified_metrics": stratified,
        "clean_metrics": clean_metrics,
    }
    return summary, tables, hypotheses
