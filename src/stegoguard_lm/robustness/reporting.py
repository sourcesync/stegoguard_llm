from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import write_csv, write_json


def _label(value: str) -> str:
    return value.replace("_", " ").replace("::", " / ")


def write_figures(tables: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir = Path(config["paths"]["step12_figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. R3 successful-bits effects for the 9 primary endpoint contrasts.
    rows = [r for r in tables["randomization_tests"] if int(r["round_id"]) == 3 and r["outcome"] == "successful_covert_bits"]
    labels = [r["contrast_id"] for r in rows]
    values = [float(r["observed_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(np.arange(len(rows)), values)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(np.arange(len(rows)), [_label(x) for x in labels], fontsize=8)
    ax.set_xlabel("Paired difference in successful covert bits per parent")
    ax.set_title("Step 12: R3 paired effect estimates")
    fig.tight_layout()
    path = out_dir / "step12_r3_effect_estimates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    # 2. Leave-one-cell-out stability ranges.
    rows = tables["r3_influence_summary"]
    values = [float(r["full_successful_covert_bits_difference"]) for r in rows]
    lower = [float(r["successful_covert_bits_loo_cell_min_effect"]) for r in rows]
    upper = [float(r["successful_covert_bits_loo_cell_max_effect"]) for r in rows]
    xerr = np.asarray([[v - lo for v, lo in zip(values, lower)], [hi - v for v, hi in zip(values, upper)]])
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(rows))
    ax.errorbar(values, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, [_label(r["contrast_id"]) for r in rows], fontsize=8)
    ax.set_xlabel("Successful-bits difference; range over 8 leave-one-cell-out analyses")
    ax.set_title("Step 12: Cell-deletion sensitivity")
    fig.tight_layout()
    path = out_dir / "step12_leave_one_cell_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    # 3. Alternative leakage outcomes at R3.
    rows = tables["r3_alternative_leakage"]
    labels = sorted({r["contrast_id"] for r in rows})
    outcomes = ["effective_bit_accuracy", "residual_capacity_bits"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, outcome in zip(axes, outcomes):
        subset = [r for r in rows if r["outcome"] == outcome]
        vals = [float(r["difference"]) for r in subset]
        ax.barh(np.arange(len(subset)), vals)
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(np.arange(len(subset)), [_label(r["contrast_id"]) for r in subset], fontsize=7)
        ax.set_title(_label(outcome))
    fig.suptitle("Step 12: R3 sensitivity to alternative leakage metrics")
    fig.tight_layout()
    path = out_dir / "step12_alternative_leakage_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    # 4. Clean intervention burden across 8 cells and four rounds.
    rows = tables["clean_cell_metrics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for defender in sorted({r["defender_policy"] for r in rows}):
        subset = [r for r in rows if r["defender_policy"] == defender]
        xs = [int(r["round_id"]) for r in subset]
        ys = [float(r["intervention_rate"]) for r in subset]
        # Plot cell points and an aggregate round mean line.
        ax.scatter(xs, ys, alpha=0.45, label=_label(defender))
        means = []
        for round_id in sorted({int(r["round_id"]) for r in subset}):
            vals = [float(r["intervention_rate"]) for r in subset if int(r["round_id"]) == round_id]
            means.append((round_id, float(np.mean(vals))))
        ax.plot([x for x, _ in means], [y for _, y in means], marker="o")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Round")
    ax.set_ylabel("Clean intervention rate")
    ax.set_title("Step 12: Clean intervention burden across balanced cells")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "step12_clean_burden_cells.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    # 5. Frozen Step 8 detector ablation reference.
    rows = tables["static_ablation_reference"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(np.arange(len(rows)), [float(r["roc_auc"]) for r in rows])
    ax.set_yticks(np.arange(len(rows)), [_label(r["experiment_key"].split("::")[-1]) for r in rows], fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Frozen Step 8 detector ablations (reference, not re-estimated)")
    fig.tight_layout()
    path = out_dir / "step12_static_ablation_reference.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    return written


def write_interpretation(summary: dict[str, Any], claims: dict[str, Any], config: dict[str, Any]) -> None:
    path = Path(config["paths"]["step12_interpretation_md"])
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# StegoGuard-LM Step 12 robustness interpretation

Protocol: `{summary['protocol_version']}`

Step 12 is a **model-free robustness, sensitivity, and boundary-condition audit** over already-frozen evidence. It does not generate attacks, add adaptive rounds, retrain the detector, change thresholds, or rerun mitigation.

## Main claim-stability results

- **Migration vs frozen/no-feedback:** {claims['C12.1_migration_vs_frozen']['assessment']}. The migration-enabled arm remains the clearest adaptive recovery mechanism after paired randomization testing and deletion sensitivity checks.
- **Migration vs fixed-channel adaptation:** {claims['C12.2_migration_vs_fixed']['assessment']}. The direction favors migration, but superiority to fixed-channel adaptation is less uniform than superiority to frozen/no-feedback.
- **Fixed-channel adaptation vs frozen/no-feedback:** {claims['C12.3_fixed_vs_frozen']['assessment']}. Within-family search produces relatively small or unstable gains.
- **Randomized vs deterministic rewriting:** {claims['C12.4_randomization_advantage']['assessment']}. Do not claim a general moving-target advantage under the tested four-round protocol.
- **Rank exact suppression vs residual information:** {claims['C12.5_rank_residual_leakage']['assessment']}. Exact rank-message failure must not be interpreted as zero leakage.

## Why Step 12 matters

Step 11 supplied the primary parent-paired analysis. Step 12 asks whether those findings are artifacts of one inferential procedure, one endpoint, a small subset of influential parents/cells, one leakage metric, or one starting-channel history. The robustness package therefore adds paired sign-flip tests with BH-FDR correction, leave-one-parent-out and leave-one-cell-out influence analysis, alternative leakage outcomes, adaptive-window contrasts, initial-channel conditioning, descriptive model/domain/length heterogeneity, and clean-burden cell checks.

## Guardrails

This audit strengthens or qualifies evidence **within the frozen pilot**. It does not establish long-run equilibrium, universal channel migration, arbitrary-channel coverage, causal generalization beyond the pre-frozen attacker policy, or zero information leakage. Stratified results are sensitivity analyses, not high-powered subgroup claims.
"""
    path.write_text(text, encoding="utf-8")


def write_qa(summary: dict[str, Any], config: dict[str, Any]) -> None:
    path = Path(config["paths"]["step12_qa_review_txt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "StegoGuard-LM Step 12 QA Review",
        "=================================",
        "",
        f"Protocol: {summary['protocol_version']}",
        f"Parents: {summary['parents']}",
        f"Trajectories: {summary['trajectories']}",
        f"Randomization test rows: {summary['randomization_test_rows']}",
        f"R3 endpoint contrasts: {summary['r3_contrast_count']}",
        f"Leave-one-parent rows: {summary['loo_parent_rows']}",
        f"Leave-one-cell rows: {summary['loo_cell_rows']}",
        f"Alternative leakage rows: {summary['alternative_leakage_rows']}",
        f"Adaptive-window rows: {summary['adaptive_window_rows']}",
        f"History-conditioned rows: {summary['history_conditioned_rows']}",
        f"Heterogeneity rows: {summary['heterogeneity_rows']}",
        f"Clean-cell rows: {summary['clean_cell_rows']}",
        f"Frozen static-ablation reference rows: {summary['static_ablation_rows']}",
        "",
        "Interpretation guardrails:",
        "- Step 12 is sensitivity/robustness analysis, not a new adaptive experiment.",
        "- Multiple-testing correction is a robustness aid, not a replacement for effect sizes and CIs.",
        "- Leave-one-out checks diagnose influence; they do not create independent replications.",
        "- Descriptive strata are not high-powered confirmatory subgroup tests.",
        "- Exact message failure is not zero leakage.",
        "- Four adaptive rounds do not establish equilibrium, emergence, or long-run convergence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_package(
    summary: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    claims: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    p = config["paths"]
    mapping = {
        "randomization_tests": "step12_randomization_tests_csv",
        "r3_influence_summary": "step12_r3_influence_summary_csv",
        "r3_leave_one_parent": "step12_loo_parent_csv",
        "r3_leave_one_cell": "step12_loo_cell_csv",
        "r3_alternative_leakage": "step12_alternative_leakage_csv",
        "adaptive_window_contrasts": "step12_adaptive_window_csv",
        "history_conditioned_r3": "step12_history_conditioned_csv",
        "heterogeneity_r3": "step12_heterogeneity_csv",
        "clean_cell_metrics": "step12_clean_cell_metrics_csv",
        "static_ablation_reference": "step12_static_ablation_reference_csv",
        "cross_stage_reference": "step12_cross_stage_reference_csv",
    }
    for table_name, key in mapping.items():
        write_csv(tables[table_name], p[key])
    write_json(claims, p["step12_claim_stability_json"])
    figure_paths = write_figures(tables, config)
    combined = dict(summary)
    combined["tables"] = tables
    combined["claim_stability"] = claims
    combined["figures"] = figure_paths
    write_json(combined, p["step12_summary_json"])
    write_interpretation(summary, claims, config)
    write_qa(summary, config)
    return figure_paths
