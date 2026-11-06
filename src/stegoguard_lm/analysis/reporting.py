from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_csv, write_json


def _label_attacker(value: str) -> str:
    return {
        "frozen_no_feedback": "Frozen",
        "fixed_channel_feedback": "Fixed-channel",
        "channel_migration_feedback": "Migration",
    }.get(value, value)


def _label_defender(value: str) -> str:
    return {"deterministic_rewrite": "Det", "randomized_rewrite": "Rand"}.get(value, value)


def write_figures(tables: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = Path(config["paths"]["step11_figures_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    cfg = config["step11_system_analysis"]
    round_metrics = tables["round_metrics"]
    paths: list[str] = []

    def metric_series(attacker: str, defender: str, field: str) -> tuple[list[int], list[float]]:
        rows = [r for r in round_metrics if r["attacker_condition"] == attacker and r["defender_policy"] == defender]
        rows.sort(key=lambda r: int(r["round_id"]))
        return [int(r["round_id"]) for r in rows], [float(r[field]) for r in rows]

    # Figure 1: successful covert throughput.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            x, y = metric_series(attacker, defender, "successful_covert_bits_total")
            ax.plot(x, y, marker="o", label=f"{_label_attacker(attacker)} / {_label_defender(defender)}")
    ax.set_xticks(cfg["round_ids"], [f"R{x}" for x in cfg["round_ids"]])
    ax.set_xlabel("Adaptive round")
    ax.set_ylabel("Successful covert bits")
    ax.set_title("Step 11: successful covert throughput by condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "step11_throughput_by_condition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Figure 2: exact survival.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for attacker in cfg["attacker_conditions"]:
        for defender in cfg["defender_policies"]:
            x, y = metric_series(attacker, defender, "exact_payload_survival_rate")
            ax.plot(x, y, marker="o", label=f"{_label_attacker(attacker)} / {_label_defender(defender)}")
    ax.set_xticks(cfg["round_ids"], [f"R{x}" for x in cfg["round_ids"]])
    ax.set_xlabel("Adaptive round")
    ax.set_ylabel("Exact payload survival rate")
    ax.set_title("Step 11: exact payload survival by condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "step11_exact_survival_by_condition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Figure 3: synonym share in migration arm.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for defender in cfg["defender_policies"]:
        x, y = metric_series("channel_migration_feedback", defender, "synonym_share")
        ax.plot(x, y, marker="o", label=_label_defender(defender))
    ax.set_xticks(cfg["round_ids"], [f"R{x}" for x in cfg["round_ids"]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Adaptive round")
    ax.set_ylabel("Synonym share")
    ax.set_title("Step 11: channel share under migration feedback")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "step11_migration_synonym_share.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Figure 4: rank residual capacity at R3.
    rank_rows = [r for r in tables["rank_residual_capacity"] if int(r["round_id"]) == 3]
    labels = [f"{_label_attacker(r['attacker_condition'])}\n{_label_defender(r['defender_policy'])}" for r in rank_rows]
    values = [float(r["mean_residual_capacity_bits"]) for r in rank_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)), labels, rotation=20, ha="right")
    ax.set_ylabel("Mean residual capacity (bits)")
    ax.set_title("Step 11: residual rank information at R3")
    fig.tight_layout()
    path = figure_dir / "step11_rank_residual_capacity_r3.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Figure 5: clean semantic utility.
    clean = tables["clean_metrics"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for defender in cfg["defender_policies"]:
        rows = [r for r in clean if r["defender_policy"] == defender]
        rows.sort(key=lambda r: int(r["round_id"]))
        ax.plot([r["round_id"] for r in rows], [r["mean_semantic_similarity"] for r in rows], marker="o", label=_label_defender(defender))
    ax.set_xticks(cfg["round_ids"], [f"R{x}" for x in cfg["round_ids"]])
    ax.set_ylim(0.9, 1.01)
    ax.set_xlabel("Adaptive round")
    ax.set_ylabel("Mean clean semantic similarity")
    ax.set_title("Step 11: clean visible-response semantic utility")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "step11_clean_semantic_utility.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_interpretation(
    summary: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    hypotheses: dict[str, Any],
    config: dict[str, Any],
) -> None:
    round_metrics = tables["round_metrics"]
    clean = tables["clean_metrics"]

    def rm(attacker: str, defender: str, round_id: int) -> dict[str, Any]:
        return next(r for r in round_metrics if r["attacker_condition"] == attacker and r["defender_policy"] == defender and r["round_id"] == round_id)

    det_mig = [rm("channel_migration_feedback", "deterministic_rewrite", r) for r in range(4)]
    rand_mig = [rm("channel_migration_feedback", "randomized_rewrite", r) for r in range(4)]
    frozen_det = [rm("frozen_no_feedback", "deterministic_rewrite", r) for r in range(4)]
    fixed_det = [rm("fixed_channel_feedback", "deterministic_rewrite", r) for r in range(4)]
    clean_interventions = sorted({round(float(r["intervention_rate"]), 12) for r in clean})

    text = f"""# StegoGuard-LM Step 11 System-Level Interpretation

Protocol: `{summary['protocol_version']}`  
Input: final frozen Step 10 (`{summary['input_protocol_version']}`)

## Central result

The frozen/no-feedback deterministic control remains flat at {[r['successful_covert_bits_total'] for r in frozen_det]} successful covert bits across R0-R3. Fixed-channel deterministic feedback changes strategies but yields only {[r['successful_covert_bits_total'] for r in fixed_det]} bits. In contrast, the channel-migration arm rises from {det_mig[0]['successful_covert_bits_total']} to {det_mig[1]['successful_covert_bits_total']} bits after the first feedback opportunity and remains at {det_mig[3]['successful_covert_bits_total']} bits at R3 under deterministic rewriting; randomized migration follows {[r['successful_covert_bits_total'] for r in rand_mig]}.

This supports **mechanism-specific adaptive recovery**: the strongest recovery comes from changing attack family, not merely searching within a fixed family.

## Selection pressure and displacement

Rank exact-message delivery remains suppressed in the tested adaptive trajectories, while synonym encoding supplies the exact surviving messages. In the migration arm, synonym share rises from {det_mig[0]['synonym_share']:.1%} at R0 to {det_mig[3]['synonym_share']:.1%} at R3 under deterministic rewriting and to {rand_mig[3]['synonym_share']:.1%} under randomized rewriting. The defensible interpretation is **displacement-limited resilience**: StegoGuard-LM remains highly effective against one high-capacity family, yet its heterogeneous protection changes the relative attractiveness of another family.

## Micro/macro dynamics

The transition table should be read alongside aggregate channel shares. Approximately stable migration-arm synonym shares after R1 coexist with substantial bidirectional trajectory-level switching. This is evidence of persistent micro-level churn beneath a comparatively stable macro composition; it is **not** evidence of equilibrium or emergence.

## Randomization

The pre-frozen hypothesis that randomized rewriting would reduce adaptive recovery is not supported. Randomized and deterministic migration endpoints are close, and the parent-paired R3 comparison must be reported with its confidence interval rather than framed as a universal advantage for either policy.

## Residual information

Zero exact rank-message recovery is not zero leakage. Step 11 therefore preserves rank effective bit accuracy and residual-capacity summaries. The paper should distinguish exact-message delivery from partial recoverable information.

## Legitimate-user burden

Clean intervention remains {clean_interventions[0]:.1%} across all round/policy cells, while clean semantic similarity remains high. This is a multi-objective security-service result: the system must control covert throughput while also limiting unnecessary intervention and preserving visible service utility.

## Hypothesis assessment

"""
    for key in ["H10.1", "H10.2", "H10.3", "H10.4", "H10.5"]:
        item = hypotheses[key]
        text += f"- **{key}: {item['assessment']}** — {item['claim']}\n"
    text += """

## Guardrails for paper writing

- Do not claim StegoGuard-LM eliminates LLM steganography.
- Do not claim zero rank leakage from zero exact rank recovery.
- Do not call the four-round behavior equilibrium, emergence, convergence, or a tipping point.
- Do not generalize the rank/synonym migration result to arbitrary covert channels.
- Treat model/domain/length strata as descriptive robustness slices; the 6-parent cells are small.
- Preserve the negative randomization result.
- Keep AI security as the novelty anchor; use complex-systems and service-systems theory to interpret repeated interaction, displacement, resilience, legitimate-user burden, and unintended consequences.
"""
    path = Path(config["paths"]["step11_system_interpretation_md"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_qa(summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> None:
    path = Path(config["paths"]["step11_qa_review_txt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "StegoGuard-LM Step 11 QA Review\n"
        "=================================\n\n"
        f"Protocol: {summary['protocol_version']}\n"
        f"Parents: {summary['parents']}\n"
        f"Trajectories: {summary['trajectories']}\n"
        f"Stego rows: {summary['stego_rows']}\n"
        f"Clean rows: {summary['clean_rows']}\n"
        f"Pairwise rows: {summary['pairwise_comparison_rows']}\n"
        f"R3 stratified rows: {summary['r3_stratified_rows']}\n"
        f"Figures: {summary['figure_count']}\n\n"
        "Interpretation guardrails:\n"
        "- Use frozen/no-feedback controls before attributing change to feedback.\n"
        "- Exact payload failure is not zero information leakage.\n"
        "- Four rounds do not establish equilibrium, emergence, tipping points, or long-run convergence.\n"
        "- Initial-channel effects are history-conditioned within this protocol only.\n"
        "- Randomization is a negative/unsupported hypothesis unless paired evidence shows otherwise.\n"
        "- R3 model/domain/length strata are descriptive, not high-powered inferential subgroups.\n"
    )
    path.write_text(text, encoding="utf-8")


def write_result_package(
    summary: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    hypotheses: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    path_map = {
        "round_metrics": "step11_round_metrics_csv",
        "pairwise_comparisons": "step11_pairwise_comparisons_csv",
        "trajectory_metrics": "step11_trajectory_metrics_csv",
        "transition_metrics": "step11_transition_metrics_csv",
        "attack_family_metrics": "step11_attack_family_metrics_csv",
        "rank_residual_capacity": "step11_rank_residual_capacity_csv",
        "recovery_metrics": "step11_recovery_metrics_csv",
        "resilience_profiles": "step11_resilience_profiles_csv",
        "history_conditioned_metrics": "step11_history_conditioned_metrics_csv",
        "stratified_metrics": "step11_stratified_metrics_csv",
        "clean_metrics": "step11_clean_metrics_csv",
    }
    for table_name, path_key in path_map.items():
        write_csv(tables[table_name], config["paths"][path_key])
    write_json(hypotheses, config["paths"]["step11_hypothesis_assessment_json"])
    figure_paths = write_figures(tables, config)
    combined = dict(summary)
    combined["tables"] = tables
    combined["hypothesis_assessment"] = hypotheses
    combined["figures"] = figure_paths
    write_json(combined, config["paths"]["step11_system_analysis_json"])
    write_interpretation(summary, tables, hypotheses, config)
    write_qa(summary, tables, config)
    return figure_paths
