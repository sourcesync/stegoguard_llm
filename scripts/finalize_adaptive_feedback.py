from __future__ import annotations

import argparse

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_jsonl


def _index(rows, label):
    result = {str(r["task_id"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate task_id in {label}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize bounded Step 10 attacker feedback for one round.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    rp = round_paths(config, args.round)
    if rp["feedback"].exists() and not args.overwrite:
        raise FileExistsError(f"Round feedback already exists: {rp['feedback']}")
    tasks = read_jsonl(rp["tasks"])
    outputs = _index(read_jsonl(rp["outputs"]), "defense outputs")
    payload = _index(read_jsonl(rp["payload"]), "payload results")
    utility = _index(read_jsonl(rp["utility"]), "utility results")

    previous_by_traj = {}
    if args.round > 0:
        previous = read_jsonl(round_paths(config, args.round - 1)["feedback"])
        previous_by_traj = {str(r["trajectory_id"]): r for r in previous if r["record_role"] == "stego_trajectory"}

    rows = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id not in outputs:
            raise ValueError(f"Missing defense output: {task_id}")
        if task_id not in utility:
            raise ValueError(f"Missing utility result: {task_id}")
        out = outputs[task_id]
        util = utility[task_id]
        base = {
            **task,
            "final_text": str(out["final_text"]),
            "text_changed": bool(out["text_changed"]),
            "applied_rewrite_seed": out.get("applied_rewrite_seed"),
            "semantic_similarity_original_to_mitigated": float(util["semantic_similarity_original_to_mitigated"]),
            "visible_instruction_retention_delta": float(util["visible_instruction_retention_delta"]),
            "mean_logp_delta": float(util["mean_logp_delta"]),
            "ppl_delta": float(util["ppl_delta"]),
            "length_ratio": float(util["length_ratio"]),
        }
        if task["record_role"] == "stego_trajectory":
            if task_id not in payload:
                raise ValueError(f"Missing payload result: {task_id}")
            pay = payload[task_id]
            exact = bool(pay["exact_payload_recovery"])
            bits = int(task["payload_bit_length"])
            prior = previous_by_traj.get(str(task["trajectory_id"]))
            changed_strategy = False
            migrated = False
            if prior is not None:
                changed_strategy = (
                    str(prior["attack_name"]) != str(task["attack_name"])
                    or int(prior["strategy_index"]) != int(task["strategy_index"])
                )
                migrated = str(prior["attack_name"]) != str(task["attack_name"])
            base.update(
                {
                    "exact_payload_recovery": exact,
                    "effective_bit_accuracy": float(pay["effective_bit_accuracy"]),
                    "effective_bit_error_rate": float(pay["effective_bit_error_rate"]),
                    "residual_capacity_bits": int(pay["residual_capacity_bits"]),
                    "residual_capacity_fraction": float(pay["residual_capacity_fraction"]),
                    "successful_covert_bits": bits if exact else 0,
                    "strategy_changed_from_prior_round": changed_strategy,
                    "channel_migrated_from_prior_round": migrated,
                    "attacker_feedback": {
                        "detected": bool(task["detected"]),
                        "intervention_route": str(task["intervention_route"]),
                        "exact_payload_recovery": exact,
                    },
                }
            )
        else:
            base.update(
                {
                    "exact_payload_recovery": None,
                    "effective_bit_accuracy": None,
                    "effective_bit_error_rate": None,
                    "residual_capacity_bits": 0,
                    "residual_capacity_fraction": 0.0,
                    "successful_covert_bits": 0,
                    "strategy_changed_from_prior_round": False,
                    "channel_migrated_from_prior_round": False,
                    "attacker_feedback": None,
                }
            )
        rows.append(base)
    atomic_write_jsonl(rows, rp["feedback"])
    print(f"Step 10 round {args.round} bounded feedback finalized.")
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {rp['feedback']}")


if __name__ == "__main__":
    main()
