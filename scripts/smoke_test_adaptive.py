from __future__ import annotations

import argparse

from stegoguard_lm.adaptive.io import derive_int, load_yaml, verify_step10_protocol
from stegoguard_lm.adaptive.strategy import next_action, synonym_strategy_masks
from stegoguard_lm.detection.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-free Step 10 adaptive policy smoke test.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    trajectories = read_jsonl(config["paths"]["adaptive_trajectory_manifest_jsonl"])
    if len(trajectories) != 288:
        raise SystemExit(f"Expected 288 trajectories; found {len(trajectories)}")
    available = {"rank_encoding": list(range(10)), "synonym_slot_codebook": [0, 1, 2, 3]}
    family, strategy, reason = next_action(
        attacker_condition="fixed_channel_feedback",
        initial_family="rank_encoding",
        prior_actions=[{"attack_name": "rank_encoding", "strategy_index": 0}],
        prior_feedback=[{"detected": True, "intervention_route": "linguistic_rewrite", "exact_payload_recovery": False}],
        available_indices=available,
    )
    assert (family, strategy, reason) == ("rank_encoding", 1, "lose_advance_same_family")
    family, strategy, reason = next_action(
        attacker_condition="channel_migration_feedback",
        initial_family="rank_encoding",
        prior_actions=[{"attack_name": "rank_encoding", "strategy_index": 0}],
        prior_feedback=[{"detected": True, "intervention_route": "linguistic_rewrite", "exact_payload_recovery": False}],
        available_indices=available,
    )
    assert (family, strategy, reason) == ("synonym_slot_codebook", 0, "lose_switch_family")
    assert synonym_strategy_masks(1, 10) == [0, 1]
    assert synonym_strategy_masks(4, 10) == list(range(10))
    seed_a = derive_int("step10_adaptive_v1", "trajectory", 1, 101042)
    seed_b = derive_int("step10_adaptive_v1", "trajectory", 1, 101042)
    seed_c = derive_int("step10_adaptive_v1", "trajectory", 2, 101042)
    assert seed_a == seed_b and seed_a != seed_c
    print("STEP 10 MODEL-FREE SMOKE TEST PASSED")
    print(f"Trajectories: {len(trajectories)}")
    print("Policy transition tests: passed")
    print("Seed determinism test: passed")


if __name__ == "__main__":
    main()
