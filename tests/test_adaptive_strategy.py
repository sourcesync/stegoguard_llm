import unittest

from stegoguard_lm.adaptive.io import derive_int
from stegoguard_lm.adaptive.strategy import (
    RANK,
    SYNONYM,
    assign_initial_channels,
    next_action,
    rank_strategy_seed,
    synonym_strategy_masks,
    transformed_codebook,
)
from stegoguard_lm.attacks.synonym_channel import SynonymCodebook, SynonymSlot


class AdaptiveStrategyTests(unittest.TestCase):
    def test_derive_int_is_deterministic_and_context_sensitive(self):
        a = derive_int("step10", "traj1", 1, 42)
        self.assertEqual(a, derive_int("step10", "traj1", 1, 42))
        self.assertNotEqual(a, derive_int("step10", "traj1", 2, 42))

    def test_balanced_initial_assignment_is_three_and_three_per_cell(self):
        parents = []
        for model in ("qwen", "mistral"):
            for domain in ("conversational", "technical"):
                for length in (64, 256):
                    for i in range(6):
                        parents.append({
                            "sample_id": f"{model}_{domain}_{length}_{i}",
                            "source_model": model,
                            "domain": domain,
                            "nominal_target_length": length,
                        })
        result = assign_initial_channels(parents)
        for model in ("qwen", "mistral"):
            for domain in ("conversational", "technical"):
                for length in (64, 256):
                    ids = sorted(f"{model}_{domain}_{length}_{i}" for i in range(6))
                    self.assertEqual([result[x] for x in ids[:3]], [RANK] * 3)
                    self.assertEqual([result[x] for x in ids[3:]], [SYNONYM] * 3)

    def test_synonym_strategy_budget_respects_capacity(self):
        self.assertEqual(synonym_strategy_masks(1, 10), [0, 1])
        self.assertEqual(synonym_strategy_masks(2, 10), [0, 1, 2, 3])
        self.assertEqual(len(synonym_strategy_masks(4, 10)), 10)

    def test_transformed_codebook_flips_only_selected_active_slots(self):
        codebook = SynonymCodebook(
            version="v1",
            slots=(
                SynonymSlot("slot_a", "small", "tiny"),
                SynonymSlot("slot_b", "quick", "rapid"),
                SynonymSlot("slot_c", "start", "begin"),
            ),
        )
        mapped = transformed_codebook(codebook, active_slot_ids=["slot_a", "slot_c"], mapping_mask=1)
        by_id = {slot.slot_id: slot for slot in mapped.slots}
        self.assertEqual((by_id["slot_a"].zero, by_id["slot_a"].one), ("tiny", "small"))
        self.assertEqual((by_id["slot_b"].zero, by_id["slot_b"].one), ("quick", "rapid"))
        self.assertEqual((by_id["slot_c"].zero, by_id["slot_c"].one), ("start", "begin"))

    def test_frozen_attacker_never_changes(self):
        action = next_action(
            attacker_condition="frozen_no_feedback",
            initial_family=RANK,
            prior_actions=[{"attack_name": RANK, "strategy_index": 0}],
            prior_feedback=[{"exact_payload_recovery": False}],
            available_indices={RANK: [0, 1], SYNONYM: [0, 1]},
        )
        self.assertEqual(action[:2], (RANK, 0))

    def test_fixed_channel_advances_after_failure_and_stays_after_success(self):
        available = {RANK: [0, 2, 5], SYNONYM: [0, 1]}
        failed = next_action(
            attacker_condition="fixed_channel_feedback",
            initial_family=RANK,
            prior_actions=[{"attack_name": RANK, "strategy_index": 2}],
            prior_feedback=[{"exact_payload_recovery": False}],
            available_indices=available,
        )
        self.assertEqual(failed[:2], (RANK, 5))
        survived = next_action(
            attacker_condition="fixed_channel_feedback",
            initial_family=RANK,
            prior_actions=[{"attack_name": RANK, "strategy_index": 2}],
            prior_feedback=[{"exact_payload_recovery": True}],
            available_indices=available,
        )
        self.assertEqual(survived[:2], (RANK, 2))

    def test_migration_switches_family_after_failure(self):
        action = next_action(
            attacker_condition="channel_migration_feedback",
            initial_family=RANK,
            prior_actions=[{"attack_name": RANK, "strategy_index": 0}],
            prior_feedback=[{"exact_payload_recovery": False}],
            available_indices={RANK: [0, 1], SYNONYM: [0, 1, 2]},
        )
        self.assertEqual(action[:2], (SYNONYM, 0))

    def test_rank_strategy_zero_preserves_static_seed(self):
        self.assertEqual(
            rank_strategy_seed(protocol="step10", parent_id="p", static_seed=123, strategy_index=0),
            123,
        )
        self.assertNotEqual(
            rank_strategy_seed(protocol="step10", parent_id="p", static_seed=123, strategy_index=1),
            123,
        )


if __name__ == "__main__":
    unittest.main()
