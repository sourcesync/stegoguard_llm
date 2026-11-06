from __future__ import annotations

from dataclasses import replace
from typing import Any

from stegoguard_lm.attacks.synonym_channel import SynonymCodebook, SynonymSlot

from .io import derive_int

RANK = "rank_encoding"
SYNONYM = "synonym_slot_codebook"
ATTACKS = (RANK, SYNONYM)


def other_family(attack_name: str) -> str:
    if attack_name == RANK:
        return SYNONYM
    if attack_name == SYNONYM:
        return RANK
    raise ValueError(f"Unsupported adaptive attack family: {attack_name}")


def assign_initial_channels(parents: list[dict[str, Any]]) -> dict[str, str]:
    cells: dict[tuple[str, str, int], list[str]] = {}
    for row in parents:
        key = (
            str(row["source_model"]),
            str(row["domain"]),
            int(row["nominal_target_length"]),
        )
        cells.setdefault(key, []).append(str(row["sample_id"]))

    result: dict[str, str] = {}
    for key, ids in sorted(cells.items()):
        ordered = sorted(ids)
        if len(ordered) != 6:
            raise ValueError(f"Balanced cell {key} must contain 6 clean parents; found {len(ordered)}")
        for sample_id in ordered[:3]:
            result[sample_id] = RANK
        for sample_id in ordered[3:]:
            result[sample_id] = SYNONYM
    return result


def rank_strategy_seed(*, protocol: str, parent_id: str, static_seed: int, strategy_index: int) -> int:
    if strategy_index == 0:
        return int(static_seed)
    return derive_int(protocol, parent_id, RANK, "strategy_seed", strategy_index, static_seed)


def synonym_strategy_masks(payload_bits: int, budget: int) -> list[int]:
    if payload_bits < 1:
        raise ValueError("payload_bits must be positive")
    unique = 1 << int(payload_bits)
    return list(range(min(int(budget), unique)))


def transformed_codebook(
    codebook: SynonymCodebook,
    *,
    active_slot_ids: list[str],
    mapping_mask: int,
) -> SynonymCodebook:
    active_index = {slot_id: index for index, slot_id in enumerate(active_slot_ids)}
    slots: list[SynonymSlot] = []
    for slot in codebook.slots:
        index = active_index.get(slot.slot_id)
        if index is not None and ((int(mapping_mask) >> index) & 1):
            slots.append(replace(slot, zero=slot.one, one=slot.zero))
        else:
            slots.append(slot)
    return SynonymCodebook(version=f"{codebook.version}+adaptive_mask_{mapping_mask}", slots=tuple(slots))


def next_action(
    *,
    attacker_condition: str,
    initial_family: str,
    prior_actions: list[dict[str, Any]],
    prior_feedback: list[dict[str, Any]],
    available_indices: dict[str, list[int]],
) -> tuple[str, int, str]:
    """Return family, strategy index, and reason for the next round.

    The policy deliberately uses only prior bounded feedback. Researcher-only
    detector scores and utility fields are not accepted by this function.
    """
    if not prior_actions:
        return initial_family, 0, "initial_strategy"

    if attacker_condition == "frozen_no_feedback":
        return initial_family, 0, "frozen_control"

    last_action = prior_actions[-1]
    last_feedback = prior_feedback[-1]
    family = str(last_action["attack_name"])
    strategy = int(last_action["strategy_index"])
    survived = bool(last_feedback["exact_payload_recovery"])

    if survived:
        return family, strategy, "win_stay"

    if attacker_condition == "fixed_channel_feedback":
        indices = list(map(int, available_indices[family]))
        if not indices:
            raise ValueError(f"No candidates available for {family}")
        if strategy not in indices:
            raise ValueError(f"Prior strategy {strategy} not in available candidates for {family}")
        position = indices.index(strategy)
        return family, indices[(position + 1) % len(indices)], "lose_advance_same_family"

    if attacker_condition == "channel_migration_feedback":
        target = other_family(family)
        indices = list(map(int, available_indices[target]))
        if not indices:
            raise ValueError(f"No candidates available for migration target {target}")
        previous_target_indices = [
            int(action["strategy_index"])
            for action in prior_actions
            if str(action["attack_name"]) == target
        ]
        if not previous_target_indices:
            target_index = indices[0]
        else:
            prior = previous_target_indices[-1]
            if prior not in indices:
                target_index = indices[0]
            else:
                target_index = indices[(indices.index(prior) + 1) % len(indices)]
        return target, target_index, "lose_switch_family"

    raise ValueError(f"Unsupported attacker condition: {attacker_condition}")
