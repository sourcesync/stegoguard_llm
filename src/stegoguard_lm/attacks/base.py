from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AttackResult:
    """Normalized result returned by every successful attack encoder."""

    attack_family: str
    attack_name: str
    attacker_access: str
    strategy_id: str
    strategy_version: str
    stego_text: str
    payload: str
    decoded_payload: str
    decode_success: bool
    exact_match: bool
    capacity_bits: int
    payload_bit_length: int
    encoding_attempts: int
    source_token_length: int | None
    length_invariant: str
    length_valid: bool
    channel_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AttackFailure:
    """Structured failure information for resumable attack generation."""

    attack_family: str
    attack_name: str
    parent_sample_id: str
    error_type: str
    error_message: str
    encoding_attempts: int
    channel_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
