from __future__ import annotations

import hashlib


def validate_bitstring(payload: str) -> None:
    if not isinstance(payload, str):
        raise TypeError("Payload must be a string.")
    if not payload:
        raise ValueError("Payload is empty.")
    if any(bit not in {"0", "1"} for bit in payload):
        raise ValueError("Payload must contain only '0' and '1'.")


def derive_payload_bits(
    *,
    manifest_version: str,
    parent_sample_id: str,
    attack_name: str,
    base_seed: int,
    num_bits: int,
) -> str:
    """Derive a reproducible payload without relying on Python's salted hash."""
    if num_bits < 1:
        raise ValueError("num_bits must be positive.")

    material = (
        f"{manifest_version}|{parent_sample_id}|{attack_name}|{base_seed}"
    ).encode("utf-8")

    output_bits = ""
    counter = 0
    while len(output_bits) < num_bits:
        digest = hashlib.sha256(
            material + counter.to_bytes(4, byteorder="big", signed=False)
        ).digest()
        output_bits += "".join(f"{byte:08b}" for byte in digest)
        counter += 1

    return output_bits[:num_bits]


def payload_sha256(payload: str) -> str:
    validate_bitstring(payload)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def derive_deterministic_int(
    *,
    namespace: str,
    parent_sample_id: str,
    base_seed: int,
    modulo: int = 2**31 - 1,
) -> int:
    if modulo < 2:
        raise ValueError("modulo must be at least 2.")

    material = f"{namespace}|{parent_sample_id}|{base_seed}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value % modulo
