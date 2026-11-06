from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackResult
from .payloads import validate_bitstring

_COMPLETE_ENDING = re.compile(r"[.!?][\"'’”)\]]*$")


@dataclass(slots=True)
class RankDecodeResult:
    payload: str
    success: bool
    observed_ranks: list[int]
    errors: list[str]


def keyed_rank_offset(
    *,
    attack_seed: int,
    encoding_nonce: int,
    token_position: int,
    prompt_fingerprint: str,
    prefix_token_ids: list[int],
) -> int:
    material = (
        f"{attack_seed}|{encoding_nonce}|{token_position}|{prompt_fingerprint}|"
        + ",".join(str(token_id) for token_id in prefix_token_ids)
    ).encode("utf-8")
    return hashlib.sha256(material).digest()[0] & 1


def bit_for_rank(rank: int, offset: int) -> str:
    if rank < 0:
        raise ValueError("rank must be nonnegative.")
    if offset not in {0, 1}:
        raise ValueError("offset must be 0 or 1.")
    return str((rank & 1) ^ offset)


def _has_complete_ending(text: str) -> bool:
    return bool(_COMPLETE_ENDING.search(text.rstrip()))


def _model_input_device(model) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        if device.type != "meta":
            return device
    except Exception:
        pass

    try:
        device = model.device
        if device.type != "meta":
            return device
    except Exception:
        pass

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _encode_prompt(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    encoded = tokenizer(prompt, return_tensors="pt")
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return encoded


def _stable_top_candidates(
    *,
    logits: torch.Tensor,
    tokenizer,
    top_k: int,
    prefix_token_ids: list[int],
    require_complete_ending: bool,
    final_candidate_search_k: int,
) -> list[int]:
    if top_k < 2:
        raise ValueError("top_k must be at least 2 for a binary channel.")

    one_dimensional = logits.detach().float().reshape(-1)
    vocab_size = int(one_dimensional.numel())
    if vocab_size < top_k:
        raise RuntimeError(
            f"Vocabulary has {vocab_size} entries, smaller than top_k={top_k}."
        )

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    initial_search = final_candidate_search_k if require_complete_ending else top_k * 4
    search_size = min(vocab_size, max(top_k, initial_search))

    while True:
        values, ids = torch.topk(one_dimensional, k=search_size, largest=True)
        pairs = sorted(
            zip(ids.cpu().tolist(), values.cpu().tolist(), strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )

        candidates: list[int] = []
        for token_id, _ in pairs:
            if token_id in special_ids:
                continue
            surface = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not surface:
                continue

            if require_complete_ending:
                candidate_text = tokenizer.decode(
                    prefix_token_ids + [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not _has_complete_ending(candidate_text):
                    continue

            candidates.append(token_id)
            if len(candidates) == top_k:
                return candidates

        if search_size == vocab_size:
            break
        search_size = min(vocab_size, search_size * 2)

    condition = "sentence-ending " if require_complete_ending else ""
    raise RuntimeError(
        f"Could not find {top_k} eligible {condition}candidates in the vocabulary."
    )


def _initial_state(model, tokenizer, prompt: str):
    device = _model_input_device(model)
    encoded = _encode_prompt(tokenizer, prompt)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(
            **encoded,
            use_cache=True,
            return_dict=True,
        )

    return (
        outputs.logits[0, -1, :],
        outputs.past_key_values,
        encoded["attention_mask"],
        device,
    )


def _advance_state(
    *,
    model,
    token_id: int,
    past_key_values,
    attention_mask: torch.Tensor,
    device: torch.device,
):
    token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
    next_attention = torch.cat(
        [
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=token_tensor,
            attention_mask=next_attention,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    return outputs.logits[0, -1, :], outputs.past_key_values, next_attention


def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def decode_rank_payload(
    *,
    model,
    tokenizer,
    prompt: str,
    stego_text: str,
    attack_seed: int,
    encoding_nonce: int,
    top_k: int,
    final_candidate_search_k: int = 128,
    require_complete_ending: bool = True,
) -> RankDecodeResult:
    observed_ids = tokenizer.encode(stego_text, add_special_tokens=False)
    if not observed_ids:
        return RankDecodeResult(
            payload="",
            success=False,
            observed_ranks=[],
            errors=["Stego text retokenized to an empty sequence."],
        )

    prompt_fingerprint = _prompt_fingerprint(prompt)
    bits: list[str] = []
    ranks: list[int] = []
    errors: list[str] = []

    logits, past, attention_mask, device = _initial_state(model, tokenizer, prompt)
    prefix: list[int] = []

    for position, observed_id in enumerate(observed_ids):
        is_final = position == len(observed_ids) - 1
        try:
            candidates = _stable_top_candidates(
                logits=logits,
                tokenizer=tokenizer,
                top_k=top_k,
                prefix_token_ids=prefix,
                require_complete_ending=is_final and require_complete_ending,
                final_candidate_search_k=final_candidate_search_k,
            )
        except Exception as exc:
            errors.append(f"Position {position}: candidate construction failed: {exc}")
            break

        if observed_id not in candidates:
            errors.append(
                f"Position {position}: observed token {observed_id} is not in "
                f"the eligible top-{top_k} candidate set."
            )
            break

        rank = candidates.index(observed_id)
        offset = keyed_rank_offset(
            attack_seed=attack_seed,
            encoding_nonce=encoding_nonce,
            token_position=position,
            prompt_fingerprint=prompt_fingerprint,
            prefix_token_ids=prefix,
        )
        bits.append(bit_for_rank(rank, offset))
        ranks.append(rank)
        prefix.append(observed_id)

        if not is_final:
            logits, past, attention_mask = _advance_state(
                model=model,
                token_id=observed_id,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
            )

    return RankDecodeResult(
        payload="".join(bits) if not errors else "",
        success=not errors and len(bits) == len(observed_ids),
        observed_ranks=ranks,
        errors=errors,
    )


def encode_rank_payload(
    *,
    model,
    tokenizer,
    prompt: str,
    payload: str,
    required_token_length: int,
    attack_seed: int,
    top_k: int,
    max_encoding_attempts: int,
    final_candidate_search_k: int = 128,
    require_complete_ending: bool = True,
    require_tokenizer_roundtrip: bool = True,
) -> AttackResult:
    validate_bitstring(payload)
    if len(payload) != required_token_length:
        raise ValueError(
            "Full-capacity rank encoding requires one payload bit per output "
            f"token: payload={len(payload)}, required={required_token_length}."
        )

    prompt_fingerprint = _prompt_fingerprint(prompt)
    last_error = "No encoding attempt was made."

    for nonce in range(max_encoding_attempts):
        try:
            logits, past, attention_mask, device = _initial_state(
                model,
                tokenizer,
                prompt,
            )
            generated_ids: list[int] = []
            chosen_ranks: list[int] = []

            for position, desired_bit in enumerate(payload):
                is_final = position == required_token_length - 1
                candidates = _stable_top_candidates(
                    logits=logits,
                    tokenizer=tokenizer,
                    top_k=top_k,
                    prefix_token_ids=generated_ids,
                    require_complete_ending=is_final and require_complete_ending,
                    final_candidate_search_k=final_candidate_search_k,
                )
                offset = keyed_rank_offset(
                    attack_seed=attack_seed,
                    encoding_nonce=nonce,
                    token_position=position,
                    prompt_fingerprint=prompt_fingerprint,
                    prefix_token_ids=generated_ids,
                )

                selected_id = None
                selected_rank = None
                for rank, candidate_id in enumerate(candidates):
                    if bit_for_rank(rank, offset) == desired_bit:
                        selected_id = candidate_id
                        selected_rank = rank
                        break

                if selected_id is None or selected_rank is None:
                    raise RuntimeError(
                        f"No candidate represents bit {desired_bit} at position {position}."
                    )

                generated_ids.append(selected_id)
                chosen_ranks.append(selected_rank)

                if not is_final:
                    logits, past, attention_mask = _advance_state(
                        model=model,
                        token_id=selected_id,
                        past_key_values=past,
                        attention_mask=attention_mask,
                        device=device,
                    )

            text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            retokenized = tokenizer.encode(text, add_special_tokens=False)
            if len(retokenized) != required_token_length:
                last_error = (
                    "length_mismatch: decoded text retokenizes to "
                    f"{len(retokenized)} tokens, expected {required_token_length}."
                )
                continue
            if require_tokenizer_roundtrip and retokenized != generated_ids:
                last_error = "tokenizer_roundtrip_failure: token IDs changed after decode/encode."
                continue
            if require_complete_ending and not _has_complete_ending(text):
                last_error = (
                    "incomplete_ending: rank output lacks sentence punctuation."
                )
                continue

            decoded = decode_rank_payload(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                stego_text=text,
                attack_seed=attack_seed,
                encoding_nonce=nonce,
                top_k=top_k,
                final_candidate_search_k=final_candidate_search_k,
                require_complete_ending=require_complete_ending,
            )
            if not decoded.success:
                last_error = "decode_failure: " + " | ".join(decoded.errors)
                continue
            if decoded.payload != payload:
                last_error = (
                    "payload_mismatch: decoded payload differs from expected payload."
                )
                continue

            return AttackResult(
                attack_family="token_whitebox",
                attack_name="rank_encoding",
                attacker_access="white_box",
                strategy_id="rank_encoding",
                strategy_version="keyed_parity_v1",
                stego_text=text,
                payload=payload,
                decoded_payload=decoded.payload,
                decode_success=True,
                exact_match=True,
                capacity_bits=required_token_length,
                payload_bit_length=len(payload),
                encoding_attempts=nonce + 1,
                source_token_length=len(retokenized),
                length_invariant="source_tokenizer_exact",
                length_valid=len(retokenized) == required_token_length,
                channel_parameters={
                    "attack_seed": attack_seed,
                    "encoding_nonce": nonce,
                    "top_k": top_k,
                    "rank_partition": "keyed_parity",
                    "strategy_version": "keyed_parity_v1",
                    "bits_per_encoded_token": 1.0,
                    "required_token_length": required_token_length,
                    "final_token_policy": "sentence_ending_candidates",
                    "final_candidate_search_k": final_candidate_search_k,
                    "generated_token_ids": generated_ids,
                    "chosen_ranks": chosen_ranks,
                    "decoded_ranks": decoded.observed_ranks,
                    "tokenizer_roundtrip_valid": retokenized == generated_ids,
                    "ending_valid": _has_complete_ending(text),
                    "prompt_fingerprint": prompt_fingerprint,
                },
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        "Rank encoding failed after "
        f"{max_encoding_attempts} nonces. Last error: {last_error}"
    )
