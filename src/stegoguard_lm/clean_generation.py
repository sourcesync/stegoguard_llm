from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_record(record: Dict[str, Any], path: str | Path) -> None:
    """Append one JSONL record and force it to disk before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def has_complete_ending(text: str) -> bool:
    stripped = text.rstrip()

    if not stripped:
        return False

    return bool(
        re.search(
            r"[.!?][\"')\]]*$",
            stripped,
        )
    )


def model_slug(model_id: str) -> str:
    slug = model_id.lower()
    slug = slug.replace("/", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def short_model_slug(model_id: str) -> str:
    if "qwen" in model_id.lower():
        return "qwen"
    if "mistral" in model_id.lower():
        return "mistral"
    return model_slug(model_id)[:20]


def load_model_and_tokenizer(model_id: str):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    model.eval()

    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")

    return model, tokenizer, model_revision, tokenizer_revision


def encode_chat_or_plain_prompt(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    """
    Use a chat template if the tokenizer has one.
    Otherwise fall back to plain text prompting.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return tokenizer(prompt, return_tensors="pt")


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def select_complete_prefix_near_target(
    tokenizer,
    text: str,
    target_length: int,
    tolerance: int,
) -> tuple[str, int] | None:
    lower_bound = target_length - tolerance
    upper_bound = target_length + tolerance

    candidates: list[tuple[int, int, str]] = []

    sentence_end_pattern = re.compile(r'[.!?]["\')\]]*(?=\s|$)')

    for match in sentence_end_pattern.finditer(text):
        candidate_text = text[: match.end()].strip()
        candidate_length = token_count(
            tokenizer,
            candidate_text,
        )

        if lower_bound <= candidate_length <= upper_bound:
            distance = abs(candidate_length - target_length)

            candidates.append(
                (
                    distance,
                    -candidate_length,
                    candidate_text,
                )
            )

    if not candidates:
        return None

    _, _, selected_text = min(candidates)
    selected_length = token_count(
        tokenizer,
        selected_text,
    )

    return selected_text, selected_length


def generate_length_controlled_clean_text(
    *,
    model,
    tokenizer,
    prompt: str,
    target_length: int,
    temperature: float,
    top_p: float,
    max_regeneration_attempts: int,
    extra_new_tokens: int,
    length_tolerance: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Generate a clean response whose complete-sentence prefix falls within
    a controlled band around the nominal target length.

    The accepted clean response records its actual source-tokenizer length.
    A later steganographic descendant must match that actual length exactly.
    """
    last_error = None

    for attempt in range(max_regeneration_attempts):
        current_seed = seed + attempt
        torch.manual_seed(current_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        encoded = encode_chat_or_plain_prompt(tokenizer, prompt)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        input_len = encoded["input_ids"].shape[-1]

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    min_new_tokens=target_length,
                    max_new_tokens=target_length + extra_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = output_ids[0][input_len:]
            raw_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            raw_token_length = token_count(tokenizer, raw_text)

            selected = select_complete_prefix_near_target(
                tokenizer=tokenizer,
                text=raw_text,
                target_length=target_length,
                tolerance=length_tolerance,
            )

            if selected is None:
                last_error = (
                    f"Attempt {attempt + 1}: no complete sentence "
                    f"within target band "
                    f"[{target_length - length_tolerance}, "
                    f"{target_length + length_tolerance}]."
                )
                continue

            final_text, final_token_length = selected
            ending_valid = has_complete_ending(final_text)

            if not ending_valid:
                last_error = (
                    f"Attempt {attempt + 1}: selected prefix did not pass "
                    f"the complete-ending check."
                )
                continue

            return {
                "text": final_text,
                "actual_token_length": final_token_length,
                "raw_token_length_before_adjustment": raw_token_length,
                "length_adjustment": "complete_prefix_near_target",
                "length_delta": final_token_length - target_length,
                "length_tolerance": length_tolerance,
                "length_policy": "complete_sentence_near_nominal_target",
                "generation_attempts": attempt + 1,
                "generation_seed": current_seed,
                "ending_valid": True,
            }

        except Exception as exc:
            last_error = f"Attempt {attempt + 1} failed with error: {exc}"
            continue

    raise RuntimeError(
        f"Failed to generate length-controlled text after "
        f"{max_regeneration_attempts} attempts. Last error: {last_error}"
    )


def filter_standard_prompts(
    prompts: List[Dict[str, Any]],
    *,
    domains: List[str],
    lengths: List[int],
) -> List[Dict[str, Any]]:
    selected = []

    allowed_domains = {d.lower() for d in domains}
    allowed_lengths = set(int(x) for x in lengths)

    for record in prompts:
        if record.get("prompt_type") != "standard":
            continue
        if record.get("domain", "").lower() not in allowed_domains:
            continue
        if int(record.get("target_length")) not in allowed_lengths:
            continue
        selected.append(record)

    return selected


def make_clean_sample_ids(
    *,
    prompt_record: Dict[str, Any],
    source_model: str,
    replicate_index: int,
) -> tuple[str, str]:
    """Return deterministic sample_id and pair_id before generation begins."""
    topic_id = prompt_record["topic_id"]
    target_length = int(prompt_record["target_length"])
    source_short = short_model_slug(source_model)

    pair_id = f"{topic_id}_{target_length}_{source_short}_" f"{replicate_index:03d}"
    sample_id = f"clean_{pair_id}"
    return sample_id, pair_id


def make_clean_sample_record(
    *,
    prompt_record: Dict[str, Any],
    source_model: str,
    model_revision: Optional[str],
    tokenizer_revision: Optional[str],
    manifest_version: str,
    text: str,
    actual_token_length: int,
    raw_token_length_before_adjustment: int,
    length_adjustment: str,
    length_tolerance: int,
    length_policy: str,
    ending_valid: bool,
    generation_attempts: int,
    generation_seed: int,
    temperature: float,
    top_p: float,
    replicate_index: int,
) -> Dict[str, Any]:
    topic_id = prompt_record["topic_id"]
    target_length = int(prompt_record["target_length"])
    sample_id, pair_id = make_clean_sample_ids(
        prompt_record=prompt_record,
        source_model=source_model,
        replicate_index=replicate_index,
    )

    return {
        # Sample identity and lineage
        "sample_id": sample_id,
        "pair_id": pair_id,
        "lineage_id": pair_id,
        # Experimental role
        "label": "clean",
        "experiment_stage": "clean_baseline",
        "round_id": 0,
        # Experimental condition
        "domain": prompt_record["domain"],
        "topic_id": topic_id,
        "topic_seed": prompt_record["topic_seed"],
        "target_length": target_length,
        "actual_token_length": actual_token_length,
        # Model provenance
        "source_model": source_model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        # Content and prompt provenance
        "text": text,
        "prompt": prompt_record["visible_prompt"],
        "prompt_id": prompt_record["prompt_id"],
        "prompt_type": prompt_record.get("prompt_type"),
        "template_version": prompt_record["template_version"],
        "replicate_index": replicate_index,
        "manifest_version": manifest_version,
        # Generation configuration
        "seed": generation_seed,
        "temperature": temperature,
        "top_p": top_p,
        # Generation outcome
        "raw_token_length_before_adjustment": raw_token_length_before_adjustment,
        "length_adjustment": length_adjustment,
        "ending_valid": ending_valid,
        "generation_attempts": generation_attempts,
        "nominal_target_length": target_length,
        "length_delta": actual_token_length - target_length,
        "length_tolerance": length_tolerance,
        "length_policy": length_policy,
        "length_valid": (abs(actual_token_length - target_length) <= length_tolerance),
        "generation_status": "accepted",
    }


def write_clean_preview(
    records: List[Dict[str, Any]], path: str | Path, limit: int = 10
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("StegoGuard-LM Clean Generation Preview\n")
        f.write("=" * 50 + "\n\n")

        for record in records[:limit]:
            f.write(f"sample_id: {record['sample_id']}\n")
            f.write(f"source_model: {record['source_model']}\n")
            f.write(f"domain: {record['domain']}\n")
            f.write(f"topic_id: {record['topic_id']}\n")
            f.write(f"target_length: {record['target_length']}\n")
            f.write(f"actual_token_length: {record['actual_token_length']}\n")
            f.write("prompt:\n")
            f.write(record["prompt"] + "\n\n")
            f.write("text:\n")
            f.write(record["text"].strip() + "\n")
            f.write("\n" + "-" * 50 + "\n\n")
