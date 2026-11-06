from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .base import AttackResult
from .payloads import derive_deterministic_int, derive_payload_bits
from .rank_channel import encode_rank_payload
from .records import build_attack_record, make_attack_sample_id
from .synonym_channel import (
    SynonymCodebook,
    discover_inplace_synonym_slots,
    encode_synonym_payload,
)
from .unicode_channel import encode_unicode_payload
from .validation import validate_attack_record_against_parent

GENERATION_PROTOCOL_VERSION = "step7_static_v1"
ATTACK_FAMILY_ORDER = ("out_of_band", "token_blackbox", "token_whitebox")
FINAL_TASK_STATUSES = {"accepted", "structural_zero_capacity", "terminal_encoding_failure"}
STEP7_IMPLEMENTATION_PATHS = (
    "src/stegoguard_lm/attacks/full_generation.py",
    "src/stegoguard_lm/attacks/payloads.py",
    "src/stegoguard_lm/attacks/unicode_channel.py",
    "src/stegoguard_lm/attacks/synonym_channel.py",
    "src/stegoguard_lm/attacks/rank_channel.py",
    "src/stegoguard_lm/attacks/records.py",
    "src/stegoguard_lm/attacks/validation.py",
    "scripts/generate_stego_samples.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: str | Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object."
                )
            records.append(value)
    return records


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def append_jsonl_record(record: dict[str, Any], path: str | Path) -> None:
    """Append one durable JSONL record.

    ensure_ascii=True intentionally makes hidden Unicode channel codepoints visible
    in the stored artifact while preserving their exact JSON representation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def atomic_write_json(value: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=True, sort_keys=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def atomic_write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def attack_protocol_subset(config: dict[str, Any]) -> dict[str, Any]:
    attacks = config["attacks"]
    full = config["full_attack_generation"]
    return {
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "manifest_version": config["metadata"]["manifest_version"],
        "payload": attacks["payload"],
        "out_of_band": attacks["out_of_band"],
        "token_blackbox": attacks["token_blackbox"],
        "token_whitebox": attacks["token_whitebox"],
        "full_attack_generation": full,
    }


def attack_protocol_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(attack_protocol_subset(config))


def step7_implementation_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    hashes: dict[str, str] = {}
    for relative in STEP7_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Step 7 implementation file not found: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def verify_frozen_clean_corpus(
    *,
    clean_path: str | Path,
    step5_manifest_path: str | Path,
    expected_record_count: int,
) -> str:
    step5_manifest = read_json(step5_manifest_path)
    actual_hash = sha256_file(clean_path)
    expected_hash = str(step5_manifest.get("sha256", "")).casefold()
    if not expected_hash:
        raise ValueError("Step 5 manifest does not contain a sha256 value.")
    if actual_hash.casefold() != expected_hash:
        raise ValueError(
            "Frozen clean corpus hash mismatch: "
            f"actual={actual_hash}, expected={expected_hash}."
        )

    records = read_jsonl(clean_path)
    if len(records) != expected_record_count:
        raise ValueError(
            f"Expected {expected_record_count} frozen clean records, found {len(records)}."
        )
    if int(step5_manifest.get("record_count", -1)) != expected_record_count:
        raise ValueError(
            "Step 5 manifest record_count does not match the Step 7 expected count."
        )
    return actual_hash


def _index_unique(records: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get(field, ""))
        if not key:
            raise ValueError(f"{label} record lacks {field}: {record}")
        if key in index:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        index[key] = record
    return index


def validate_synonym_capacity_audit(
    *,
    capacity_audit: dict[str, Any],
    clean_records: list[dict[str, Any]],
    expected_record_count: int,
) -> dict[str, dict[str, Any]]:
    if str(capacity_audit.get("matrix")) != "full":
        raise ValueError("Synonym capacity audit must be the full 160-record matrix.")
    if int(capacity_audit.get("record_count", -1)) != expected_record_count:
        raise ValueError(
            "Synonym capacity audit record_count does not match the frozen corpus."
        )

    rows = capacity_audit.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Synonym capacity audit must contain a rows list.")
    if len(rows) != expected_record_count:
        raise ValueError(
            f"Expected {expected_record_count} synonym capacity rows, found {len(rows)}."
        )

    clean_by_id = _index_unique(clean_records, "sample_id", "clean")
    row_by_id = _index_unique(rows, "sample_id", "synonym capacity")
    if set(clean_by_id) != set(row_by_id):
        missing = sorted(set(clean_by_id) - set(row_by_id))
        extra = sorted(set(row_by_id) - set(clean_by_id))
        raise ValueError(
            "Synonym capacity audit sample IDs do not match the frozen clean corpus. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    positive = 0
    zero = 0
    distribution: Counter[int] = Counter()
    for sample_id, clean in clean_by_id.items():
        row = row_by_id[sample_id]
        expected_relations = {
            "source_model": clean["source_model"],
            "domain": clean["domain"],
            "nominal_target_length": int(clean["target_length"]),
            "actual_token_length": int(clean["actual_token_length"]),
        }
        for field, expected in expected_relations.items():
            observed = row.get(field)
            if field.endswith("length"):
                observed = int(observed)
            if observed != expected:
                raise ValueError(
                    f"Synonym capacity row {sample_id} has {field}={observed!r}; "
                    f"expected {expected!r}."
                )

        capacity_bits = int(row.get("capacity_bits", -1))
        slot_ids = row.get("eligible_slot_ids")
        if not isinstance(slot_ids, list):
            raise ValueError(f"Synonym capacity row {sample_id} lacks eligible_slot_ids.")
        if capacity_bits != len(slot_ids):
            raise ValueError(
                f"Synonym capacity row {sample_id} reports capacity={capacity_bits} "
                f"but contains {len(slot_ids)} eligible slots."
            )
        if bool(row.get("has_capacity")) != (capacity_bits > 0):
            raise ValueError(f"Synonym capacity has_capacity mismatch for {sample_id}.")

        distribution[capacity_bits] += 1
        if capacity_bits > 0:
            positive += 1
        else:
            zero += 1

    if int(capacity_audit.get("records_with_capacity", -1)) != positive:
        raise ValueError("Synonym capacity audit records_with_capacity is inconsistent.")
    if int(capacity_audit.get("records_without_capacity", -1)) != zero:
        raise ValueError("Synonym capacity audit records_without_capacity is inconsistent.")

    stored_distribution = {
        int(key): int(value)
        for key, value in dict(capacity_audit.get("capacity_distribution", {})).items()
    }
    if dict(sorted(distribution.items())) != dict(sorted(stored_distribution.items())):
        raise ValueError("Synonym capacity distribution is internally inconsistent.")

    return row_by_id


def _strategy_fields(settings: dict[str, Any], family: str) -> tuple[str, str, str]:
    if family == "token_blackbox":
        return (
            str(settings.get("attacker_access", "black_box_tokenizer_aware")),
            str(settings.get("strategy_id", "synonym_inplace_substitution")),
            str(settings.get("strategy_version", "synonym_slots_v1+inplace_v1")),
        )
    if family == "token_whitebox":
        return (
            str(settings.get("attacker_access", "white_box")),
            "rank_encoding",
            "keyed_parity_v1",
        )
    if family == "out_of_band":
        version = str(settings.get("framing_version", "unicode_frame_v1"))
        return (
            str(settings.get("attacker_access", "text_transformation")),
            "unicode_hidden_channel",
            version,
        )
    raise ValueError(f"Unsupported attack family: {family}")


def build_attack_task_manifest(
    *,
    config: dict[str, Any],
    clean_records: list[dict[str, Any]],
    capacity_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_config = config["full_attack_generation"]
    expected_clean = int(full_config["expected_clean_parents"])
    expected_opportunities = int(full_config["expected_attack_opportunities"])
    if len(clean_records) != expected_clean:
        raise ValueError(
            f"Expected {expected_clean} clean parents, found {len(clean_records)}."
        )

    clean_by_id = _index_unique(clean_records, "sample_id", "clean")
    capacity_by_id = validate_synonym_capacity_audit(
        capacity_audit=capacity_audit,
        clean_records=clean_records,
        expected_record_count=expected_clean,
    )

    manifest_version = str(config["metadata"]["manifest_version"])
    attacks_config = config["attacks"]
    base_seed = int(attacks_config["payload"]["base_seed"])
    protocol_hash = attack_protocol_sha256(config)

    tasks: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []
    task_index = 0

    for parent in clean_records:
        parent_id = str(parent["sample_id"])
        parent_actual = int(parent["actual_token_length"])
        capacity_row = capacity_by_id[parent_id]

        for family in ATTACK_FAMILY_ORDER:
            settings = attacks_config[family]
            if not bool(settings.get("enabled", True)):
                raise ValueError(
                    f"Step 7 frozen protocol requires {family} to remain enabled."
                )

            attack_name = str(settings["name"])
            attacker_access, strategy_id, strategy_version = _strategy_fields(
                settings, family
            )

            eligible = True
            eligibility_status = "eligible"
            eligibility_reason = None
            eligible_slot_ids: list[str] = []

            if family == "token_whitebox":
                available_capacity_bits = parent_actual
                expected_payload_bits = parent_actual
                payload_policy = "one_bit_per_generated_token"
                attempt_budget = int(settings["max_encoding_attempts"])
            elif family == "out_of_band":
                available_capacity_bits = int(settings["payload_bits"])
                expected_payload_bits = int(settings["payload_bits"])
                payload_policy = "fixed_configured_payload_bits"
                attempt_budget = 1
            elif family == "token_blackbox":
                available_capacity_bits = int(capacity_row["capacity_bits"])
                eligible_slot_ids = [
                    str(slot_id) for slot_id in capacity_row["eligible_slot_ids"]
                ]
                expected_payload_bits = available_capacity_bits
                payload_policy = "use_full_available_capacity"
                attempt_budget = int(settings.get("payload_nonce_attempts", 16))
                if available_capacity_bits == 0:
                    eligible = False
                    eligibility_status = "structural_zero_capacity"
                    eligibility_reason = "no_tokenizer_stable_codebook_slot"
            else:  # pragma: no cover - guarded by family constant
                raise ValueError(f"Unsupported attack family: {family}")

            attack_seed = derive_deterministic_int(
                namespace=attack_name,
                parent_sample_id=parent_id,
                base_seed=base_seed,
            )
            task = {
                "task_index": task_index,
                "task_id": f"{attack_name}::{parent_id}",
                "expected_sample_id": make_attack_sample_id(parent, attack_name),
                "parent_sample_id": parent_id,
                "pair_id": parent["pair_id"],
                "lineage_id": parent["lineage_id"],
                "attack_family": family,
                "attack_name": attack_name,
                "attacker_access": attacker_access,
                "strategy_id": strategy_id,
                "strategy_version": strategy_version,
                "source_model": parent["source_model"],
                "domain": parent["domain"],
                "topic_id": parent["topic_id"],
                "nominal_target_length": int(parent["target_length"]),
                "parent_actual_token_length": parent_actual,
                "eligible": eligible,
                "eligibility_status": eligibility_status,
                "eligibility_reason": eligibility_reason,
                "available_capacity_bits": available_capacity_bits,
                "expected_payload_bits": expected_payload_bits,
                "payload_policy": payload_policy,
                "expected_payload_rate_bpt": (
                    expected_payload_bits / parent_actual
                    if eligible and parent_actual > 0
                    else 0.0
                ),
                "synonym_eligible_slot_ids": eligible_slot_ids,
                "attack_seed": attack_seed,
                "attempt_budget": attempt_budget,
                "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
                "attack_protocol_sha256": protocol_hash,
                "manifest_version": manifest_version,
            }
            tasks.append(task)
            if not eligible:
                structural.append(dict(task))
            task_index += 1

    if len(clean_by_id) != expected_clean:
        raise ValueError("Clean sample_id uniqueness check failed.")
    if len(tasks) != expected_opportunities:
        raise ValueError(
            f"Expected {expected_opportunities} task rows, built {len(tasks)}."
        )

    validate_attack_task_manifest(
        tasks=tasks,
        config=config,
        expected_structural_count=len(structural),
    )
    return tasks, structural


def validate_attack_task_manifest(
    *,
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    expected_structural_count: int | None = None,
) -> dict[str, Any]:
    full_config = config["full_attack_generation"]
    expected_clean = int(full_config["expected_clean_parents"])
    expected_opportunities = int(full_config["expected_attack_opportunities"])
    if len(tasks) != expected_opportunities:
        raise ValueError(
            f"Task manifest has {len(tasks)} rows; expected {expected_opportunities}."
        )

    task_by_id = _index_unique(tasks, "task_id", "attack task")
    sample_by_id = _index_unique(tasks, "expected_sample_id", "attack task")
    if len(task_by_id) != len(sample_by_id):
        raise ValueError("Task/sample identifier uniqueness mismatch.")

    per_parent: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    counts_by_family: Counter[str] = Counter()
    structural = 0
    eligible = 0
    for task in tasks:
        per_parent[str(task["parent_sample_id"])].append(task)
        family = str(task["attack_family"])
        counts_by_family[family] += 1
        if bool(task["eligible"]):
            eligible += 1
            if task.get("eligibility_status") != "eligible":
                raise ValueError(f"Eligible task has invalid status: {task['task_id']}")
            if int(task.get("expected_payload_bits", 0)) < 1:
                raise ValueError(f"Eligible task has no payload: {task['task_id']}")
        else:
            structural += 1
            if family != "token_blackbox":
                raise ValueError(
                    f"Only synonym tasks may be structurally ineligible: {task['task_id']}"
                )
            if task.get("eligibility_status") != "structural_zero_capacity":
                raise ValueError(
                    f"Ineligible synonym task has unexpected status: {task['task_id']}"
                )
            if int(task.get("available_capacity_bits", -1)) != 0:
                raise ValueError(
                    f"Structural zero-capacity task reports nonzero capacity: {task['task_id']}"
                )

    if len(per_parent) != expected_clean:
        raise ValueError(
            f"Task manifest covers {len(per_parent)} parents; expected {expected_clean}."
        )
    for parent_id, parent_tasks in per_parent.items():
        families = {str(task["attack_family"]) for task in parent_tasks}
        if families != set(ATTACK_FAMILY_ORDER) or len(parent_tasks) != 3:
            raise ValueError(
                f"Parent {parent_id} does not have exactly one task per attack family."
            )

    expected_family_count = expected_clean
    for family in ATTACK_FAMILY_ORDER:
        if counts_by_family[family] != expected_family_count:
            raise ValueError(
                f"Family {family} has {counts_by_family[family]} tasks; "
                f"expected {expected_family_count}."
            )

    if expected_structural_count is not None and structural != expected_structural_count:
        raise ValueError(
            f"Structural task count {structural} != expected {expected_structural_count}."
        )

    return {
        "attack_opportunities": len(tasks),
        "eligible_tasks": eligible,
        "structural_ineligible_tasks": structural,
        "counts_by_family": dict(sorted(counts_by_family.items())),
        "clean_parents": len(per_parent),
    }


def build_generation_manifest(
    *,
    config: dict[str, Any],
    task_manifest_path: str | Path,
    structural_path: str | Path,
    clean_path: str | Path,
    capacity_audit_path: str | Path,
    codebook_path: str | Path,
    task_stats: dict[str, Any],
    project_root: str | Path = ".",
) -> dict[str, Any]:
    return {
        "manifest_version": str(config["metadata"]["manifest_version"]),
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "status": "frozen_for_generation",
        "expected_clean_parents": int(
            config["full_attack_generation"]["expected_clean_parents"]
        ),
        "expected_attack_opportunities": int(
            config["full_attack_generation"]["expected_attack_opportunities"]
        ),
        **task_stats,
        "synonym_payload_policy": str(
            config["full_attack_generation"]["synonym_payload_policy"]
        ),
        "terminal_failure_policy": "final_after_frozen_channel_attempt_budget",
        "resume_retries_terminal_failures": False,
        "clean_corpus_sha256": sha256_file(clean_path),
        "synonym_capacity_audit_sha256": sha256_file(capacity_audit_path),
        "synonym_codebook_sha256": sha256_file(codebook_path),
        "attack_protocol_sha256": attack_protocol_sha256(config),
        "implementation_files_sha256": step7_implementation_hashes(project_root),
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "structural_ineligible_sha256": sha256_file(structural_path),
        "task_manifest_path": str(task_manifest_path),
        "structural_ineligible_path": str(structural_path),
        "clean_corpus_path": str(clean_path),
        "synonym_capacity_audit_path": str(capacity_audit_path),
        "synonym_codebook_path": str(codebook_path),
    }


def verify_generation_manifest(
    *,
    config: dict[str, Any],
    generation_manifest: dict[str, Any],
    task_manifest_path: str | Path,
    structural_path: str | Path,
    clean_path: str | Path,
    capacity_audit_path: str | Path,
    codebook_path: str | Path,
    project_root: str | Path = ".",
) -> None:
    expected_protocol = attack_protocol_sha256(config)
    checks = {
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "manifest_version": str(config["metadata"]["manifest_version"]),
        "clean_corpus_sha256": sha256_file(clean_path),
        "synonym_capacity_audit_sha256": sha256_file(capacity_audit_path),
        "synonym_codebook_sha256": sha256_file(codebook_path),
        "attack_protocol_sha256": expected_protocol,
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "structural_ineligible_sha256": sha256_file(structural_path),
    }
    for field, expected in checks.items():
        observed = str(generation_manifest.get(field, ""))
        if observed.casefold() != str(expected).casefold():
            raise ValueError(
                f"Step 7 generation manifest drift for {field}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    observed_impl = generation_manifest.get("implementation_files_sha256")
    if not isinstance(observed_impl, dict):
        raise ValueError("Step 7 generation manifest lacks implementation file hashes.")
    expected_impl = step7_implementation_hashes(project_root)
    if {str(k): str(v).casefold() for k, v in observed_impl.items()} != {
        str(k): str(v).casefold() for k, v in expected_impl.items()
    }:
        raise ValueError(
            "Step 7 implementation code has changed since the task universe was frozen. "
            "Create a new experiment/version rather than mixing code versions."
        )


def _derive_fixed_payload(
    *,
    config: dict[str, Any],
    task: dict[str, Any],
    num_bits: int,
    parent_sample_id_override: str | None = None,
) -> str:
    return derive_payload_bits(
        manifest_version=str(config["metadata"]["manifest_version"]),
        parent_sample_id=(
            parent_sample_id_override
            if parent_sample_id_override is not None
            else str(task["parent_sample_id"])
        ),
        attack_name=str(task["attack_name"]),
        base_seed=int(config["attacks"]["payload"]["base_seed"]),
        num_bits=num_bits,
    )


def execute_eligible_task(
    *,
    config: dict[str, Any],
    task: dict[str, Any],
    parent: dict[str, Any],
    tokenizer,
    model=None,
    codebook: SynonymCodebook | None = None,
) -> AttackResult:
    if not bool(task.get("eligible")):
        raise ValueError(f"Cannot execute structurally ineligible task: {task['task_id']}")
    if str(parent["sample_id"]) != str(task["parent_sample_id"]):
        raise ValueError("Task/parent mismatch.")
    if str(parent["source_model"]) != str(task["source_model"]):
        raise ValueError("Task source_model does not match parent.")

    family = str(task["attack_family"])
    settings = config["attacks"][family]
    required_length = int(parent["actual_token_length"])
    expected_bits = int(task["expected_payload_bits"])
    attack_seed = int(task["attack_seed"])

    if family == "out_of_band":
        payload = _derive_fixed_payload(config=config, task=task, num_bits=expected_bits)
        result = encode_unicode_payload(
            clean_text=parent["text"],
            payload=payload,
            tokenizer=tokenizer,
            zero_codepoint=str(settings["bit_zero_codepoint"]),
            one_codepoint=str(settings["bit_one_codepoint"]),
            prefix_codepoint=str(settings["prefix_codepoint"]),
            suffix_codepoint=str(settings["suffix_codepoint"]),
            framing_version=str(settings["framing_version"]),
            embed_position=str(settings["embed_position"]),
        )

    elif family == "token_whitebox":
        if model is None:
            raise ValueError("Rank encoding requires the source model.")
        payload = _derive_fixed_payload(config=config, task=task, num_bits=expected_bits)
        result = encode_rank_payload(
            model=model,
            tokenizer=tokenizer,
            prompt=parent["prompt"],
            payload=payload,
            required_token_length=required_length,
            attack_seed=attack_seed,
            top_k=int(settings["top_k"]),
            max_encoding_attempts=int(settings["max_encoding_attempts"]),
            final_candidate_search_k=int(settings["final_candidate_search_k"]),
            require_complete_ending=bool(settings["require_complete_ending"]),
            require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
        )

    elif family == "token_blackbox":
        if codebook is None:
            raise ValueError("Synonym encoding requires the frozen synonym codebook.")

        eligible_slots = discover_inplace_synonym_slots(
            clean_text=parent["text"],
            tokenizer=tokenizer,
            codebook=codebook,
            required_token_length=required_length,
            require_complete_ending=bool(settings["require_complete_ending"]),
            require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
            forbid_format_codepoints=bool(settings.get("forbid_format_codepoints", True)),
        )
        observed_slot_ids = [slot.slot_id for slot in eligible_slots]
        expected_slot_ids = [str(value) for value in task.get("synonym_eligible_slot_ids", [])]
        if observed_slot_ids != expected_slot_ids:
            raise RuntimeError(
                "capacity_drift: current tokenizer/codebook discovery differs from the "
                f"frozen audit. observed={observed_slot_ids}, expected={expected_slot_ids}."
            )
        if len(observed_slot_ids) != expected_bits:
            raise RuntimeError(
                "capacity_drift: frozen synonym expected_payload_bits does not match "
                f"current eligible slots ({expected_bits} vs {len(observed_slot_ids)})."
            )

        nonce_attempts = int(settings.get("payload_nonce_attempts", 16))
        result = None
        last_error = "payload selection not attempted"
        for payload_nonce in range(nonce_attempts):
            payload = _derive_fixed_payload(
                config=config,
                task=task,
                num_bits=expected_bits,
                parent_sample_id_override=(
                    f"{parent['sample_id']}|synonym_nonce={payload_nonce}"
                ),
            )
            try:
                candidate = encode_synonym_payload(
                    clean_text=parent["text"],
                    tokenizer=tokenizer,
                    payload=payload,
                    codebook=codebook,
                    required_token_length=required_length,
                    max_payload_bits=expected_bits,
                    min_payload_bits=expected_bits,
                    require_text_change=bool(settings.get("require_text_change", True)),
                    forbid_format_codepoints=bool(
                        settings.get("forbid_format_codepoints", True)
                    ),
                    require_complete_ending=bool(settings["require_complete_ending"]),
                    require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
                )
            except RuntimeError as exc:
                last_error = str(exc)
                continue

            if candidate.payload_bit_length != expected_bits:
                last_error = (
                    "payload_length_shortfall: synonym encoder returned "
                    f"{candidate.payload_bit_length} bits; expected {expected_bits}."
                )
                continue
            candidate.channel_parameters.update(
                {
                    "payload_nonce": payload_nonce,
                    "payload_nonce_attempts": payload_nonce + 1,
                    "requested_payload": payload,
                    "frozen_available_capacity_bits": expected_bits,
                    "frozen_eligible_slot_ids": expected_slot_ids,
                }
            )
            result = candidate
            break

        if result is None:
            raise RuntimeError(
                "Synonym full-capacity payload selection failed after "
                f"{nonce_attempts} deterministic nonce attempts. Last error: {last_error}"
            )
    else:
        raise ValueError(f"Unsupported attack family: {family}")

    if result.payload_bit_length != expected_bits:
        raise RuntimeError(
            f"Attack returned {result.payload_bit_length} payload bits; "
            f"task requires {expected_bits}."
        )
    return result


def build_full_attack_record(
    *,
    config: dict[str, Any],
    task: dict[str, Any],
    parent: dict[str, Any],
    result: AttackResult,
) -> dict[str, Any]:
    record = build_attack_record(
        parent_record=parent,
        result=result,
        manifest_version=str(config["metadata"]["manifest_version"]),
        experiment_stage="full_static_attack_generation",
        attack_seed=int(task["attack_seed"]),
    )
    parent_length = int(parent["actual_token_length"])
    record.update(
        {
            "task_id": task["task_id"],
            "task_index": int(task["task_index"]),
            "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
            "eligibility_status": "eligible",
            "available_capacity_bits": int(task["available_capacity_bits"]),
            "expected_payload_bits": int(task["expected_payload_bits"]),
            "payload_policy": task["payload_policy"],
            "payload_rate_bpt": result.payload_bit_length / parent_length,
            "attack_protocol_sha256": task["attack_protocol_sha256"],
            "generation_status": "accepted",
        }
    )
    return record


def validate_full_record_against_task(
    *,
    record: dict[str, Any],
    task: dict[str, Any],
    parent: dict[str, Any],
) -> list[str]:
    errors = validate_attack_record_against_parent(record, parent)
    expected = {
        "task_id": task["task_id"],
        "sample_id": task["expected_sample_id"],
        "parent_sample_id": task["parent_sample_id"],
        "attack_family": task["attack_family"],
        "attack_name": task["attack_name"],
        "source_model": task["source_model"],
        "experiment_stage": "full_static_attack_generation",
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "eligibility_status": "eligible",
        "payload_policy": task["payload_policy"],
        "attack_protocol_sha256": task["attack_protocol_sha256"],
        "generation_status": "accepted",
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            errors.append(
                f"{field} mismatch: {record.get(field)!r} != {expected_value!r}."
            )

    numeric_expected = {
        "task_index": int(task["task_index"]),
        "available_capacity_bits": int(task["available_capacity_bits"]),
        "expected_payload_bits": int(task["expected_payload_bits"]),
        "payload_bit_length": int(task["expected_payload_bits"]),
    }
    for field, expected_value in numeric_expected.items():
        try:
            observed = int(record.get(field, -1))
        except (TypeError, ValueError):
            observed = -1
        if observed != expected_value:
            errors.append(f"{field} mismatch: {observed} != {expected_value}.")

    parent_length = int(parent["actual_token_length"])
    expected_rate = int(task["expected_payload_bits"]) / parent_length
    try:
        observed_rate = float(record.get("payload_rate_bpt"))
    except (TypeError, ValueError):
        observed_rate = -1.0
    if abs(observed_rate - expected_rate) > 1e-12:
        errors.append(
            f"payload_rate_bpt mismatch: {observed_rate} != {expected_rate}."
        )

    if str(record.get("attacker_access")) != str(task["attacker_access"]):
        errors.append("attacker_access does not match frozen task.")
    if str(record.get("strategy_id")) != str(task["strategy_id"]):
        errors.append("strategy_id does not match frozen task.")
    if str(record.get("strategy_version")) != str(task["strategy_version"]):
        errors.append("strategy_version does not match frozen task.")
    return errors


def build_terminal_failure_record(
    *,
    task: dict[str, Any],
    parent: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "task_index": int(task["task_index"]),
        "sample_id": task["expected_sample_id"],
        "parent_sample_id": task["parent_sample_id"],
        "pair_id": parent["pair_id"],
        "lineage_id": parent["lineage_id"],
        "attack_family": task["attack_family"],
        "attack_name": task["attack_name"],
        "source_model": task["source_model"],
        "domain": task["domain"],
        "nominal_target_length": int(task["nominal_target_length"]),
        "parent_actual_token_length": int(task["parent_actual_token_length"]),
        "available_capacity_bits": int(task["available_capacity_bits"]),
        "expected_payload_bits": int(task["expected_payload_bits"]),
        "attempt_budget": int(task["attempt_budget"]),
        "status": "terminal_encoding_failure",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "attack_seed": int(task["attack_seed"]),
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "attack_protocol_sha256": task["attack_protocol_sha256"],
        "manifest_version": task["manifest_version"],
    }


def index_completed_records(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    return _index_unique(records, "task_id", label)


def account_task_inventory(
    *,
    tasks: list[dict[str, Any]],
    accepted_records: list[dict[str, Any]],
    terminal_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    task_by_id = _index_unique(tasks, "task_id", "attack task")
    accepted_by_id = index_completed_records(accepted_records, label="accepted attack")
    failure_by_id = index_completed_records(terminal_failures, label="terminal failure")

    overlap = set(accepted_by_id) & set(failure_by_id)
    if overlap:
        raise ValueError(
            "Tasks cannot be both accepted and terminal failures: "
            + ", ".join(sorted(overlap)[:5])
        )

    unknown_accepted = set(accepted_by_id) - set(task_by_id)
    unknown_failures = set(failure_by_id) - set(task_by_id)
    if unknown_accepted or unknown_failures:
        raise ValueError(
            "Persisted Step 7 outputs contain task IDs absent from the frozen manifest. "
            f"accepted={sorted(unknown_accepted)[:5]}, failures={sorted(unknown_failures)[:5]}"
        )

    structural_ids = {
        task_id for task_id, task in task_by_id.items() if not bool(task["eligible"])
    }
    if structural_ids & set(accepted_by_id):
        raise ValueError("Structurally ineligible tasks appear in accepted outputs.")
    if structural_ids & set(failure_by_id):
        raise ValueError("Structurally ineligible tasks appear in terminal-failure outputs.")

    for task_id, failure in failure_by_id.items():
        if failure.get("status") != "terminal_encoding_failure":
            raise ValueError(f"Failure {task_id} has unexpected status.")

    eligible_ids = set(task_by_id) - structural_ids
    completed_eligible = set(accepted_by_id) | set(failure_by_id)
    pending_ids = eligible_ids - completed_eligible

    accounting_total = len(accepted_by_id) + len(failure_by_id) + len(structural_ids) + len(pending_ids)
    if accounting_total != len(tasks):
        raise ValueError("Step 7 accounting invariant failed.")

    counts_by_attack = Counter(
        record["attack_name"] for record in accepted_records
    )
    terminal_by_attack = Counter(
        record["attack_name"] for record in terminal_failures
    )
    status = (
        "complete_no_terminal_failures"
        if not pending_ids and not failure_by_id
        else "complete_with_terminal_failures"
        if not pending_ids
        else "incomplete"
    )

    return {
        "attack_opportunities": len(tasks),
        "eligible_tasks": len(eligible_ids),
        "structural_ineligible_tasks": len(structural_ids),
        "accepted_tasks": len(accepted_by_id),
        "terminal_encoding_failures": len(failure_by_id),
        "pending_tasks": len(pending_ids),
        "accounting_total": accounting_total,
        "status": status,
        "pending_task_ids": sorted(pending_ids),
        "counts_by_attack": dict(sorted(counts_by_attack.items())),
        "terminal_failures_by_attack": dict(sorted(terminal_by_attack.items())),
    }


def task_matches_filters(
    task: dict[str, Any],
    *,
    attack_families: set[str] | None = None,
    source_models: set[str] | None = None,
    domains: set[str] | None = None,
    nominal_lengths: set[int] | None = None,
) -> bool:
    if attack_families is not None and str(task["attack_family"]) not in attack_families:
        return False
    if source_models is not None and str(task["source_model"]) not in source_models:
        return False
    if domains is not None and str(task["domain"]) not in domains:
        return False
    if nominal_lengths is not None and int(task["nominal_target_length"]) not in nominal_lengths:
        return False
    return True


def select_stratified_preview(
    records: list[dict[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, str, int]] = set()
    for record in records:
        cell = (
            str(record["attack_name"]),
            str(record["source_model"]),
            str(record["domain"]),
            int(record["nominal_target_length"]),
        )
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        selected.append(record)
        if len(selected) >= limit:
            return selected
    return selected


def write_full_preview(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_stratified_preview(records)
    lines: list[str] = []
    for record in selected:
        text = str(record["text"])
        if record["attack_family"] == "out_of_band":
            visible = text.encode("unicode_escape").decode("ascii")
        else:
            visible = text
        lines.extend(
            [
                "=" * 96,
                f"sample_id: {record['sample_id']}",
                f"task_id: {record['task_id']}",
                f"parent_sample_id: {record['parent_sample_id']}",
                f"attack: {record['attack_family']} / {record['attack_name']}",
                f"source_model: {record['source_model']}",
                f"domain: {record['domain']}",
                f"nominal_target_length: {record['nominal_target_length']}",
                f"parent_actual_token_length: {record['parent_actual_token_length']}",
                f"payload_bit_length: {record['payload_bit_length']}",
                f"available_capacity_bits: {record['available_capacity_bits']}",
                f"payload_rate_bpt: {record['payload_rate_bpt']:.8f}",
                f"payload: {record['payload']}",
                f"decoded_payload: {record['decoded_payload']}",
                "text:",
                visible,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_corpus_summary(
    *,
    tasks: list[dict[str, Any]],
    accepted_records: list[dict[str, Any]],
    terminal_failures: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accounting = account_task_inventory(
        tasks=tasks,
        accepted_records=accepted_records,
        terminal_failures=terminal_failures,
    )
    structural = [task for task in tasks if not bool(task["eligible"])]
    eligible_synonym = [
        task
        for task in tasks
        if task["attack_family"] == "token_blackbox" and bool(task["eligible"])
    ]

    counts_by_model = Counter(record["source_model"] for record in accepted_records)
    counts_by_cell = Counter(
        (
            record["attack_name"],
            record["source_model"],
            record["domain"],
            int(record["nominal_target_length"]),
        )
        for record in accepted_records
    )
    exact_rate = (
        sum(bool(record.get("exact_match")) for record in accepted_records)
        / len(accepted_records)
        if accepted_records
        else None
    )

    synonym_capacities = [int(task["available_capacity_bits"]) for task in eligible_synonym]
    summary: dict[str, Any] = {
        **accounting,
        "accepted_counts_by_source_model": dict(sorted(counts_by_model.items())),
        "accepted_counts_by_attack_model_domain_length": {
            " | ".join(map(str, key)): value
            for key, value in sorted(counts_by_cell.items())
        },
        "exact_decode_success_rate_accepted": exact_rate,
        "structural_ineligible_reason_counts": dict(
            sorted(Counter(task["eligibility_reason"] for task in structural).items())
        ),
        "synonym_eligible_covers": len(eligible_synonym),
        "synonym_capacity_distribution_eligible": dict(
            sorted(Counter(synonym_capacities).items())
        ),
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
    }
    if run_metadata:
        summary["last_run"] = run_metadata
    return summary


def is_expected_terminal_attack_failure(task: dict[str, Any], exc: Exception) -> bool:
    """Return True only for frozen attack-search exhaustion, not infrastructure bugs.

    Step 7 must not misclassify OOMs, device-placement problems, model-loading
    problems, protocol drift, or validator bugs as scientific attack failures.
    Only explicit exhaustion emitted by the frozen rank or synonym search is
    terminalized.
    """
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc)
    lowered = message.casefold()
    infrastructure_markers = (
        "out of memory",
        "can't allocate memory",
        "cannot allocate memory",
        "cuda error",
        "cublas",
        "cudnn",
        "device-side assert",
        "meta device",
        "expected all tensors to be on the same device",
    )
    if any(marker in lowered for marker in infrastructure_markers):
        return False

    family = str(task.get("attack_family"))
    if family == "token_whitebox":
        return message.startswith("Rank encoding failed after ")
    if family == "token_blackbox":
        return message.startswith("Synonym full-capacity payload selection failed after ")
    return False
