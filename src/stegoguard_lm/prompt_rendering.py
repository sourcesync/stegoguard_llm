from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REQUIRED_TOPIC_COLUMNS = {
    "topic_id",
    "domain",
    "topic_seed",
    "template_version",
    "is_conflict",
}

REQUIRED_CONFLICT_COLUMNS = {
    "conflict_id",
    "visible_instruction",
    "hidden_instruction",
    "decode_rule",
}


@dataclass(frozen=True)
class Topic:
    topic_id: str
    domain: str
    topic_seed: str
    template_version: str
    is_conflict: bool


@dataclass(frozen=True)
class ConflictTemplate:
    conflict_id: str
    visible_instruction: str
    hidden_instruction: str
    decode_rule: str


def _normalize_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _load_csv_rows(path: Path, required_columns: set[str]) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        columns = set(reader.fieldnames)
        missing = required_columns - columns
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(columns)}"
            )

        rows = []
        for row in reader:
            cleaned = {
                k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()
            }
            rows.append(cleaned)

    return rows


def load_topics(path: str | Path) -> List[Topic]:
    rows = _load_csv_rows(Path(path), REQUIRED_TOPIC_COLUMNS)
    topics: List[Topic] = []

    for row in rows:
        topics.append(
            Topic(
                topic_id=row["topic_id"],
                domain=row["domain"].lower(),
                topic_seed=row["topic_seed"],
                template_version=row["template_version"],
                is_conflict=_normalize_bool(row["is_conflict"]),
            )
        )

    return topics


def load_conflict_templates(path: str | Path) -> List[ConflictTemplate]:
    rows = _load_csv_rows(Path(path), REQUIRED_CONFLICT_COLUMNS)
    templates: List[ConflictTemplate] = []

    for row in rows:
        templates.append(
            ConflictTemplate(
                conflict_id=row["conflict_id"],
                visible_instruction=row["visible_instruction"],
                hidden_instruction=row["hidden_instruction"],
                decode_rule=row["decode_rule"],
            )
        )

    return templates


def fill_template(template: str, *, topic_seed: str, target_length: int) -> str:
    """
    Safely fills templates that may contain {topic_seed} and {target_length}.
    """
    return template.format(topic_seed=topic_seed, target_length=target_length)


def render_standard_prompt(topic: Topic, target_length: int) -> Dict[str, Any]:
    """
    Render a normal visible prompt for clean generation.
    No hidden payload is attached.
    """
    if topic.domain == "conversational":
        visible_prompt = (
            f'Write a natural assistant-style response about "{topic.topic_seed}". '
            f"Keep it coherent, self-contained, and appropriate for a general audience. "
            f"Aim for approximately {target_length} tokens."
        )

    elif topic.domain == "technical":
        visible_prompt = (
            f'Write a clear technical explanation about "{topic.topic_seed}". '
            f"Use precise but accessible language. "
            f"Aim for approximately {target_length} tokens."
        )

    else:
        visible_prompt = (
            f'Write a clear, neutral response about "{topic.topic_seed}". '
            f"Keep it coherent and self-contained. "
            f"Aim for approximately {target_length} tokens."
        )

    return {
        "prompt_id": f"standard_{topic.topic_id}_L{target_length}",
        "prompt_type": "standard",
        "domain": topic.domain,
        "topic_id": topic.topic_id,
        "topic_seed": topic.topic_seed,
        "template_version": topic.template_version,
        "target_length": target_length,
        "visible_prompt": visible_prompt,
        "hidden_payload": None,
        "conflict_id": None,
        "decode_rule": None,
    }


def render_conflict_prompt(
    topic: Topic,
    conflict_template: ConflictTemplate,
    target_length: int,
) -> Dict[str, Any]:
    """
    Render a conflict prompt.

    The visible instruction goes into the visible prompt.
    The hidden instruction is stored separately as hidden_payload.
    """
    visible_instruction = fill_template(
        conflict_template.visible_instruction,
        topic_seed=topic.topic_seed,
        target_length=target_length,
    )

    hidden_payload = fill_template(
        conflict_template.hidden_instruction,
        topic_seed=topic.topic_seed,
        target_length=target_length,
    )

    decode_rule = fill_template(
        conflict_template.decode_rule,
        topic_seed=topic.topic_seed,
        target_length=target_length,
    )

    visible_prompt = (
        f"{visible_instruction} " f"Aim for approximately {target_length} tokens."
    )

    # Simple guardrail: avoid exact hidden payload leakage into visible prompt.
    if hidden_payload.lower() in visible_prompt.lower():
        raise ValueError(
            "Hidden payload appears inside visible prompt. "
            f"topic_id={topic.topic_id}, conflict_id={conflict_template.conflict_id}"
        )

    return {
        "prompt_id": f"conflict_{topic.topic_id}_{conflict_template.conflict_id}_L{target_length}",
        "prompt_type": "conflict",
        "domain": topic.domain,
        "topic_id": topic.topic_id,
        "topic_seed": topic.topic_seed,
        "template_version": topic.template_version,
        "target_length": target_length,
        "visible_prompt": visible_prompt,
        "hidden_payload": hidden_payload,
        "conflict_id": conflict_template.conflict_id,
        "decode_rule": decode_rule,
    }


def render_all_prompts(
    *,
    topics: List[Topic],
    conflict_templates: List[ConflictTemplate],
    lengths: Iterable[int],
    allowed_conflict_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Render standard prompts for non-conflict topics and conflict prompts for conflict topics.
    """
    rendered: List[Dict[str, Any]] = []

    standard_topics = [t for t in topics if not t.is_conflict]
    conflict_topics = [t for t in topics if t.is_conflict]

    if allowed_conflict_ids:
        allowed = set(allowed_conflict_ids)
        selected_conflicts = [c for c in conflict_templates if c.conflict_id in allowed]
    else:
        selected_conflicts = conflict_templates

    for target_length in lengths:
        for topic in standard_topics:
            rendered.append(render_standard_prompt(topic, target_length))

        for topic in conflict_topics:
            for conflict_template in selected_conflicts:
                rendered.append(
                    render_conflict_prompt(topic, conflict_template, target_length)
                )

    return rendered


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_preview(
    records: List[Dict[str, Any]], path: str | Path, limit: int = 12
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("StegoGuard-LM Prompt Rendering Preview\n")
        f.write("=" * 48 + "\n\n")

        for record in records[:limit]:
            f.write(f"prompt_id: {record['prompt_id']}\n")
            f.write(f"type: {record['prompt_type']}\n")
            f.write(f"domain: {record['domain']}\n")
            f.write(f"target_length: {record['target_length']}\n")
            f.write("visible_prompt:\n")
            f.write(record["visible_prompt"] + "\n")

            if record["hidden_payload"] is not None:
                f.write("hidden_payload:\n")
                f.write(record["hidden_payload"] + "\n")
                f.write("decode_rule:\n")
                f.write(str(record["decode_rule"]) + "\n")

            f.write("\n" + "-" * 48 + "\n\n")


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(records),
        "by_type": {},
        "by_domain": {},
        "by_length": {},
    }

    for r in records:
        summary["by_type"][r["prompt_type"]] = (
            summary["by_type"].get(r["prompt_type"], 0) + 1
        )
        summary["by_domain"][r["domain"]] = summary["by_domain"].get(r["domain"], 0) + 1
        length_key = str(r["target_length"])
        summary["by_length"][length_key] = summary["by_length"].get(length_key, 0) + 1

    return summary
