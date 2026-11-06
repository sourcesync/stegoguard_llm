from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_leave_one_topic_out_splits(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    topic_ids = sorted({str(row["topic_id"]) for row in records})
    folds: list[dict[str, Any]] = []
    all_ids = {str(row["sample_id"]) for row in records}

    for fold_index, test_topic in enumerate(topic_ids):
        test_ids = sorted(
            str(row["sample_id"])
            for row in records
            if str(row["topic_id"]) == test_topic
        )
        train_ids = sorted(all_ids - set(test_ids))
        train_topics = [topic for topic in topic_ids if topic != test_topic]
        folds.append(
            {
                "fold_id": f"topic_{fold_index:02d}_{test_topic}",
                "test_topic_id": test_topic,
                "train_topic_ids": train_topics,
                "train_sample_ids": train_ids,
                "test_sample_ids": test_ids,
            }
        )

    result = {
        "protocol_version": "step8_static_v1",
        "split_strategy": "leave_one_topic_out",
        "topic_count": len(topic_ids),
        "topic_ids": topic_ids,
        "fold_count": len(folds),
        "folds": folds,
    }
    validate_splits(records, result)
    return result


def validate_splits(
    records: list[dict[str, Any]],
    split_definition: dict[str, Any],
) -> None:
    by_id = {str(row["sample_id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Detection records contain duplicate sample IDs.")
    folds = split_definition.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Split definition contains no folds.")

    test_occurrences: defaultdict[str, int] = defaultdict(int)
    parent_test_fold: dict[str, str] = {}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        train_ids = set(map(str, fold.get("train_sample_ids", [])))
        test_ids = set(map(str, fold.get("test_sample_ids", [])))
        if train_ids & test_ids:
            raise ValueError(f"Train/test sample overlap in {fold_id}.")
        if train_ids | test_ids != set(by_id):
            raise ValueError(f"Fold {fold_id} does not partition the full dataset.")

        train_topics = {str(by_id[sample_id]["topic_id"]) for sample_id in train_ids}
        test_topics = {str(by_id[sample_id]["topic_id"]) for sample_id in test_ids}
        if train_topics & test_topics:
            raise ValueError(f"Topic leakage detected in {fold_id}.")
        if test_topics != {str(fold["test_topic_id"])}:
            raise ValueError(f"Unexpected test topics in {fold_id}: {sorted(test_topics)}")

        for sample_id in test_ids:
            test_occurrences[sample_id] += 1
            parent_id = str(by_id[sample_id]["clean_parent_id"])
            previous = parent_test_fold.get(parent_id)
            if previous is not None and previous != fold_id:
                raise ValueError(
                    f"Parent lineage {parent_id} appears in multiple test folds."
                )
            parent_test_fold[parent_id] = fold_id

    missing_or_repeated = {
        sample_id: count
        for sample_id, count in test_occurrences.items()
        if count != 1
    }
    missing = sorted(set(by_id) - set(test_occurrences))
    if missing or missing_or_repeated:
        raise ValueError(
            f"Every sample must appear in exactly one test fold. "
            f"missing={missing[:5]}, repeated={list(missing_or_repeated.items())[:5]}"
        )
