from __future__ import annotations

from typing import Any

from stegoguard_lm.detection.evaluator_scoring import score_response_distribution
from stegoguard_lm.detection.semantic import (
    cosine_similarity,
    mean_pool_embedding,
    visible_instruction_retention,
)


def evaluate_utility_after_mitigation(
    *,
    model: Any,
    tokenizer: Any,
    visible_prompt: str,
    original_text: str,
    mitigated_text: str,
    original_semantic_row: dict[str, Any],
    original_distribution_row: dict[str, Any],
    original_embedding: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Measure post-mitigation utility with the frozen evaluator-A model.

    The original Step 8 feature rows provide the pre-mitigation baselines.  Only
    prompt and text are passed to the evaluator model.
    """

    if original_embedding is None:
        original_embedding = mean_pool_embedding(
            model=model,
            tokenizer=tokenizer,
            text=original_text,
        )
    mitigated_embedding = mean_pool_embedding(
        model=model,
        tokenizer=tokenizer,
        text=mitigated_text,
    )
    semantic_similarity = cosine_similarity(original_embedding, mitigated_embedding)
    instruction_after = visible_instruction_retention(
        model=model,
        tokenizer=tokenizer,
        visible_prompt=visible_prompt,
        response=mitigated_text,
    )
    distribution_after = score_response_distribution(
        model=model,
        tokenizer=tokenizer,
        prompt=visible_prompt,
        response=mitigated_text,
    )
    instruction_before = float(original_semantic_row["visible_instruction_retention"])
    mean_logp_before = float(original_distribution_row["mean_logp"])
    ppl_before = float(original_distribution_row["ppl"])
    tokens_before = int(original_distribution_row["response_token_count"])
    tokens_after = int(distribution_after["response_token_count"])
    result = {
        "semantic_similarity_original_to_mitigated": float(semantic_similarity),
        "visible_instruction_retention_before": instruction_before,
        "visible_instruction_retention_after": float(instruction_after),
        "visible_instruction_retention_delta": float(instruction_after - instruction_before),
        "mean_logp_before": mean_logp_before,
        "mean_logp_after": float(distribution_after["mean_logp"]),
        "mean_logp_delta": float(distribution_after["mean_logp"] - mean_logp_before),
        "ppl_before": ppl_before,
        "ppl_after": float(distribution_after["ppl"]),
        "ppl_delta": float(distribution_after["ppl"] - ppl_before),
        "response_token_count_before": tokens_before,
        "response_token_count_after": tokens_after,
        "absolute_token_change": tokens_after - tokens_before,
        "length_ratio": tokens_after / tokens_before if tokens_before else None,
        "logp_var_after": float(distribution_after["logp_var"]),
        "sentence_logp_var_after": float(distribution_after["sentence_logp_var"]),
    }
    return result, original_embedding
