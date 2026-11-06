from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]*(?=\s|$)')


@dataclass
class EvaluatorBundle:
    model: Any
    tokenizer: Any
    model_revision: str | None
    tokenizer_revision: str | None


def load_evaluator(model_id: str) -> EvaluatorBundle:
    """Load a causal-LM evaluator lazily so model-free tests need no transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return EvaluatorBundle(
        model=model,
        tokenizer=tokenizer,
        model_revision=getattr(model.config, "_commit_hash", None),
        tokenizer_revision=tokenizer.init_kwargs.get("_commit_hash"),
    )


def encode_visible_prompt(tokenizer: Any, prompt: str):
    import torch

    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return ids
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def sentence_logp_variance(
    text: str,
    token_logprobs: Iterable[float],
    token_offsets: Iterable[tuple[int, int]],
) -> float:
    logps = list(map(float, token_logprobs))
    offsets = list(token_offsets)
    if len(logps) != len(offsets):
        raise ValueError("token_logprobs and token_offsets must have equal length")
    spans = sentence_spans(text)
    sentence_values: list[list[float]] = [[] for _ in spans]
    for logp, (start, end) in zip(logps, offsets):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        for index, (span_start, span_end) in enumerate(spans):
            if span_start <= midpoint <= span_end:
                sentence_values[index].append(logp)
                break
    sentence_means = [float(np.mean(values)) for values in sentence_values if values]
    if len(sentence_means) < 2:
        return 0.0
    return float(np.var(sentence_means, ddof=0))


def summarize_token_logprobs(
    *,
    text: str,
    token_logprobs: list[float],
    token_offsets: list[tuple[int, int]],
) -> dict[str, Any]:
    if not token_logprobs:
        raise ValueError("Cannot summarize an empty response token sequence.")
    values = np.asarray(token_logprobs, dtype=float)
    mean_logp = float(values.mean())
    return {
        "response_token_count": int(values.size),
        "mean_logp": mean_logp,
        "ppl": float(math.exp(min(50.0, -mean_logp))),
        "logp_var": float(values.var(ddof=0)),
        "sentence_logp_var": sentence_logp_variance(
            text,
            token_logprobs,
            token_offsets,
        ),
        "token_logprobs": [float(value) for value in token_logprobs],
    }


def score_response_distribution(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    import torch

    prompt_ids = encode_visible_prompt(tokenizer, prompt)
    response_encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids_list = list(map(int, response_encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in response_encoded["offset_mapping"]]
    if not response_ids_list:
        raise ValueError("Evaluator tokenizer produced no response tokens.")

    response_ids = torch.tensor([response_ids_list], dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids.cpu(), response_ids], dim=1)
    input_device = getattr(model, "device", None)
    if input_device is None:
        input_device = next(model.parameters()).device
    full_ids = full_ids.to(input_device)
    attention_mask = torch.ones_like(full_ids)

    prompt_len = int(prompt_ids.shape[1])
    response_len = len(response_ids_list)
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[0, prompt_len - 1 : prompt_len + response_len - 1, :]
        targets = full_ids[0, prompt_len : prompt_len + response_len]
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        normalizers = torch.logsumexp(logits.float(), dim=-1)
        logprobs = (target_logits.float() - normalizers).detach().cpu().tolist()

    return summarize_token_logprobs(
        text=response,
        token_logprobs=[float(value) for value in logprobs],
        token_offsets=offsets,
    )
