from __future__ import annotations

import unittest

import torch

from stegoguard_lm.mitigation.rewriting import (
    build_rewrite_prompt,
    compute_max_new_tokens,
    rewrite_response,
)


class FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long)}

    def encode(self, text, add_special_tokens=False):
        return [5] * max(1, len(text.split()))

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "Rewritten answer."


class FakeModel:
    device = torch.device("cpu")

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        extra = torch.tensor([[9, 10]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class MitigationRewritingTests(unittest.TestCase):
    def test_prompt_contains_visible_request_and_response_without_payload_language(self) -> None:
        prompt = build_rewrite_prompt(visible_prompt="Explain indexing.", response="Indexes help queries.")
        self.assertIn("Explain indexing.", prompt)
        self.assertIn("Indexes help queries.", prompt)
        self.assertIn("Return only the rewritten", prompt)

    def test_max_new_tokens_respects_hard_cap(self) -> None:
        self.assertEqual(
            compute_max_new_tokens(
                input_response_tokens=300,
                max_length_ratio=1.3,
                max_extra_tokens=48,
                hard_cap=420,
            ),
            420,
        )

    def test_max_new_tokens_uses_ratio_and_extra(self) -> None:
        self.assertEqual(
            compute_max_new_tokens(
                input_response_tokens=100,
                max_length_ratio=1.2,
                max_extra_tokens=10,
                hard_cap=500,
            ),
            130,
        )

    def test_fake_deterministic_rewrite_runs_end_to_end(self) -> None:
        result = rewrite_response(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            visible_prompt="Explain indexing.",
            response="Indexes help queries.",
            mode="deterministic_rewrite",
            seed=None,
            max_length_ratio=1.3,
            max_extra_tokens=48,
            hard_cap=420,
            randomized_temperature=0.8,
            randomized_top_p=0.9,
        )
        self.assertEqual(result.text, "Rewritten answer.")
        self.assertFalse(result.do_sample)
        self.assertIsNone(result.seed)

    def test_fake_randomized_rewrite_uses_frozen_seed_and_sampling(self) -> None:
        result = rewrite_response(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            visible_prompt="Explain indexing.",
            response="Indexes help queries.",
            mode="randomized_rewrite",
            seed=9101,
            max_length_ratio=1.3,
            max_extra_tokens=48,
            hard_cap=420,
            randomized_temperature=0.8,
            randomized_top_p=0.9,
        )
        self.assertTrue(result.do_sample)
        self.assertEqual(result.seed, 9101)
        self.assertAlmostEqual(result.temperature, 0.8)
        self.assertAlmostEqual(result.top_p, 0.9)


if __name__ == "__main__":
    unittest.main()
