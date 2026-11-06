import types
import unittest

import torch

from stegoguard_lm.attacks.rank_channel import (
    decode_rank_payload,
    encode_rank_payload,
)


class CharacterTokenizer:
    def __init__(self):
        self.id_to_text = {
            0: "",
            1: "",
            2: "",
            3: "a",
            4: "b",
            5: "c",
            6: "d",
            7: ".",
            8: "!",
            9: "?",
            10: "x",
            11: "y",
        }
        self.text_to_id = {text: token_id for token_id, text in self.id_to_text.items() if text}
        self.all_special_ids = [0, 1, 2]
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.chat_template = None

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.text_to_id[character] for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        output = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            output.append(self.id_to_text[token_id])
        return "".join(output)


class DeterministicModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight=self.weight)

    def parameters(self):
        yield self.weight

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, use_cache, return_dict, kwargs
        batch, length = input_ids.shape
        vocab_size = 12
        logits = torch.full((batch, length, vocab_size), -100.0)
        # Stable descending order: a, b, c, d, x, y, ., !, ?.
        scores = {
            3: 10.0,
            4: 9.0,
            5: 8.0,
            6: 7.0,
            10: 6.0,
            11: 5.0,
            7: 4.0,
            8: 3.0,
            9: 2.0,
        }
        for token_id, score in scores.items():
            logits[:, :, token_id] = score
        prior = int(past_key_values or 0)
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=prior + length,
        )


class RankChannelIntegrationTests(unittest.TestCase):
    def test_fake_model_round_trip(self):
        tokenizer = CharacterTokenizer()
        model = DeterministicModel()
        payload = "0101"

        result = encode_rank_payload(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            payload=payload,
            required_token_length=4,
            attack_seed=123,
            top_k=2,
            max_encoding_attempts=3,
            final_candidate_search_k=12,
            require_complete_ending=True,
            require_tokenizer_roundtrip=True,
        )

        self.assertEqual(len(tokenizer.encode(result.stego_text)), 4)
        self.assertIn(result.stego_text[-1], ".!?")
        self.assertEqual(result.decoded_payload, payload)
        self.assertTrue(result.exact_match)

        decoded = decode_rank_payload(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            stego_text=result.stego_text,
            attack_seed=123,
            encoding_nonce=result.channel_parameters["encoding_nonce"],
            top_k=2,
            final_candidate_search_k=12,
            require_complete_ending=True,
        )
        self.assertTrue(decoded.success)
        self.assertEqual(decoded.payload, payload)


if __name__ == "__main__":
    unittest.main()
