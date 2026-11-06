import unittest

from stegoguard_lm.attacks.rank_channel import bit_for_rank, keyed_rank_offset


class RankPartitionTests(unittest.TestCase):
    def test_rank_partition(self):
        self.assertEqual([bit_for_rank(rank, 0) for rank in range(6)], list("010101"))
        self.assertEqual([bit_for_rank(rank, 1) for rank in range(6)], list("101010"))

    def test_offset_is_deterministic_and_context_sensitive(self):
        kwargs = {
            "attack_seed": 42,
            "encoding_nonce": 0,
            "token_position": 3,
            "prompt_fingerprint": "abc",
            "prefix_token_ids": [5, 6, 7],
        }
        self.assertEqual(keyed_rank_offset(**kwargs), keyed_rank_offset(**kwargs))
        changed = dict(kwargs)
        changed["prefix_token_ids"] = [5, 6, 8]
        values = {
            keyed_rank_offset(**kwargs),
            keyed_rank_offset(**changed),
        }
        self.assertTrue(values.issubset({0, 1}))

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            bit_for_rank(-1, 0)
        with self.assertRaises(ValueError):
            bit_for_rank(0, 2)


if __name__ == "__main__":
    unittest.main()
