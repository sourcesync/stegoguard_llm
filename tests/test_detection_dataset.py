from __future__ import annotations

import unittest

from stegoguard_lm.detection.datasets import build_detection_dataset


class DetectionDatasetTests(unittest.TestCase):
    def _clean(self, sample_id: str, model: str, domain: str, length: int, topic: str):
        pair = sample_id.removeprefix("clean_")
        return {
            "sample_id": sample_id,
            "pair_id": pair,
            "lineage_id": pair,
            "label": "clean",
            "domain": domain,
            "topic_id": topic,
            "topic_seed": topic,
            "target_length": length,
            "actual_token_length": length,
            "source_model": model,
            "text": "A normal complete response.",
            "prompt": "Give a normal response.",
            "prompt_id": f"p_{topic}_{length}",
            "manifest_version": "pilot_v0",
        }

    def _stego(self, clean, attack):
        return {
            "sample_id": f"stego_{attack}_{clean['pair_id']}",
            "parent_sample_id": clean["sample_id"],
            "attack_family": "x",
            "attack_name": attack,
            "validation_status": "accepted",
            "prompt": clean["prompt"],
            "text": clean["text"] + attack,
            "topic_id": clean["topic_id"],
            "payload_bit_length": 1,
            "payload_rate_bpt": 1 / clean["actual_token_length"],
            "manifest_version": "pilot_v0",
        }

    def test_builds_common_and_balanced_membership(self):
        cleans = []
        for model in ("qwen", "mistral"):
            for domain in ("conversational", "technical"):
                for length in (64, 256):
                    for i in range(2):
                        cleans.append(
                            self._clean(
                                f"clean_{domain[:4]}_{length}_{model}_{i}",
                                model,
                                domain,
                                length,
                                f"{domain[:4]}_{i}",
                            )
                        )
        stego = []
        for clean in cleans:
            for attack in ("rank_encoding", "synonym_slot_codebook", "unicode_hidden_channel"):
                stego.append(self._stego(clean, attack))
        records, manifest = build_detection_dataset(
            clean_records=cleans,
            stego_records=stego,
            seed=42,
            balanced_per_cell=1,
        )
        self.assertEqual(len(records), len(cleans) * 4)
        self.assertEqual(manifest["common_parent_count_all_three"], len(cleans))
        self.assertEqual(manifest["balanced_common_parent_count"], 8)
        self.assertEqual(sum(r["is_balanced_common"] and r["label"] == "clean" for r in records), 8)

    def test_rejects_missing_parent(self):
        clean = self._clean("clean_a", "qwen", "conversational", 64, "conv_001")
        bad = self._stego(clean, "rank_encoding")
        bad["parent_sample_id"] = "clean_missing"
        with self.assertRaises(ValueError):
            build_detection_dataset(
                clean_records=[clean],
                stego_records=[bad],
                seed=42,
                balanced_per_cell=0,
            )


if __name__ == "__main__":
    unittest.main()
