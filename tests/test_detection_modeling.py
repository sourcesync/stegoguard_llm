from __future__ import annotations

import unittest

from stegoguard_lm.detection.modeling import FeatureStore, run_topic_disjoint_cv
from stegoguard_lm.detection.splits import build_leave_one_topic_out_splits


class DetectionModelingTests(unittest.TestCase):
    def test_topic_disjoint_cv_produces_one_prediction_per_row(self):
        dataset = []
        eval_a = {}
        eval_b = {}
        semantic = {}
        unicode = {}
        for topic_index, topic in enumerate(("conv_001", "conv_002", "tech_001", "tech_002")):
            for parent_index in range(2):
                parent = f"clean_{topic}_{parent_index}"
                for label, sample_id, shift in (
                    ("clean", parent, 0.0),
                    ("stego", f"stego_{topic}_{parent_index}", 1.0),
                ):
                    y = int(label == "stego")
                    dataset.append(
                        {
                            "sample_id": sample_id,
                            "clean_parent_id": parent,
                            "pair_id": parent,
                            "topic_id": topic,
                            "source_model": "qwen",
                            "domain": "technical" if topic.startswith("tech") else "conversational",
                            "nominal_target_length": 64,
                            "attack_name": "rank_encoding" if y else "clean",
                            "label": label,
                            "y": y,
                            "payload_bit_length": 64 if y else 0,
                            "payload_rate_bpt": 1.0 if y else 0.0,
                        }
                    )
                    eval_a[sample_id] = {
                        "sample_id": sample_id,
                        "mean_logp": -2.0 - shift,
                        "ppl": 8.0 + 8.0 * shift,
                        "logp_var": 0.2 + shift,
                        "sentence_logp_var": 0.1 + shift,
                        "token_logprobs": [-2.0 - shift] * 5,
                    }
                    eval_b[sample_id] = {
                        "sample_id": sample_id,
                        "mean_logp": -2.1 - 0.8 * shift,
                        "ppl": 8.5 + 7.0 * shift,
                        "logp_var": 0.25 + shift,
                        "sentence_logp_var": 0.12 + shift,
                        "token_logprobs": [-2.1 - 0.8 * shift] * 5,
                    }
                    semantic[sample_id] = {
                        "sample_id": sample_id,
                        "paraphrase_embedding_similarity": 0.95 - 0.2 * shift,
                        "visible_instruction_retention": 0.95 - 0.3 * shift,
                    }
                    unicode[sample_id] = {"sample_id": sample_id, "has_suspicious_codepoint": 0}
        splits = build_leave_one_topic_out_splits(dataset)
        store = FeatureStore(
            dataset_by_id={r["sample_id"]: r for r in dataset},
            evaluator_a_by_id=eval_a,
            evaluator_b_by_id=eval_b,
            semantic_by_id=semantic,
            unicode_by_id=unicode,
        )
        predictions = run_topic_disjoint_cv(
            store=store,
            split_definition=splits,
            population_ids={r["sample_id"] for r in dataset},
            variant="full_fusion",
            c=1.0,
            max_iter=1000,
            seed=42,
        )
        self.assertEqual(len(predictions), len(dataset))
        self.assertEqual(len({r["sample_id"] for r in predictions}), len(dataset))
        self.assertTrue(all(0.0 <= r["score"] <= 1.0 for r in predictions))


if __name__ == "__main__":
    unittest.main()
