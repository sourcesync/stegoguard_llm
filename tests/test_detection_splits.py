from __future__ import annotations

import unittest

from stegoguard_lm.detection.splits import build_leave_one_topic_out_splits, validate_splits


class DetectionSplitTests(unittest.TestCase):
    def test_topic_disjoint_and_parent_grouped(self):
        records = []
        for topic in ("conv_001", "conv_002", "tech_001"):
            parent = f"clean_{topic}"
            for sample, y in ((parent, 0), (f"stego_{topic}", 1)):
                records.append(
                    {
                        "sample_id": sample,
                        "clean_parent_id": parent,
                        "topic_id": topic,
                        "y": y,
                    }
                )
        splits = build_leave_one_topic_out_splits(records)
        self.assertEqual(splits["fold_count"], 3)
        validate_splits(records, splits)
        for fold in splits["folds"]:
            test = set(fold["test_sample_ids"])
            for topic in ("conv_001", "conv_002", "tech_001"):
                parent = f"clean_{topic}"
                child = f"stego_{topic}"
                self.assertEqual(parent in test, child in test)


if __name__ == "__main__":
    unittest.main()
