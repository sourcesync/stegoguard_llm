from __future__ import annotations

import unittest

from stegoguard_lm.robustness.analysis import _contrast_id, _r3_contrast_specs, split_roles


class Step12AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "step12_robustness": {
                "endpoint_round": 3,
                "defender_policies": ["det", "rand"],
                "condition_contrasts": [["fixed", "frozen"], ["migration", "frozen"], ["migration", "fixed"]],
                "defender_contrast": ["rand", "det"],
                "attacker_conditions": ["frozen", "fixed", "migration"],
            }
        }

    def test_endpoint_contrast_universe_has_nine_rows(self) -> None:
        specs = _r3_contrast_specs(self.config)
        self.assertEqual(len(specs), 9)
        self.assertTrue(all(spec["round_id"] == 3 for spec in specs))

    def test_contrast_id_is_stable(self) -> None:
        spec = _r3_contrast_specs(self.config)[0]
        self.assertEqual(_contrast_id(spec), "det::fixed-vs-frozen")

    def test_split_roles(self) -> None:
        rows = [
            {"record_role": "stego_trajectory"},
            {"record_role": "clean_control"},
            {"record_role": "other"},
        ]
        stego, clean = split_roles(rows)
        self.assertEqual(len(stego), 1)
        self.assertEqual(len(clean), 1)


if __name__ == "__main__":
    unittest.main()
