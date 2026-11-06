import unittest

from stegoguard_lm.adaptive.io import round_paths


class AdaptiveProtocolTests(unittest.TestCase):
    def test_round_paths_are_round_scoped(self):
        config = {"paths": {"adaptive_output_dir": "outputs/pilot_v0/adaptive/step10"}}
        paths = round_paths(config, 2)
        self.assertEqual(str(paths["tasks"]).replace("\\", "/"), "outputs/pilot_v0/adaptive/step10/round_2/round_tasks.jsonl")
        self.assertEqual(str(paths["feedback"]).replace("\\", "/"), "outputs/pilot_v0/adaptive/step10/round_2/feedback_results.jsonl")


if __name__ == "__main__":
    unittest.main()
