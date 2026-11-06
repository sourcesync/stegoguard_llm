from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stegoguard_lm.robustness.io import hash_files, verify_hash_mapping


class Step12ProvenanceTests(unittest.TestCase):
    def test_hash_verification_detects_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.txt"
            path.write_text("a", encoding="utf-8")
            mapping = hash_files([path])
            self.assertEqual(verify_hash_mapping(mapping), [])
            path.write_text("b", encoding="utf-8")
            self.assertTrue(verify_hash_mapping(mapping))


if __name__ == "__main__":
    unittest.main()
