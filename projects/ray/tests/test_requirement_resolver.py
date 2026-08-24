"""Behavior tests for selecting upstream Ray test requirements."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESOLVER = _PROJECT_ROOT / "scripts" / "resolve_test_requirements.py"


class TestRequirementResolver(unittest.TestCase):
    def test_selects_only_requested_target_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = root / "test-requirements.txt"
            common.write_text(
                "pytest==7.4.4\n"
                "moto[s3,server]==5.1.18\n",
                encoding="utf-8",
            )
            train = root / "train-test-requirements.txt"
            train.write_text("boto3==1.29.7\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(_RESOLVER),
                    "--requirement",
                    str(common),
                    "pytest",
                    "--requirement",
                    str(train),
                    "boto3",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.splitlines(),
                ["pytest==7.4.4", "boto3==1.29.7"],
            )

    def test_missing_target_declaration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "requirements.txt"
            requirements.write_text("pytest==7.4.4\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(_RESOLVER),
                    "--requirement",
                    str(requirements),
                    "boto3",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("does not declare boto3", completed.stderr)


if __name__ == "__main__":
    unittest.main()
