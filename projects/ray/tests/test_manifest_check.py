"""Tests for Ray's project-local explicit-path manifest checker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHECKER = _PROJECT_ROOT / "scripts" / "check_manifest.py"
_MANIFEST = _PROJECT_ROOT / "examples_manifest.yaml"
_PATHS = (
    "python/ray/tests/accelerators/test_npu.py",
    "python/ray/train/tests/test_torch_device_manager.py",
)


class TestRayManifestCheck(unittest.TestCase):
    def _run(self, target_root: Path, result_path: Path, output_path: Path):
        self.assertTrue(_CHECKER.is_file(), _CHECKER)
        env = os.environ.copy()
        env["GITHUB_OUTPUT"] = str(output_path)
        return subprocess.run(
            [
                sys.executable,
                str(_CHECKER),
                "--target-root",
                str(target_root),
                "--manifest",
                str(_MANIFEST),
                "--result-json",
                str(result_path),
                "--target-repo",
                "ray-project/ray",
                "--target-ref",
                "master",
                "--trigger",
                "workflow_dispatch",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_existing_explicit_paths_produce_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in _PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pass\n", encoding="utf-8")
            result_path = root / "result.json"
            output_path = root / "github-output"

            completed = self._run(root, result_path, output_path)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["new_paths"], [])
            self.assertEqual(result["stale_paths"], [])
            output = output_path.read_text(encoding="utf-8")
            self.assertIn("supported_matrix<<EOF", output)
            self.assertIn("--device=/dev/davinci0", output)
            self.assertIn("has_supported=true", output)

    def test_missing_supported_path_fails_after_writing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / _PATHS[0]
            existing.parent.mkdir(parents=True)
            existing.write_text("pass\n", encoding="utf-8")
            result_path = root / "result.json"
            output_path = root / "github-output"

            completed = self._run(root, result_path, output_path)

            self.assertNotEqual(completed.returncode, 0)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["stale_paths"], [_PATHS[1]])
            self.assertIn("supported upstream Ray path missing", completed.stderr)


if __name__ == "__main__":
    unittest.main()
