"""Behavior tests for verifying the Ray wheel/source test overlay."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VERIFIER = _PROJECT_ROOT / "scripts" / "verify_ray_test_support.py"


class TestRayTestSupport(unittest.TestCase):
    def _make_layout(self, root: Path, installed_version: str) -> tuple[Path, Path]:
        target = root / "target"
        target_ray = target / "python" / "ray"
        target_tests = target_ray / "tests"
        target_tests.mkdir(parents=True)
        (target_ray / "_version.py").write_text(
            'version = "2.58.0"\n', encoding="utf-8"
        )
        (target_tests / "__init__.py").write_text("", encoding="utf-8")
        (target_tests / "conftest.py").write_text("", encoding="utf-8")

        site = root / "site"
        installed_ray = site / "ray"
        installed_ray.mkdir(parents=True)
        (installed_ray / "__init__.py").write_text(
            f'__version__ = "{installed_version}"\n', encoding="utf-8"
        )
        return target, site

    def _run(self, target: Path, site: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site)
        return subprocess.run(
            [sys.executable, str(_VERIFIER), str(target)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_accepts_matching_version_and_target_test_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target, site = self._make_layout(Path(tmp), "2.58.0")
            (site / "ray" / "tests").symlink_to(
                target / "python" / "ray" / "tests",
                target_is_directory=True,
            )

            completed = self._run(target, site)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Ray test support verified", completed.stdout)

    def test_rejects_test_package_outside_target_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, site = self._make_layout(root, "2.58.0")
            installed_tests = site / "ray" / "tests"
            installed_tests.mkdir()
            (installed_tests / "__init__.py").write_text("", encoding="utf-8")
            (installed_tests / "conftest.py").write_text("", encoding="utf-8")

            completed = self._run(target, site)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("was not linked from target checkout", completed.stderr)

    def test_rejects_installed_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target, site = self._make_layout(Path(tmp), "2.59.0")
            (site / "ray" / "tests").symlink_to(
                target / "python" / "ray" / "tests",
                target_is_directory=True,
            )

            completed = self._run(target, site)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Ray version mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
