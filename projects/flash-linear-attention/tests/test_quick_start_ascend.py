"""End-to-end test for the flash-linear-attention Ascend Quick Start."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _path in (_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _e2e_enabled() -> bool:
    return os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "cosdt-ci-test/flash-linear-attention-quick-start"
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = "/usr/local/sbin:/usr/local/bin"
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{path_dirs}:{current_path}"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(f"CANN environment script is missing: {cls._CANN_SET_ENV}")

        merged = subprocess.run(
            ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1; env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key] = value
        print("setup: sourced CANN environment")

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires an Ascend NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == "__main__":
    unittest.main()
