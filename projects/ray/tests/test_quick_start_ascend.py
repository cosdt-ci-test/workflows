"""End-to-end test for projects/ray/docs/Quick-start-Ascend.md."""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _e2e_enabled() -> bool:
    return os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1200
    USER_AGENT = "cosdt-ci-test/ray-quick-start"
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

    @classmethod
    def prepare_environment(cls) -> None:
        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(f"CANN environment script not found: {cls._CANN_SET_ENV}")
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
            os.environ.setdefault(key, value)

        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        subprocess.run(
            [
                "python",
                "-c",
                "import torch, torch_npu; "
                "assert torch.npu.is_available(); "
                "assert torch.npu.device_count() >= 2; "
                "print(torch.__version__, torch_npu.__version__)",
            ],
            check=True,
        )

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
