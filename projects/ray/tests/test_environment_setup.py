"""Behavior tests for Ray Quick Start environment bootstrapping."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_quick_start_ascend as quick_start


class TestQuickStartEnvironmentSetup(unittest.TestCase):
    def test_missing_torch_stack_is_installed_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cann_env = root / "set_env.sh"
            cann_env.write_text("export RAY_CANN_TEST=ready\n", encoding="utf-8")
            log_path = root / "python.log"
            installed = root / "torch-installed"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
                "if [[ \"$1\" == \"-m\" && \"$2\" == \"pip\" ]]; then\n"
                "  touch \"$FAKE_TORCH_INSTALLED\"\n"
                "  exit 0\n"
                "fi\n"
                "if [[ ! -f \"$FAKE_TORCH_INSTALLED\" ]]; then exit 1; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = {
                "PATH": f"{root}:{os.environ['PATH']}",
                "FAKE_PYTHON_LOG": str(log_path),
                "FAKE_TORCH_INSTALLED": str(installed),
                "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            }
            with (
                patch.object(
                    quick_start.TestQuickStartAscend,
                    "_CANN_SET_ENV",
                    str(cann_env),
                ),
                patch.dict(os.environ, env, clear=False),
            ):
                try:
                    quick_start.TestQuickStartAscend.prepare_environment()
                except subprocess.CalledProcessError as exc:
                    self.fail(
                        "prepare_environment validated torch before installing "
                        f"the missing stack: {exc}"
                    )

            commands = log_path.read_text(encoding="utf-8")
            self.assertIn("-m pip install", commands)
            self.assertIn("torch==2.9.0", commands)
            self.assertIn("torch_npu==2.9.0.post2", commands)


if __name__ == "__main__":
    unittest.main()
