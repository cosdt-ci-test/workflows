"""Run the human_action Quick Start with the shared Markdown executor."""

import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from textwrap import dedent, indent

from workflows.markdown_doc_test_base import MarkdownDocTestBase, TestCommand


_PROJECT = Path(__file__).resolve().parents[1]
_ENABLED = os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "cosdt-ci-test/vbench-quick-start"
    ERROR_MARKERS = (*MarkdownDocTestBase.ERROR_MARKERS, "ERR99999")

    def pre_process(self):
        return (_PROJECT / "docs/Quick-start-Ascend.md").read_text(encoding="utf-8")

    @staticmethod
    def _checked_api(body):
        prefix = dedent('''\
            import os
            from pathlib import Path
            import torch_npu
            import vbench.human_action as human_action
            from scripts.prepare_assets import UMT_FILENAME
            from scripts.runtime_checks import check_api

            cache = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench"))
            with check_api(human_action, cache / "umt_model" / UMT_FILENAME):
        ''')
        return prefix + indent(body, "    ")

    def execute(self, commands, results):
        api = [command for command in commands if isinstance(command, TestCommand)
               and command.id == "human-action-api"]
        if len(api) != 1 or api[0].language != "python":
            raise ValueError("Expected exactly one Python human-action-api example")
        checked = [replace(command, cmd=self._checked_api(command.cmd))
                   if command is api[0] else command for command in commands]
        super().execute(checked, results)

    @classmethod
    def setUpClass(cls):
        if not _ENABLED:
            return
        cann_env = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        if not Path(cann_env).is_file():
            raise RuntimeError(f"Missing CANN environment: {cann_env}")
        result = subprocess.run(
            ["bash", "-c", f"source {cann_env} >/dev/null 2>&1 && env"],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                os.environ[key] = value
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VBENCH_CACHE_DIR", "/root/.cache/vbench")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ["PIP_CONSTRAINT"] = str(_PROJECT / "constraints-npu.txt")
        os.environ["UV_CONSTRAINT"] = str(_PROJECT / "constraints-npu.txt")
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONNOUSERSITE"] = "1"
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    @unittest.skipUnless(_ENABLED, "requires Ascend NPU; set NPU_READY=true")
    def test_runs_doc(self):
        self.run_template()


if __name__ == "__main__":
    unittest.main()
