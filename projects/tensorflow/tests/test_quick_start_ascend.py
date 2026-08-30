"""Execute the TensorFlow 1.15 + TF Adapter 9.1.0 Quick Start on NPU."""

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


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() == "true")


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get("NPU_READY"))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """Fetch, parse, and execute the labelled documentation blocks."""

    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "cosdt-ci-test/tensorflow-quick-start"
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        "GEInitialize failed",
        "NpuOptimizer init failed",
        "device open failed",
        "ERR99999",
        "EZ9999",
    )

    _CANN_SET_ENV_CANDIDATES = (
        "/usr/local/Ascend/cann/set_env.sh",
        "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    )
    _TFPLUGIN_INSTALL_PATH = "/root/.cache/tensorflow/tfplugin-9.1.0"
    _HDF5_LIBRARY_PATH = "/usr/local/hdf5/lib"

    @classmethod
    def prepare_environment(cls) -> None:
        cann_set_env = next(
            (path for path in cls._CANN_SET_ENV_CANDIDATES if os.path.isfile(path)),
            None,
        )
        if cann_set_env is None:
            raise RuntimeError(
                "CANN set_env.sh not found in expected image paths: "
                + ", ".join(cls._CANN_SET_ENV_CANDIDATES)
            )

        merged = subprocess.run(
            ["bash", "-c", f"source {cann_set_env} >/dev/null 2>&1; env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.startswith(("ASCEND", "HCCL")) or key in {
                "PATH",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
            }:
                os.environ[key] = value

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["TFPLUGIN_INSTALL_PATH"] = cls._TFPLUGIN_INSTALL_PATH
        os.environ["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (cls._TFPLUGIN_INSTALL_PATH, existing_pythonpath)
            if item
        )
        existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            item
            for item in (cls._HDF5_LIBRARY_PATH, existing_library_path)
            if item
        )
        os.environ["JOB_ID"] = "tensorflow-quick-start"
        os.environ["ASCEND_DEVICE_ID"] = "0"
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        print(f"setup: sourced CANN environment from {cann_set_env}")

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == "__main__":
    unittest.main()
