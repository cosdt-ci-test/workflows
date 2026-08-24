"""Quick-start-Ascend documentation test for DeepSpeed on Ascend NPU.

Built on top of ``MarkdownDocTestBase`` (shared engine). The document
under test is ``projects/deepspeed/docs/Quick-start-Ascend.md``, which
follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters.

Environment variables (injected by the shared engine
``quick-start-template.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; captured by the hidden
                                  ``#test-setup store="upstream_ref"`` block
                                  in the doc, then loaded into test commands
                                  where ``<UPSTREAM_REF>`` appears.
    ``NPU_READY=true``            Required; gates the E2E test class.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set."""
    return _is_truthy(os.environ.get('NPU_READY'))


def _ensure_torch_npu():
    """Install torch + torch_npu if not already available."""
    try:
        import torch
        import torch_npu
        print(f'setup: found torch {torch.__version__}, torch_npu {torch_npu.__version__}')
        return
    except ImportError:
        print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             '--extra-index-url', 'https://repo.huaweicloud.com/ascend/repos/pypi',
             '--trusted-host', 'repo.huaweicloud.com',
             'torch==2.9.0', 'torch_npu==2.9.0.post2'],
            check=True)
        import torch
        import torch_npu
        print(f'setup: installed torch {torch.__version__}, torch_npu {torch_npu.__version__}')


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            _ensure_torch_npu()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end tests require NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()