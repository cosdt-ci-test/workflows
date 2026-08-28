"""Cheap contract: example run script, Quick Start doc, and README
share the same transfer success / failure markers.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_RUN = (_PROJECT / 'scripts' / 'run_example.sh').read_text(encoding='utf-8')
_DOC = (_PROJECT / 'docs' / 'Quick-start-Ascend.md').read_text(encoding='utf-8')
_README = (_PROJECT / 'README.md').read_text(encoding='utf-8')

_FAIL_MARKERS = (
    'Failed to install Ascend transport',
    'getTransferStatus FAILED',
    'Sync data transfer timeout',
)
_SUCCESS_MARKERS = (
    'Success to initialize adxl engine',
    'Test completed:',
)


class TestGuardMarkers(unittest.TestCase):
    def test_fail_markers_aligned(self) -> None:
        for marker in _FAIL_MARKERS:
            with self.subTest(marker=marker):
                self.assertIn(marker, _RUN)
                self.assertIn(marker, _DOC)
                self.assertIn(marker, _README)

    def test_success_markers_aligned(self) -> None:
        for marker in _SUCCESS_MARKERS:
            with self.subTest(marker=marker):
                self.assertIn(marker, _RUN)
                self.assertIn(marker, _DOC)
                self.assertIn(marker, _README)

    def test_device_buffer_proof_aligned(self) -> None:
        self.assertIn('npu:[0-9]+', _RUN)
        self.assertIn('Success to initialize adxl engine', _DOC)
        self.assertIn('npu:<logicid>', _README)
        self.assertIn('mem type:device', _RUN)
        self.assertIn('mem type:device', _README)


if __name__ == '__main__':
    unittest.main()
