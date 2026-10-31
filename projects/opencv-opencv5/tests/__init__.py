"""Tests package marker (opencv-opencv5 guard).

Single responsibility: inject the repo root's ``src/`` into ``sys.path``
so that ``from workflows.markdown_doc_test_base import ...`` can resolve.

Framework deps (mistune) are installed by the common quick-start workflow
template, not at import time here.

Why it lives here:
* unittest treats ``tests/`` as a package; the parent ``__init__.py``
  executes before any submodule import.
* It runs before ``tests/test_*.py`` import, which is the earliest
  opportunity to inject ``sys.path``.

Why this file is not optional:
* CI run 40993454296 died at ``python -m unittest tests.test_quick_start_ascend_opencv5``
  with ``ModuleNotFoundError: No module named 'workflows'`` -- without
  this bootstrap the ``workflows.*`` import in the test module cannot
  resolve and the guard never reaches the doc.
* ``python -m unittest tests.<module>`` imports the ``tests`` package
  first, so this file runs before the test module's own imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap: make ``workflows.*`` resolvable.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ -> opencv-opencv5/ -> projects/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
