"""Tests package marker.

Single responsibility: inject the repo root's ``src/`` into ``sys.path``
so that ``from workflows.markdown_doc_test_base import ...`` can resolve.

Framework deps (mistune) are installed by the common quick-start workflow
template, not at import time here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
