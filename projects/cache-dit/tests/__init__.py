"""Tests package marker.

Single responsibility: inject the repo root's ``src/`` into ``sys.path``
so that ``from workflows.markdown_doc_test_base import ...`` can resolve.

The ``import workflows`` statement also triggers ``src/workflows/__init__.py``
— that file installs the framework's own dependency (mistune) with a
``find_spec`` guard so warm runners don't redo the pip install. This file
no longer installs dependencies.

Why it lives here:
* unittest treats ``tests/`` as a package; the parent ``__init__.py``
  executes before any submodule import.
* It runs before ``tests/test_*.py`` import, which is the earliest
  opportunity to inject ``sys.path`` and trigger the framework package
  initialization.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap: make ``workflows.*`` resolvable + trigger
# src/workflows/__init__.py (which installs the framework deps).
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ -> cache-dit/ -> projects/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Trigger src/workflows/__init__.py: installs mistune (with find_spec guard)
import workflows  # noqa: F401  # side-effect import; we don't use the symbol