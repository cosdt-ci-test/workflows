"""Shared workflow utilities and base classes.

This is the namespace package for code reused across projects (e.g. the
markdown documentation test framework). Submodules are typically imported
as ``workflows.<module>`` after ``sys.path`` is set up to include the
repo root and ``src/``.

Framework dependencies
======================

``markdown_doc_test_base.py`` does ``import mistune`` at the top of the
module (line 39), which runs before ``setUpClass`` and before ``setUp``,
so any import that triggers ``workflows.*`` requires ``mistune`` to be
installed first. Two paths trigger this ``__init__.py``:

- Project ``tests/__init__.py`` injects sys.path then ``import workflows``
  (chain trigger).
- Terminal use, e.g. ``python -c "import workflows"``.

Two layers of guard:

1. ``importlib.util.find_spec('mistune')`` only inspects sys.path; if
   already installed, no subprocess is spawned.
2. Python's package ``__init__`` is naturally idempotent — even on
   multiple imports, this module body runs at most once per process, so
   a cold runner installs once and a warm runner never re-pip-installs.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

# Framework dependency: only install when missing. The ``>=3,<4`` pin is
# required because ``markdown_doc_test_base._scan_blocks`` relies on the
# v3 AST shape (``attrs.info`` / ``block_html.raw``); v4 breaks CI mid-run.
# Inherit ``PIP_INDEX_URL`` / ``PIP_TRUSTED_HOST`` from the workflow
# job-level env (cluster cache + trusted-host).
if importlib.util.find_spec('mistune') is None:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'mistune>=3,<4'],
        check=True,
    )
