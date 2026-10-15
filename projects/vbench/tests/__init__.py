"""Make the shared Markdown runner importable from the project directory."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
