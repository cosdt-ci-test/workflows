"""Tests package marker.

通过 ``tests`` 父包隐式把仓库根的 ``src/`` 加入 ``sys.path``，让
``from workflows.markdown_doc_test_base import …`` 在任何子测试
模块（``tests.<project>.test_*``）里都能解析。所有项目测试文件
不再重复 ``sys.path.insert`` boilerplate。

为什么放这里：
* unittest 把 ``tests/`` 视作 package，子模块 import 时父包
  ``__init__.py`` 会被执行。
* ``__init__.py`` 在 ``tests/test_*.py`` import 之前跑，正好
  是注入 sys.path 的最早时机——无须 ``noqa: E402``。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ → ms-swift/ → projects/ → workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)