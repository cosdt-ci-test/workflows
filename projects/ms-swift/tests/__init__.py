"""Tests package marker。

唯一职责：把仓库根的 ``src/`` 加入 ``sys.path``，让随后
``from workflows.markdown_doc_test_base import …`` 能解析。

``import workflows`` 这条语句同时触发了 ``src/workflows/__init__.py``——
框架自身依赖（mistune）的安装在那里完成（带 ``find_spec`` 守卫，避免热
runner 重复 pip install）。本文件不再负责装依赖。

为什么放这里：
* unittest 把 ``tests/`` 视作 package，子模块 import 时父包
  ``__init__.py`` 会被执行。
* ``__init__.py`` 在 ``tests/test_*.py`` import 之前跑，正好是注入
  ``sys.path`` + 触发框架包初始化的最早时机。
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap：让 ``workflows.*`` 可解析 + 触发 src/workflows/__init__.py
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ → ms-swift/ → projects/ → workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# 触发 src/workflows/__init__.py：装 mistune 依赖（里面已经有 find_spec 守卫）
import workflows  # noqa: F401  # 仅触发包初始化，无实际符号使用
