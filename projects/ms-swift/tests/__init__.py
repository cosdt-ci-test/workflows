"""Tests package marker。

两条职责，按顺序执行（在 ``tests/test_*.py`` 被 import **之前**完成）：

1. 把仓库根的 ``src/`` 加入 ``sys.path``，让
   ``from workflows.markdown_doc_test_base import …`` 能解析。
2. 确保 ``mistune`` 已装。``markdown_doc_test_base.py`` 的 line 39
   顶层 import mistune，发生在任何 ``tests/test_*.py`` import 之前；
   如果不预先装好 import 阶段就会 ModuleNotFoundError——比 setUpClass
   早，比 setup_for_test 早，**必须**在 unittest 解析包阶段就准备好。

为什么放这里：
* unittest 把 ``tests/`` 视作 package，子模块 import 时父包
  ``__init__.py`` 会被执行。
* ``__init__.py`` 在 ``tests/test_*.py`` import 之前跑，正好
  是注入 sys.path + 装框架依赖的最早时机。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

# 1) sys.path bootstrap：让 ``workflows.*`` 可解析
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ → ms-swift/ → projects/ → workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# 2) mistune 守卫：仅在缺失时安装，钉 ``>=3,<4`` 因为 _scan_blocks 依赖
# v3 AST 形态（attrs.info / block_html.raw）；v4 会让 CI 中途炸。
if importlib.util.find_spec('mistune') is None:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'mistune>=3,<4'],
        check=True,
    )