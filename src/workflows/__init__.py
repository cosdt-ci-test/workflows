"""共享 workflow 工具与基类的命名空间包。

所有项目复用的源码放在这里（如 markdown 文档测试框架）。子模块通常通过
``sys.path`` 注入仓库根 + ``src/`` 后按 ``workflows.<模块名>`` 导入。

框架自身依赖
==============

``markdown_doc_test_base.py`` 顶层 ``import mistune``（line 39），比
``setUpClass`` 早、比 ``setUp`` 早——任何 import 触发 ``workflows.*``
之前必须先装。两种路径会触发本 ``__init__.py``：

- 项目 ``tests/__init__.py`` 注入 sys.path 后 ``import workflows`` 链式触发；
- 终端直接 ``python -c "import workflows"`` 测试时。

两层守卫：

1. ``importlib.util.find_spec('mistune')`` 只查 sys.path，不实际 import——
   已装就直接走，省一次 subprocess。
2. Python 解释器对包 ``__init__`` 自带去重，即使被多次 import 也只跑一次本
   模块体，所以冷 runner 装过一次后，热 runner 不会重复 pip install。

为什么 ``pip install`` 强制公网
--------------------------------

GitHub workflow 在 job-level 可以设 ``PIP_INDEX_URL`` 指向 cluster-internal
PyPI cache（NPU runner 上不可达）。框架 ``pip install`` 强制 ``--index-url``
指向公网 PyPI，与项目自身的 wheel 解析策略无关——即使外层 env 改了
PIP_INDEX_URL，框架照样能装。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

# 框架依赖：仅在缺失时安装。钉 ``>=3,<4`` 因为
# ``markdown_doc_test_base._scan_blocks`` 依赖 v3 AST 形态
#（attrs.info / block_html.raw）；v4 会让 CI 中途炸。
# 强制公网 PyPI，不受外层 PIP_INDEX_URL / cluster cache 影响。
if importlib.util.find_spec('mistune') is None:
    subprocess.run(
        [
            sys.executable, '-m', 'pip', 'install',
            '--index-url', 'https://pypi.org/simple',
            'mistune>=3,<4',
        ],
        check=True,
    )
