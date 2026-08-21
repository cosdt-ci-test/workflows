"""共享 workflow 工具与基类的命名空间包。

所有项目复用的源码放在这里（如 markdown 文档测试框架）。子模块通常通过
``sys.path`` 注入仓库根 + ``src/`` 后按 ``workflows.<模块名>`` 导入。
"""

from __future__ import annotations