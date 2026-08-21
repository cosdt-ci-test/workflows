"""Quick-start-Ascend 文档测试：基于 ``MarkdownDocTestBase`` 契约的端到端用例。

被测文档：``projects/ms-swift/docs/Quick-start-Ascend.md``（遵循
``docs/markdown_doc_test_label.md`` 契约：每个 ``shell`` 代码块带 ``#test`` /
``#test-setup`` / ``#test-result`` 标签与 ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` 参数）。

跑法：``python -m unittest tests.test_quick_start_ascend -v 2>&1``

环境变量（由 GitHub workflow ``ms-swift-quick-start.yml`` 注入）：
    ``MONITORED_DOC_URL``         必填，被测文档的原始 URL。
    ``UPSTREAM_REF``              可选，``load="upstream_ref>>ref"`` 的
                                  ``upstream_ref`` 实际取值。
    ``UPSTREAM_COMMIT``           可选，被 ``pre_process`` 用于把 doc 中的
                                  ``<UPSTREAM_REF>`` 占位符替换成确切 SHA。
    ``SWIFT_NPU_E2E`` 已废弃（v1 老测试遗留），新约定一律用 ``NPU_READY``。
                                  CI runner 上设 ``NPU_READY=true`` 解除
                                  skip；本地开发机不设也能 import / 静态
                                  检查通过（类直接 skip）。

端到端测试只在 NPU runner 上跑：本地开发机 / 普通 ubuntu runner 没有
``/dev/davinci*`` 设备，硬跑会因 ``import torch_npu`` 失败。
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase

def _is_truthy(value: str | None) -> bool:
    """``'true'`` → True（大小写不敏感），其它（含未设）→ False。"""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """``NPU_READY=true`` 时放开 skip。"""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` 端到端测试：拉 doc -> 校验契约 -> 顺序执行
    ``#test-setup`` / ``#test`` -> 比对 ``#test-result``。"""

    # swift sft 全量训练可能跑 30+ 分钟；覆盖基类 1800s 默认值。
    DEFAULT_COMMAND_TIMEOUT = 3600

    # 进程级 CUDA 排除清单。原写在 workflow step 内，作为子进程 env 透传给
    # pip / uv / swift 自身的 wheel 解析。挪到测试层后写到 /tmp 并 export，
    # 子进程 (subprocess.run 默认继承父 env) 同样生效。
    _CUDA_CONSTRAINTS = (
        'modelscope==1.37.0',
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/ms_swift_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + 华为云 ascend 双源镜像。
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _CLUSTER_TRUSTED = 'cache-service.nginx-pypi-cache.svc.cluster.local'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit：source 一次拿到 ASCEND_HOME / LD_LIBRARY_PATH 等 env。
    # 路径写死，跟 GitHub workflow container 镜像 (CI_IMAGE) 绑定。
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # pre_setup：CUDA 约束 + uv + torch 栈探测 + transformers/peft
    # ----------------------------------------------------------

    def pre_setup(self) -> None:
        """CANN env + CUDA 约束 + uv + torch 栈探测 + transformers/peft 一气装好。

        原写死在 ``ms-swift-quick-start.yml`` 的 ``Run quick start test``
        step 里，每次 cycle 都重做一遍。现在挪到测试层，由 ``setUpClass``
        触发一次；workflow 该 step 只剩 ``python -m unittest …``。

        关键点：
        * CANN env（``ASCEND_HOME`` 等）通过 ``bash -c 'source X && env'``
          注入到 ``os.environ``；后续 ``swift sft`` / ``swift infer`` 子
          进程自动继承。Path 写死，跟 runner 镜像绑定。
        * 不硬钉 torch 版本：镜像里是 ``2.9.0+cpu``，cluster cache 不认
          ``+cpu`` 这种 local version label，会去外部 simple page 查然后
          失败（064a5d7 / 7136ed1 都栽过）。先 probe，匹配就跳过。
        * ``torch.__version__ == '2.9.0+cpu'`` 的 ``+cpu`` 是 libtorch
          构建变体名，不是运行时；计算走 ``torch_npu`` (CANN 后端)。
        * ``PIP_CONSTRAINT`` / ``UV_CONSTRAINT`` 是进程级 env，对 doc 里
          ``#test-setup pip install ms-swift -U`` 那段也生效——前提是
          子进程继承父进程 env（Python ``subprocess.run`` 默认如此）。
        """
        # 0) CANN env：source set_env.sh 后拿 env 流，merge 进 os.environ
        if os.path.isfile(self._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {self._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # 不覆盖 workflow 显式注入的 env（jobs.env / steps.env）；
                # 只补 CANN 缺失的键，避免冲突。
                os.environ.setdefault(key, value)
            self.log('pre_setup: sourced CANN env from set_env.sh')
        else:
            self.log(
                f'pre_setup: skipping CANN env source ({self._CANN_SET_ENV} not present)'
            )

        # 1) CUDA 排除清单 + 进程级 env
        with open(self._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(self._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = self._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = self._CONSTRAINTS_FILE

        # 2) uv：test 的 setUpClass 会调 ``uv pip install``，比 pip 处理
        # PEP 517 build deps 更稳。
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch 栈探测：版本匹配则复用镜像预装的 wheel，避免再走
        # cluster cache 触发 ``+cpu`` 解析。
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,  # probe 的成败本身就是分支信号，不能 raise
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            self.log(f'pre_setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            self.log('pre_setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', self._CLUSTER_INDEX,
                    '--extra-index-url', self._ASCEND_EXTRA,
                    '--trusted-host', self._CLUSTER_TRUSTED,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 4) transformers / peft：doc 表格钉死的版本约束。
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'transformers<5.0', 'peft<0.19'],
            check=True,
        )

    # ----------------------------------------------------------
    # pre_process：拉 doc + 把 <UPSTREAM_REF> 替换成实际 commit
    # ----------------------------------------------------------

    # <UPSTREAM_REF> 出现形态：单 token，前后空白/标点分隔。
    _UPSTREAM_REF_PATTERN = re.compile(r'<UPSTREAM_REF>')

    def pre_process(self) -> str:
        """拉被测文档，并把 ``<UPSTREAM_REF>`` 替换成 workflow 注入的 SHA。

        替代基类默认实现：基类只读 ``MONITORED_DOC_URL`` 拿 doc 文本，不做
        占位符替换。``Quick-start-Ascend.md`` 的源码安装块写
        ``cd ms-swift && git checkout <UPSTREAM_REF>``——必须替换成确切
        SHA 后才能在 NPU runner 上 checkout 到对应 commit。
        """
        text = super().pre_process()
        upstream_commit = os.environ.get('UPSTREAM_COMMIT', '').strip()
        if upstream_commit:
            text = self._UPSTREAM_REF_PATTERN.sub(upstream_commit, text)
            self.log(
                f'pre_process: substituted <UPSTREAM_REF> -> '
                f'{upstream_commit[:12]}'
            )
        # UPSTREAM_REF 注入到子进程环境，runner 的 capture 路径靠它。
        # 若用户显式设过不要覆盖；否则用 UPSTREAM_COMMIT 兜底。
        os.environ.setdefault('UPSTREAM_REF', upstream_commit)
        return text

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """整套测试类只跑一次 env setup：CUDA 约束 + uv + torch 栈 +
        transformers / peft。subsequent test 方法不会再装一遍。

        受 ``NPU_READY`` 门控：本地开发机不设环境变量时，连 ``import
        torch_npu`` 都不该尝试；只跑静态解析 / skip 检查。
        """
        if not _e2e_enabled():
            return
        cls().pre_setup()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """模板方法入口。基类 ``run_template()`` 跑完 ``pre_setup`` ->
        ``pre_process`` -> ``parse`` -> ``execute`` -> ``post_process``
        全流程。"""
        self.run_template()


if __name__ == '__main__':
    unittest.main()
