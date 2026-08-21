"""Quick-start-Ascend 文档测试：基于 ``MarkdownDocTestBase`` 契约的端到端用例。

被测文档：``projects/ms-swift/docs/Quick-start-Ascend.md``（遵循
``docs/markdown_doc_test_label.md`` 契约：每个 ``shell`` 代码块带 ``#test`` /
``#test-setup`` / ``#test-result`` 标签与 ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` 参数）。

跑法：``python -m unittest tests.test_quick_start_ascend -v 2>&1``

环境变量（由 GitHub workflow ``ms-swift-quick-start.yml`` 注入）：
    ``MONITORED_DOC_URL``         必填，被测文档的原始 URL。
    ``UPSTREAM_REF``              必填，bash 直接读 ``$UPSTREAM_REF`` 拿到
                                  最新 release tag；通过 ``#test-setup
                                  store="upstream_ref"`` 的 stdout 注入
                                  ``captures``，最终替换 doc 命令体中的 ``<ref>``。
    ``NPU_READY=true``            必填，否则整个类跳过。端到端测试只在 NPU runner
                                  上跑：本地开发机 / 普通 ubuntu runner 没有
                                  ``/dev/davinci*`` 设备，硬跑会因
                                  ``import torch_npu`` 失败。
                                  ``SWIFT_NPU_E2E`` 已废弃（v1 老测试遗留）。

端到端测试只在 NPU runner 上跑：本地开发机 / 普通 ubuntu runner 没有
``/dev/davinci*`` 设备，硬跑会因 ``import torch_npu`` 失败。
"""

from __future__ import annotations

import os
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
    # prepare_environment：CUDA 约束 + uv + torch 栈探测 + transformers/peft
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env + CUDA 约束 + uv + torch 栈探测 + transformers/peft 一气装好。

        类级 setup：整个测试类只装一次，由 ``setUpClass`` 调一次得起。
        无关 ``unittest.TestCase.setUp``（那是个生命周期 hook，会在每个
        test method 跑前都调一次——不适合做前置安装）。
        """
        # 0) CANN env：source set_env.sh 后拿 env 流，merge 进 os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # 不覆盖 workflow 显式注入的 env（jobs.env / steps.env）；
                # 只补 CANN 缺失的键，避免冲突。
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA 排除清单 + 进程级 env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv：test 的 setup 会调 ``uv pip install``，比 pip 处理
        # PEP 517 build deps 更稳。继承外层 ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        #（yml job-level env 设的 cluster cache 路径与 trusted-host）。
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
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    '--trusted-host', cls._CLUSTER_TRUSTED,
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
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """整套测试类只跑一次 env setup：CUDA 约束 + uv + torch 栈 +
        transformers/peft + CANN env。``@unittest.skipIf`` 装饰器在
        未设 ``NPU_READY`` 时让整个类跳过，``setUpClass`` 也不会被调。
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """模板方法入口。基类 ``run_template()`` 跑完 ``pre_process`` ->
        ``parse`` -> ``execute`` -> ``post_process`` 全流程。``prepare_environment``
        由 ``setUpClass`` 调一次，不在 ``run_template`` 里。"""
        self.run_template()


if __name__ == '__main__':
    unittest.main()
