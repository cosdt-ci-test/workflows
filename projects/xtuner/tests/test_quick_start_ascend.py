"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/xtuner/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``xtuner-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release tag. The value is
                                  captured into ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<ref>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``.

    Scope: install + import + MODES smoke + two V1 NPU SFT smoke runs
    (LLM Qwen3-tiny on ``tests/resource/openai_sft.jsonl`` for 3 steps,
    MLLM Intern-S1-tiny on ``tests/resource/mllm_sft_*`` for 3 steps,
    both with ``TrainerConfig(dist_backend="npu:hccl")`` so the V1
    trainer goes through ``init_process_group(backend="hccl")`` on the
    real NPU). The V1 path was kept off the doc while DeepSpeed /
    flash-attn / GroupedGEMM on NPU was assumed unstable — that turned
    out to be wrong: xtuner V1 ships dedicated NPU kernels
    (``xtuner/v1/ops/flash_attn/npu.py``,
    ``xtuner/v1/ops/moe/npu/group_gemm.py``,
    ``xtuner/v1/ops/tensor_parallel/npu.py``) and a working upstream
    NPU submit script (``examples/v1/scripts/run_rl_submit_npu.sh``),
    so the doc was extended to cover both LLM and MLLM smoke runs.
    """

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 min: covers cold install + two 3-step SFT runs
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Originally written inside the
    # workflow step as a child-process env passed through to pip / uv /
    # xtuner's own wheel resolver. Moved to the test layer: write to /tmp
    # and export; subprocesses (subprocess.run inherits parent env by
    # default) see it the same way.
    _CUDA_CONSTRAINTS = (
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
    _CONSTRAINTS_FILE = '/tmp/xtuner_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``xtuner-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + torch stack probe
    # (xtuner itself + its v1.0.1 runtime deps are installed by the doc's
    #  `## 安装 xtuner` blocks via `uv pip install --no-deps xtuner` +
    #  manual dep install; no model download, so no modelscope cache binding)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv + torch stack probe.

        The doc's ``## 安装 xtuner`` section is the single source of truth
        for which xtuner version gets installed; this class only handles
        ``torch`` / ``torch_npu`` here (via the cluster cache + Huawei
        ascend dual-source). All other packages — ``xtuner`` itself plus
        its v1.0.1 runtime deps (``mmengine==0.11.0rc2`` /
        ``transformers==5.2.0`` / ``peft>=0.14.0`` etc.) — install
        themselves in document order via the ``#test`` machinery
        (``xtuner-install-binary`` / ``xtuner-install-source``).

        Why ``--no-deps`` on the xtuner line: ``bitsandbytes==0.45.0``
        (xtuner v0.2.0 + v1.0.1 hard pin) has no aarch64 wheel on PyPI
        (only manylinux_2_24_x86_64 + win_amd64), so a normal
        ``pip install xtuner`` on the aarch64 NPU runner hits
        ``ResolutionImpossible``. ``import xtuner`` itself never
        touches bitsandbytes (xtuner/__init__.py only does
        ``from mmengine.utils import digit_version`` +
        ``from .entry_point import cli``), and the two V1 SFT smoke
        runs (LLM ``Qwen3Dense8BConfig(num_hidden_layers=3,
        hidden_size=512)`` and MLLM ``InternS1MiniConfig(text_config=
        Qwen3Dense8BConfig(vocab_size=300, ...))``) both end up on
        ``xtuner/v1/train/toy_tokenizer.UTF8ByteTokenizer`` (Trainer
        fallback when ``tokenizer_path=None``) — neither embeds nor
        forward pass ever touches bitsandbytes. The doc body mirrors
        this rationale and installs the v1.0.1 runtime deps
        (mmengine + transformers + peft + datasets + einops + loguru +
        openpyxl + scikit-image + scipy + SentencePiece + tiktoken +
        transformers_stream_generator + cyclopts + opencv-python-headless +
        timm + pyarrow + pydantic + tensorboard + xxhash + imageio +
        py-libnuma + GitPython) explicitly.

        No modelscope cache bind-mount: the LLM/MLLM SFT smoke uses
        xtuner's own toy model configs (``examples/v1/config/
        sft_qwen3_tiny.py``, ``examples/v1/config/
        sft_intern_s1_tiny_config.py``) plus the toy datasets
        (``tests/resource/openai_sft.jsonl``,
        ``tests/resource/mllm_sft_text_example_data.jsonl``,
        ``tests/resource/mllm_sft_single_image_example_data.jsonl``,
        ``tests/resource/mscoco_dog_000000319154.jpg``), all shipped
        in the xtuner source tree after ``git clone`` + ``git checkout
        <ref>`` — no HF / ModelScope model download, so the host cache
        is not touched.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc's ``xtuner-install-source`` block calls
        # ``uv pip install -e .`` which handles PEP 517 build deps more
        # reliably than pip. Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe + install: when version matches the image's
        # pre-installed wheels, reuse them to avoid the cluster cache
        # triggering ``+cpu`` resolution.
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
            check=False,  # probe's success/failure is the branch signal — don't raise
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
            # Cold fallback: image's pre-installed wheels have drifted
            # away from torch 2.9.0 (e.g. newer base image rolled
            # forward). Installing torch + torch_npu without their
            # ABI-matched siblings leaves MLLM broken — Intern-S1's
            # vision tower pulls torchvision at import time, and the
            # upstream `examples/v1/config/sft_intern_s1_tiny_config.py`
            # imports timm via xtuner.v1.model. Pin all three to the
            # torch-2.9.0-aligned versions; the doc's `## 安装 xtuner`
            # section installs the rest of v1.0.1's deps explicitly.
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2 torchvision==0.24.0 timm')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                    'torchvision==0.24.0', 'timm',
                ],
                check=True,
            )

        # 4) xtuner itself is NOT installed here — the doc's
        # ``## 安装 xtuner`` blocks ``xtuner-install-binary`` /
        # ``xtuner-install-source`` exercise both binary and source install
        # paths against the upstream release tag injected by the workflow,
        # so a broken install block surfaces here as a fuzzy mismatch
        # against ``xtuner xxx`` rather than being masked by a
        # pre-installed copy.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv +
        torch stack.

        ``xtuner`` is NOT installed here — see ``prepare_environment`` for why.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY``
        is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()