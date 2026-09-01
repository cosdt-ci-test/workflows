"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/modelscope/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by
``modelscope-quick-start.yml``):
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
from workflows.modelscope_cache import (
    ensure_safetensors,
    purge_corrupt_models,
    resolve_modelscope_cache,
)


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

    Scope: torch + torch_npu install + modelscope source install at the
    latest upstream release tag (``git clone`` + ``uv pip install -e
    '.[framework]'`` + ``transformers<5.0`` cap) + Qwen2.5-0.5B-Instruct
    download via the ``modelscope download`` CLI (lands in the
    bind-mounted ``~/.cache/modelscope``) + a single-card text-generation
    smoke on a 910B4 via ``AutoModelForCausalLM...to('npu:0')``.

    The NPU smoke deliberately goes through ``from_pretrained`` +
    ``.to('npu:0')`` (not ``pipeline(..., device='npu:0')``): modelscope's
    ``verify_device`` (``modelscope/utils/device.py``) only accepts
    ``cpu`` / ``cuda`` / ``gpu``, and the patched transformers
    ``AutoModelForCausalLM.from_pretrained`` forwards kwargs straight to
    transformers (no bare ``device=`` kwarg). ``import torch_npu`` in the
    doc body registers the ``npu`` device before ``.to``.
    """

    DEFAULT_COMMAND_TIMEOUT = 1200  # 20 min: cold git clone + framework install + 1 GB model download + 64-token generation
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. modelscope's ``[framework]``
    # extra pulls transformers / datasets / scipy etc.; on aarch64 the
    # cluster pip/uv resolver can still pick CUDA-typed transitive
    # wheels (e.g. nvidia-* via a loosely-pinned dep) that fail to run
    # on the NPU. Pre-exclude the whole CUDA toolchain so the resolver
    # falls back to pure-Python / CPU / CANN-typed alternatives.
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
    _CONSTRAINTS_FILE = '/tmp/modelscope_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``modelscope-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + cache
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv +
        validate the modelscope cache.

        ``modelscope`` itself is NOT installed here — the doc's
        ``## 安装 modelscope`` block exercises the source install path
        (``git clone`` at the latest release tag + ``uv pip install -e
        '.[framework]'``), so a broken install surfaces as a fuzzy
        mismatch against ``modelscope xxx`` rather than being masked by
        a pre-installed copy.
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

        # 2) uv: the doc's source-install block calls ``uv pip install``
        # which handles PEP 517 build deps more reliably than pip.
        # Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml
        # job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) ASCEND_RT_VISIBLE_DEVICES=0: the doc's ``check-torch`` step
        # hardcodes ``count: 1`` in its ``#test-result`` block. The
        # single-card runner label (``linux-aarch64-a2-1``) already
        # exposes one device, but set the env at process level so every
        # doc subprocess sees exactly card 0 (inherited via
        # ``env=os.environ.copy()``).
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        # 4) safetensors + cache validation: persistent host-side bind
        # mount (``/data/ci-cache/modelscope/modelscope`` ->
        # ``/root/.cache/modelscope``) can hold truncated safetensors
        # from interrupted downloads. Walk every shard under each model
        # dir and purge on failure; modelscope will re-download cleanly
        # on next access.
        ensure_safetensors()
        purge_corrupt_models(resolve_modelscope_cache())

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA
        constraints + uv + ASCEND_RT_VISIBLE_DEVICES + cache validation.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
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
