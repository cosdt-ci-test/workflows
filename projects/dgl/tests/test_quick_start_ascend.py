"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/dgl/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``dgl-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Injected by the engine but NOT consumed
                                  by the doc body. The upstream fork
                                  (BUPT-GAMMA/dgl-ascend) publishes no
                                  releases or tags, so the engine's ref
                                  fallback chain bottoms out at
                                  ``/commits/HEAD`` and anchors the fork's
                                  default-branch HEAD sha as the change
                                  key. The doc's clone block just clones
                                  the default branch (master) — exactly
                                  what a user gets.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.

The doc body is cwd-relative ("wherever you run it is the project
root"): clone to ``./dgl-ascend``, build there, examples run from the
clone's ``examples/pytorch/lightgcn``. CI pins the execution cwd in
``prepare_environment`` via ``os.chdir('/root/dgl-test')``.
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

    Scope (single card):
      * The doc installs the pinned torch stack (torch 2.9.0 +
        torch_npu 2.9.0.post2) — a deliberate attempt to reuse the
        CI's existing torch line rather than the fork's documented
        torch 2.8.0 / py3.10. The fork's CMake does NOT pin a torch
        version: it just ``find_package(Torch REQUIRED)`` and links
        ``c10_npu`` from torch_npu. ``import dgl`` + ``torch.npu.
        is_available()`` are asserted in the verify block.
      * DGL-Ascend itself is installed from source: ``git clone`` the
        fork + ``git submodule update --init --recursive`` +
        ``bash script/build_dgl_ascend.sh`` (CMake ``-DUSE_ASCEND=ON
        -DUSE_CUDA=OFF``, SOC=910B4) + ``pip install -e .``.
      * Smoke: LightGCN trains one epoch on gowalla (``wget`` the
        official dgl-data zip + ``unzip`` + ``python main.py
        --dataset gowalla --batch 2048 --recdim 64 --epochs 1
        --device npu``), asserted via the ``Average BPR Loss`` log line.
    """

    DEFAULT_COMMAND_TIMEOUT = 7200  # 2h baseline; source build of DGL rides on this
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
    )

    # Process-level CUDA exclusion list. The cluster's PyPI cache
    # auto-resolves transitive CUDA deps from torch on aarch64; those
    # CUDA wheels have no aarch64+torch_npu ABI match. Excluding them
    # with ``<0`` forces ``resolution-impossible`` early so the resolver
    # picks the pure-CPU / torch_npu wheels instead.
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
    _CONSTRAINTS_FILE = '/tmp/dgl_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # Doc execution cwd for CI: the doc body is cwd-relative (clone to
    # ./dgl-ascend, build there, examples run from the clone), so
    # "wherever the user runs it" is the project root. CI chdirs to
    # /root/dgl-test to keep the clone + build out of the checkout dir.
    _PROJECT_ROOT = '/root/dgl-test'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv +
    # torch stack probe + doc execution cwd + card pin
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv +
        probe torch stack + chdir to the doc cwd + pin card 0.

        ``dgl-ascend`` itself is NOT installed here — the doc's install
        block exercises the source build (``git clone`` + ``bash
        script/build_dgl_ascend.sh`` + ``pip install -e .``), so a
        broken build surfaces as a fuzzy mismatch in the doc's verify
        block rather than being masked by a pre-built copy.

        The doc body is cwd-relative; ``os.chdir(_PROJECT_ROOT)`` here
        makes every doc command run under /root/dgl-test (the engine
        executes each block with ``cwd=Path.cwd()``).
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

        # 2) uv: the doc's install block calls ``uv pip install ...``.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: reuse the image's torch stack if it
        # imports and sees the NPU; otherwise the doc's
        # `dgl-install-torch` #test-setup block installs the pinned stack.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            'raise SystemExit(0 if torch.npu.is_available() else 1)\n'
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
            print('setup: torch stack probe failed, doc install-torch will install the pinned stack')

        # 4) execution cwd: chdir to /root/dgl-test.
        try:
            os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        except OSError as exc:
            print(f'setup: doc cwd mkdir failed: {exc}')
        os.chdir(cls._PROJECT_ROOT)
        print(f'setup: cwd -> {os.getcwd()}')

        # 4.5) ASCEND_RT_VISIBLE_DEVICES=0: pin card 0 for the single-card smoke.
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class. The ``if _e2e_enabled()``
        body guard keeps heavy setup from firing when ``NPU_READY`` is unset."""
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()