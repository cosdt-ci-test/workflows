"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/torchtune/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the ``quick-start-template.yml``
engine that ``torchtune-quick-start.yml`` delegates to):
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
from pathlib import Path

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

    The test subclass itself does not own any ``test_*`` method beyond the
    template-method entry; the doc body is the spec. ``prepare_environment``
    makes sure ``torch_npu`` is importable + the cluster ``pip`` mirror is
    bound before the framework starts executing doc commands (the doc
    body itself does ``uv pip install torchtune`` and ``uv pip install -e .``
    on a ``<ref>`` checkout, but the baseline ``torch`` / ``torch_npu`` /
    ``modelscope`` deps still need to be on the runner for the doc to do
    anything useful).
    """

    # 60 min per command: long enough for the ~1 GB Qwen2.5-0.5B
    # snapshot_download + a 3-step tune run (cold cache + first NPU
    # compile); short enough to fail fast on hangs.
    DEFAULT_COMMAND_TIMEOUT = 3600

    # Monitored source is the cosdt-ci-test/workflows fork (this repo):
    # the doc lives at projects/torchtune/docs/Quick-start-Ascend.md and
    # the engine sets MONITORED_DOC_URL to the raw.githubusercontent.com
    # URL for the same path. Upstream torchtune's first finetune tutorial
    # lives at meta-pytorch.org/torchtune/0.6/tutorials/first_finetune_tutorial.html
    # and is referenced from inside the doc body, but it is NOT the file
    # under test.
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the other
    # projects' tests: write to /tmp and export, so subprocesses
    # (subprocess.run inherits parent env by default) see it. torchtune's
    # own dependency tree pulls `datasets` + `huggingface_hub` which can
    # transitively drag in CUDA wheels without this constraint.
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
    _CONSTRAINTS_FILE = '/tmp/torchtune_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + torch stack probe
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + uv + torch stack probe.

        torchtune itself is intentionally NOT pre-installed here: the doc
        body runs ``uv pip install torchtune`` (binary path) and
        ``uv pip install -e .`` (source path) against the upstream release
        tag injected by the workflow, so the test exercises the exact
        install path users get.

        What this hook does pre-install:
            * CANN env (so torch_npu is importable);
            * CUDA exclusion list (so an accidental ``nvidia-cudnn`` pull
              doesn't shadow the NPU build);
            * torch / torch_npu, only if the image's pre-installed wheels
              don't already match the version matrix.
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
                # workflow; only fill in CANN keys that are missing.
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

        # 2) uv: the doc body's binary install path passes
        # ``--index-url https://mirrors.aliyun.com/pypi/simple`` explicitly
        # to dodge the cluster cache (the cluster PyPI mirror doesn't
        # always have a fresh torchtune wheel on release day), so the
        # ``PIP_INDEX_URL`` / ``UV_INDEX_URL`` job-level env in
        # ``quick-start-template.yml`` is overridden for that command and
        # doesn't apply here. The source install path (``uv pip install -e .``)
        # has no explicit index-url and picks up the cluster env. uv's
        # output is also cleaner than pip's "Obtaining/Installing
        # collected packages" preamble so the #test-result fuzzy match
        # against ``torchtune xxx`` isn't polluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when version matches the image's
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
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 4) ``modelscope`` is NOT pre-installed here: the doc's
        # ``check-ml-deps`` block installs it via the doc body itself,
        # so a broken install block surfaces here as a fuzzy mismatch
        # against ``modelscope xxx`` rather than being masked by a
        # pre-installed copy.
        #
        # ``torchtune`` is also NOT pre-installed: it's the subject of
        # the test and gets installed by the doc's ``## 安装 torchtune``
        # blocks (``torchtune-install-binary`` / ``torchtune-install-source``)
        # which exercise both binary and source install paths.

        # Modelscope cache: pinned to a test-scoped subdir outside the
        # bind-mount. The host-side /data/ci-cache/modelscope persists
        # across CI runs and accumulates stale files (incomplete downloads,
        # model revision drift, cross-project leftovers) that would
        # otherwise surface here as hash mismatches. This keeps each run's
        # cache isolated in the container's local fs.
        os.environ.setdefault(
            'MODELSCOPE_CACHE', str(Path.home() / '.cache' / 'modelscope_quick_start_test'),
        )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv + torch stack.

        ``modelscope`` / ``torchtune`` are NOT installed here — the doc's
        own labeled blocks install them in document order, so a broken
        install block fails loudly instead of being masked.

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