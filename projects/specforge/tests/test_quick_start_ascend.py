"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/specforge/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``specforge-quick-start.yml``):
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

    The test subclass itself does not own any ``test_*`` method beyond the
    template-method entry; the doc body is the spec. ``prepare_environment``
    makes sure ``torch_npu`` is importable + the cluster ``pip`` mirror is
    bound + ``uv`` is on PATH before the framework starts executing doc
    commands (the doc body itself does ``uv pip install specforge`` and
    ``uv pip install -e .`` against the upstream release tag injected by
    the workflow, but the baseline ``torch`` / ``torch_npu`` / ``modelscope``
    deps still need to be on the runner for the doc to do anything useful).
    """

    # 90 min per command: long enough for the smoke script which does
    # model snapshot_download (~5–10 min cold cache, ~8 GB Qwen3.5-4B)
    # + SGLang first-time graph compile on NPU (5–10 min) + the 1-step
    # specforge train. Short enough to fail fast on a hang.
    DEFAULT_COMMAND_TIMEOUT = 5400

    # Monitored source is the cosdt-ci-test/workflows fork (this repo):
    # the doc lives at projects/specforge/docs/Quick-start-Ascend.md.
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
    # (subprocess.run inherits parent env by default) see it. specforge's
    # own dependency tree pulls ``accelerate`` / ``transformers`` /
    # ``huggingface-hub`` which can transitively drag in CUDA wheels
    # without this constraint.
    _CUDA_CONSTRAINTS = (
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nrtc<0',
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
    _CONSTRAINTS_FILE = '/tmp/specforge_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + uv in one go.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.

        What this hook does NOT pre-install:
            * ``torch`` / ``torch_npu`` / ``sglang``: the doc's
              ``check-torch`` block installs them, so a broken install
              block surfaces here as a fuzzy mismatch against
              ``torch=2.11.0+cpu`` / ``torch_npu=2.11.0`` rather than
              being masked by a pre-installed copy that happens to
              satisfy the assertion.
            * ``modelscope``: same reasoning — the doc's
              ``install-deps`` block pulls it.
            * ``specforge``: same reasoning — the doc's
              ``specforge-install-source`` block pulls it from the
              upstream release tag injected by the workflow.
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

        # 2) uv: the doc body's install commands all use ``uv pip
        # install``, which handles PEP 517 build deps more reliably
        # than pip. Inherits ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path +
        # trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA
        constraints + uv.

        ``torch`` / ``torch_npu`` / ``modelscope`` / ``specforge`` are
        NOT installed here — the doc's own labeled blocks install them
        in document order, so a broken install block fails loudly
        instead of being masked by a pre-installed copy.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below
        is what actually keeps heavy setup from firing when
        ``NPU_READY`` is unset.
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