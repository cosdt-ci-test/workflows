"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/xdit/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``xdit-quick-start.yml``):
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
    ``NPU_COUNT``                 Optional; number of visible NPU devices the
                                  CI runner surfaces. ``0`` / unset -> skip.
                                  1 -> single-card mode; >=2 -> multi-card
                                  mode. The doc drives both blocks; the
                                  engine injects this so the test can short-
                                  circuit the multi-card block on 1-card
                                  runners without parsing the doc twice.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.modelscope_cache import purge_corrupt_models, resolve_modelscope_cache


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

    Scope (single + multi-card):
      * Single-card smoke: ``xdit --ulysses_degree 1 --ring_degree 1`` on
        SD 3.5 medium (ModelScope snapshot). Brings the install +
        import + xfuser runtime + runner init + ``epoch time`` print
        round-trip online on 1 card.
      * Multi-card smoke (>=2 NPU): ``xdit --ulysses_degree 2
        --ring_degree 1`` runs ``torchrun --nproc_per_node=2 -m
        xfuser.runner ...`` so xfuser's CLI bridge auto-spawns
        ``torch.distributed.run``; two ranks then ``init_process_group
        (backend="hccl")`` and ``xfuser/envs.get_torch_distributed_backend
        ()`` returns ``"hccl"`` (verified by the doc's
        ``xfuser-detect-npu`` block). PR #566 / mainline restrict NPU
        paths to single-node DP / USP / CFG-parallel (no PipeFusion),
        which the doc's ``--ulysses_degree 2 --ring_degree 1
        --pipefusion_parallel_degree 1`` (default) keeps within.
    """

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 min baseline; subclass hooks may bump per block
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
    )

    # Process-level CUDA exclusion list. Originally written inside the
    # workflow step as a child-process env passed through to pip / uv /
    # xfuser's own wheel resolver. Moved to the test layer: write to /tmp
    # and export; subprocesses (subprocess.run inherits parent env by
    # default) see it the same way.
    #
    # Why: the cluster's PyPI cache (Nginx frontend behind the test
    # job's UV_INDEX_URL) auto-resolves transitive CUDA deps from
    # torch / diffusers (e.g. nvidia-cublas-cu12 pulled in via aarch64
    # wheel resolution on torch 2.9.0). On an aarch64 NPU runner those
    # CUDA wheels have no aarch64+torch_npu ABI match and break xfuser's
    # import chain. Excluding them with ``<0`` forces
    # ``resolution-impossible`` early so the resolver picks the
    # pure-CPU / torch_npu wheels instead.
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
    _CONSTRAINTS_FILE = '/tmp/xdit_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (CI_IMAGE).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ModelScope cache root used by the doc's ``#test-setup
    # store="model_path"`` step (snapshot_download). The CI runner
    # bind-mounts a host-side persistent cache here so SD 3.5 medium
    # weights (~4 GB) survive across runs.
    #
    # The mount is provided by the workflow's ``container_options::
    # --volume=/data/ci-cache/modelscope/xdit:/root/.cache/modelscope``.
    # Inheriting the standard env (``MODELSCOPE_CACHE``) keeps the same
    # resolution path users have locally and matches the rest of the
    # guard repo's projects (ms-swift, diffusers, xtuner, etc.).
    _MODELSCOPE_CACHE_SUBDIR = 'xdit'  # the ``<project>`` token in the bind-mount path

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + modelscope cache validation
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + uv + modelscope cache
        validation in one go.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.

        xfuser itself + its NPU-critical transitive deps
        (``diffusers`` / ``transformers`` / ``accelerate`` / ``peft`` /
        ``einops`` / ``sentencepiece`` / ``beautifulsoup4`` /
        ``modelscope`` / ``yunchang`` / ``distvae`` / ``av``) install
        themselves in document order via the ``#test`` machinery; this
        class only handles torch stack + uv + modelscope cache health.
        ``torch`` / ``torch_npu`` already ship in the CANN 9.1.0 base
        image (`torch==2.9.0+cpu` / `torch_npu==2.9.0.post2`) so the
        doc's install step is a no-op idempotent reinstall against the
        cluster cache — this keeps the doc's install instructions
        honest (a user running on a bare Ubuntu CANN image still gets
        the right torch pinned).
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

        # 2) uv: the doc's ``xfuser-install-source`` block calls
        # ``uv pip install -e .`` which handles PEP 517 build deps more
        # reliably than pip. Inherit ``PIP_INDEX_URL`` +
        # ``PIP_TRUSTED_HOST`` from the yml job-level env (cluster cache
        # path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) modelscope cache: persistent host-side bind mount can hold
        # truncated safetensors from interrupted runs. Walk every shard
        # under each model dir and purge it on failure; modelscope
        # will re-download cleanly on next access. Implementation
        # lives in workflows.modelscope_cache; see that module's
        # docstring for the full rationale.
        purge_corrupt_models(resolve_modelscope_cache())

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA
        constraints + uv + modelscope cache validation.

        ``@unittest.skipIf`` only skips the test *method* —
        ``setUpClass`` itself always runs. The ``if _e2e_enabled()``
        body guard below is what actually keeps heavy setup from
        firing when ``NPU_READY`` is unset.
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
