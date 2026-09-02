"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/xdit/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``xdit-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Injected by the engine but NOT consumed
                                  by the doc body: the doc installs
                                  ``xfuser`` from PyPI, so there is no
                                  source checkout to pin. The monitor
                                  still resolves the latest release id
                                  as the change key.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.

The doc body is cwd-relative ("wherever you run it is the project
root"): script + outputs land in the cwd (``./results``); weights go
to the default modelscope hub cache via the embedded
``snapshot_download``. CI pins the execution cwd in
``prepare_environment`` via ``os.chdir('/root/xdit-test')``.
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

    Scope (single card):
      * The doc installs the pinned torch stack (torch 2.9.0 +
        torch_npu 2.9.0.post2 + triton 3.5.* — xfuser >= 0.6.0 imports
        triton unconditionally in ``core/sparge_attention/block_mask.py``
        without declaring it) + installs xfuser + modelscope from PyPI
        (``uv pip install xfuser "modelscope==1.37.0"``; xfuser 0.6.0
        is a pure-Python wheel, flash-attn is only an optional extra)
        and verifies the NPU dispatch (``xfuser/envs.py`` returns
        ``get_torch_distributed_backend() == "hccl"``).
      * Smoke: a minimal single-card SD3 script (heredoc-written
        ``sd3_npu.py``) resolves the model path via an embedded
        ``modelscope.snapshot_download('stabilityai/
        stable-diffusion-3-medium-diffusers')`` (full repo ~30 GB into
        the default hub cache, bind-mounted from the host) —
        ``torchrun --nproc_per_node=1`` initialises the hccl runtime,
        ``xFuserStableDiffusion3Pipeline`` generates one 256x256
        1-step image, saved to ``results/sd3_npu.png`` and structurally
        verified (PNG magic + size floor).
      * Multi-card paths (USP / DP / TP) are pointer-only (a doc note
        linking upstream examples); the guard exercises the single-card
        path only.
    """

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 min baseline; cold SD3 download (~30 GB) rides on the yml-level budget
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

    # Doc execution cwd for CI: the doc body is cwd-relative (the smoke
    # writes sd3_npu.py and ./results/ into the cwd) so "wherever the
    # user runs it" is the project root. CI chdirs to /root/xdit-test
    # to keep the heredoc script + output images out of the checkout
    # dir. Model weights land in the default modelscope hub cache
    # (~/.cache/modelscope), which the workflow yml bind-mounts from
    # the host (/data/ci-cache/modelscope/xdit) for persistence.
    _PROJECT_ROOT = '/root/xdit-test'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv +
    # torch stack probe + doc execution cwd + card pin + cache
    # validation (xfuser install + model pull live in the doc body)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv +
        probe torch stack + chdir to the doc cwd + validate the
        modelscope cache.

        ``xfuser`` itself is NOT installed here — the doc's install
        block exercises the PyPI install path (``uv pip install
        xfuser``), so a broken release surfaces as a fuzzy mismatch in
        the doc's verify block rather than being masked by a
        pre-installed copy.

        The doc body is cwd-relative; ``os.chdir(_PROJECT_ROOT)`` here
        makes every doc command run under /root/xdit-test (the engine
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

        # 2) uv: the doc's install block calls ``uv pip install xfuser``
        # which resolves deps with PEP 517 reliability. Inherit
        # ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml job-level
        # env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when a pre-installed stack imports and
        # sees the NPU, reuse it (bare-metal / images that ship torch).
        # The plain CANN base image ships none - the probe fails and
        # the doc's `xdit-install-torch` #test-setup block installs the
        # pinned stack (torch 2.9.0 + torch_npu 2.9.0.post2 +
        # triton 3.5.*), which is then what we test against.
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
            # Cold fallback: no usable torch stack yet (the plain CANN
            # base image ships none). The doc's install-torch block
            # installs the pinned stack; this branch just records that
            # the probe didn't match (useful diagnostic in CI logs).
            print('setup: torch stack probe failed, doc install-torch will install the pinned stack')

        # 4) execution cwd: chdir to /root/xdit-test — the doc body is
        # cwd-relative (the smoke's heredoc script + ./results/ land in
        # the cwd) and the engine runs every block with
        # cwd=Path.cwd(). Keeps run artifacts out of the checkout dir.
        try:
            os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        except OSError as exc:
            print(f'setup: doc cwd mkdir failed: {exc}')
        os.chdir(cls._PROJECT_ROOT)
        print(f'setup: cwd -> {os.getcwd()}')

        # 4.5) ASCEND_RT_VISIBLE_DEVICES=0: the cluster's NPU runner
        # label is `linux-aarch64-a2-2` (2 cards); the cluster
        # device-plugin still passes both /dev/davinci* into the
        # container. The doc's smoke is single-card, so pin card 0 at
        # process level here — the value is inherited by every doc
        # subprocess (MarkdownDocTestBase passes ``env=os.environ.copy()``
        # to each subprocess).
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        # 5) safetensors + cache validation: the modelscope hub cache
        # is bind-mounted from the host (the doc's embedded
        # snapshot_download writes the full SD3 repo there, ~30 GB), so
        # interrupted runs can leave truncated shards behind. Walk every
        # shard and purge the model dir on failure; modelscope will
        # re-download cleanly on next access.
        # ensure_safetensors pulls in safetensors (transitive of
        # torch); install defensively in case the CANN base image
        # rolls forward.
        ensure_safetensors()
        try:
            purge_corrupt_models(resolve_modelscope_cache())
        except Exception as exc:
            # purge_corrupt_models is best-effort: a permission error
            # or missing dir shouldn't abort the test. Log and
            # continue; the doc's snapshot_download will surface real
            # download failures via its own rc.
            print(f'setup: cache purge skipped ({exc})')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA
        constraints + uv + torch probe + doc cwd chdir + cache validation.

        ``xfuser`` is NOT installed here — see ``prepare_environment``
        for why.

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
