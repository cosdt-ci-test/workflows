"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/speculators/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``speculators-quick-start.yml``):
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
    ``#test-result``."""

    # 30 min per command: long enough for the ~17 GB combined
    # z-lab/Qwen3-8B-DFlash-b16 (~1 GB) + Qwen/Qwen3-8B verifier
    # (~16 GB) ModelScope downloads on first run + speculators convert
    # pipeline (weight remap + CPU from_pretrained + NaN check); short
    # enough to fail fast on hangs. Smaller than the originally considered
    # EAGLE-3 + Llama-3.1 pair (~32 GB) which we dropped because the
    # verifier is gated on HF Hub and not on ModelScope.
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Originally written inside the
    # workflow step as a child-process env passed through to pip / uv /
    # speculators' own wheel resolver. Moved to the test layer: write to
    # /tmp and export; subprocesses (subprocess.run inherits parent env
    # by default) see it the same way.
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
    _CONSTRAINTS_FILE = '/tmp/speculators_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    # Both URLs are exposed as env vars (``PIP_INDEX_URL`` /
    # ``UV_INDEX_URL`` + ``UV_EXTRA_INDEX_URL``) by the engine template
    # at .github/workflows/quick-start-template.yml, so individual
    # ``#test`` blocks don't need to repeat them on the command line —
    # bare ``pip install`` / ``uv pip install`` already routes through
    # the cluster cache and falls back to huawei ascend for uv.

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (CI_IMAGE).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + torch stack probe
    # + safetensors (transformers / speculators are installed by the doc's
    #  `### 前置安装` / `## 安装 Speculators` blocks; HF cache is left at
    #  the container default so the workflow's bind mount applies)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv + safetensors
        + purge stale modelscope cache shards.

        The doc's ``### 前置安装`` and ``## 安装 Speculators`` sections are
        the single source of truth for which packages + versions get
        installed — ``torch`` + ``torch_npu`` (the doc's
        ``install-torch`` #test-setup), ``modelscope`` (via ``install-deps``),
        and ``speculators`` (via ``speculators-install-binary`` /
        ``speculators-install-source``) all install themselves in document
        order via the ``#test`` machinery. This class no longer touches
        the torch stack — putting it here would mean a torch mismatch
        (bare CANN image vs doc's pinned 2.10.0+cpu) shows up in
        ``prepare_environment`` log rather than in the doc's
        ``check-npu-runtime`` #test where it actually belongs.

        The defensive ``safetensors`` install (speculators depends on
        it for weight IO and may not be on the CANN base image) and the
        modelscope cache purge are still done here.

        ModelScope cache (``$MODELSCOPE_CACHE`` or its default
        ``~/.cache/modelscope``) is left at its default — the workflow's
        bind mount targets ``/root/.cache/modelscope`` directly so the
        cache survives across runs. Same pattern as peft / diffusers.

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

        # 2) uv: the doc's ``speculators-install-source`` block calls
        # ``uv pip install -e .`` which handles PEP 517 build deps more
        # reliably than pip. Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack: NOT installed here. Lives in the doc's
        # ``### 前置安装`` #test-setup ``install-torch`` block — putting
        # the install in this hook would mean a torch version mismatch
        # shows up in prepare_environment log rather than in the doc's
        # ``check-npu-runtime`` #test where the mismatch actually matters.
        #
        # 4) safetensors: speculators reads model weights via
        # safetensors. It's a base dep of speculators (its pyproject
        # lists ``safetensors`` as required), so it ships in once
        # ``speculators-install-*`` runs in the doc; this defensive
        # install catches the case where CANN base image lacks it
        # before any speculators code touches weight files.
        ensure_safetensors()

        # 5) Cache validation: persistent host-side bind mount can hold
        # truncated safetensors from interrupted runs. Walk every shard
        # under each model dir and purge it on failure; modelscope will
        # re-download cleanly on next access. Implementation lives in
        # workflows.modelscope_cache; see that module's docstring for
        # the full rationale.
        purge_corrupt_models(resolve_modelscope_cache())

        # 6) modelscope / speculators / torch / torch_npu are installed
        # by the doc's ``### 前置安装`` block ``install-deps`` (it carries
        # the install + verify pair itself). This class no longer
        # installs them — keeping install here on top would just be
        # redundant ``uv pip install``-idempotent noise.
        #
        # ``transformers`` is NOT installed here either: it's a base
        # dep of speculators (speculators' pyproject pins
        # ``transformers>=4.56.1,<5.15.0``) and gets pulled in
        # transitively by ``speculators-install-binary`` /
        # ``speculators-install-source`` in the doc.
        #
        # ``speculators`` is also NOT installed here: it's the subject
        # of the test and gets installed by the doc's ``## 安装
        # Speculators`` blocks, which exercise both binary and source
        # install paths.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints
        + uv + safetensors + modelscope cache purge.

        ``torch`` / ``torch_npu`` / ``modelscope`` / ``speculators`` are
        NOT installed here: the doc's own labeled blocks install them in
        document order, so a broken install block fails loudly instead of
        being masked by a pre-installed copy.

        ModelScope cache is left at its container default so the
        workflow's bind mount at ``/root/.cache/modelscope`` applies.
        See ``prepare_environment`` for the full rationale.

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