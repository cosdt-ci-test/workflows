"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/lightx2v/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``lightx2v-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``               Consumed by the doc body: the hidden
                                   ``store="upstream_ref"`` setup block
                                   captures it and the source-install
                                   block ``load``s it into
                                   ``git clone --branch <ref>``.
                                   LightX2V has no stable release tag
                                   (zero releases, zero tags, rolling
                                   main), so the trigger pins
                                   ``fixed_ref: main`` and this is
                                   always ``main``.
    ``NPU_READY=true``             Required, otherwise the class is skipped.
                                   End-to-end tests only run on the NPU runner:
                                   local dev machines / normal ubuntu runners
                                   have no ``/dev/davinci*`` device, and the
                                   hard run would fail on ``import torch_npu``.
    ``PROJECT_ROOT``               CI override (set in ``prepare_environment``):
                                   ``/root/lightx2v-test``. Doc body uses
                                   ``${PROJECT_ROOT:-/home/coder/work/lightx2v-test}``,
                                   so on CI it picks up the CI path, on
                                   coder it defaults to the persistent volume.
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

    Scope: mindiesd image stack (torch/torch_npu preinstalled, reused via
    a version-agnostic probe) + modelscope + LightX2V source install
    (with minimal unconditional stub packages for cv2 / decord /
    torchaudio — audited absent from the image, and the smoke path never
    calls them — plus the real triton wheel, since an empty triton stub
    drives torch._inductor into real triton code paths it cannot
    survive) + include-filtered
    ModelScope weight pulls (12 GB Wan2.2-I2V-A14B base T5/VAE/
    tokenizer + 4.8 GB CLIP vision encoder from Wan2.1-I2V-14B-480P
    + 30 GB I2V int8 distill split dirs) + I2V single-card smoke
    (4-step distilled 14B int8 at 480P on a 910B4). The doc's
    I2V config is adapted single-card 32 GB (720P -> 480P, t5/vae
    cpu_offload on, `parallel` and `video_frame_interpolation`
    dropped), so only one card is touched.

    The Wan2.2 T2V distill route and the Qwen-Image-Edit I2I route
    are documented as pointer-only sections (no #test blocks): the
    T2V distill weights are HF-only (not on ModelScope), and the
    Qwen Lightning route needs an extra ~20 GB FP8 ckpt plus
    undetermined text-encoder components — both are follow-up work
    after the I2V path is green.
    """

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 min: cold download of 28 GB base alone can hit ~1h on slow cluster egress; per-block 4-step gen ~1-3 min
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
        'RuntimeError: Failed to load the backend extension: torch_npu',  # torch_npu loaded outside CANN env (CI runner may differ)
    )

    # Process-level CUDA exclusion list. Originally written inside the
    # workflow step as a child-process env passed through to pip / uv /
    # lightx2v's own wheel resolver. Moved to the test layer: write to /tmp
    # and export; subprocesses (subprocess.run inherits parent env by
    # default) see it the same way.
    #
    # LightX2V pulls in a few CUDA-typed transitive deps via the
    # sglang wheel (the Wan2.2-Distill-Models MS repo also lists
    # sglang==0.5.14 in some uploads); same caveat as specforge:
    # `Requires-Dist: cuda-python` with `<0` exclude marker isn't
    # always honored, and uv's resolver picks CUDA wheels that fail
    # on aarch64 NPU. Pre-exclude the whole CUDA toolchain so the
    # resolver falls back to the CANN-typed / pure-Python alternatives
    # immediately.
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
    _CONSTRAINTS_FILE = '/tmp/lightx2v_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``lightx2v-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # PROJECT_ROOT override for CI: the doc body defaults to
    # ``/home/coder/work/lightx2v-test`` (coder persistent volume); on
    # the CI runner the path doesn't exist and the user is root with
    # ``$HOME=/root``. Set /root/lightx2v-test — the workflow yml
    # bind-mounts /data/ci-cache/lightx2v-models onto the `models/`
    # subdir so downloaded weights persist across runs.
    _PROJECT_ROOT = '/root/lightx2v-test'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv +
    # torch stack probe + PROJECT_ROOT override + cache validation
    # (lightx2v itself + its aarch64 stub packages + ModelScope
    # weight pulls all live in the doc body)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv +
        probe torch stack + set PROJECT_ROOT + validate modelscope cache.

        ``lightx2v`` itself is NOT installed here — the doc's
        ``## 安装 LightX2V`` block exercises the source install path
        (``git clone`` + ``uv pip install --no-deps -v .`` +
        aarch64 stub package setup), so a broken install surfaces as a
        fuzzy mismatch against ``lightx2v version: xxx`` rather than
        being masked by a pre-installed copy.

        ``PROJECT_ROOT`` is set here (not in the workflow yml) so the
        value is process-level visible to every doc command (the
        MarkdownDocTestBase passes ``env=os.environ.copy()`` to each
        subprocess). The doc's ``export PROJECT_ROOT=${PROJECT_ROOT:-...}``
        pick-up then resolves to the CI path.

        No modelscope cache bind is set here either — the workflow
        yml declares the bind at ``container_options`` time, which
        makes it visible to ``resolve_modelscope_cache()`` via the
        standard ``/root/.cache/modelscope`` path that modelscope
        writes to by default. The workflow yml also bind-mounts
        ``/data/ci-cache/lightx2v-models`` onto
        ``$PROJECT_ROOT/models`` so the doc's ``--local_dir
        "$PROJECT_ROOT/models/..."`` writes also persist.
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

        # 2) uv: the doc's ``lightx2v-install-source`` block calls
        # ``uv pip install --no-deps -v .`` which handles PEP 517 build
        # deps more reliably than pip. Inherit ``PIP_INDEX_URL`` +
        # ``PIP_TRUSTED_HOST`` from the yml job-level env (cluster
        # cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 2.5) triton: NOT installed here — the doc's aarch64 stub
        # section installs the *real* wheel (`uv pip install
        # 'triton==3.5.*'`). Two dead ends ruled this out empirically:
        # a `pip install triton<3.0` pin fails the resolver (no 2.x
        # aarch64 wheel on the cluster index), and an empty import-time
        # stub makes ``import triton`` succeed *too well* — torch then
        # walks its real triton code paths (_inductor triton_heuristics
        # subclasses triton.Config, reads GPUTarget/knobs) and dies
        # layer by layer (run#1 tl.math AttributeError, run#2
        # inspect TypeError, run#3 module() arity TypeError). triton
        # 3.5.x ships cp312 aarch64 manylinux wheels with zero runtime
        # deps (the CUDA constraint list never conflicts), and matches
        # torch 2.9's official triton line.

        # 3) torch stack probe + install: when the image's pre-installed
        # stack imports and sees the NPU, reuse it (mindiesd image:
        # vllm-omni base carries torch 2.9.0 + torch_npu, audited from
        # the public Dockerfile). Version-agnostic on purpose: the doc
        # no longer installs torch, so any version the image ships is
        # the version we test against.
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
            # Cold fallback: the image's torch stack failed to import or
            # see the NPU. The doc body no longer reinstalls torch —
            # its check-torch block will surface the same error; this
            # branch just records that the probe didn't match (useful
            # diagnostic in CI logs).
            print('setup: torch stack probe failed, doc check-torch will surface the error')

        # 4) PROJECT_ROOT: CI runs as root on an ephemeral container.
        # The doc body's `mkdir -p "$PROJECT_ROOT"` writes to
        # /root/lightx2v-test here; the workflow yml bind-mounts the
        # persistent volume onto the `models/` subdir so model
        # downloads survive across runs. Pre-create the parent here
        # so the bind-mount is visible to the first doc command (a
        # bind-mount onto a missing target dir fails on some
        # kernels).
        os.environ['PROJECT_ROOT'] = cls._PROJECT_ROOT
        try:
            os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        except OSError as exc:
            print(f'setup: PROJECT_ROOT mkdir failed: {exc}')

        # 4.5) ASCEND_RT_VISIBLE_DEVICES=0: the cluster's NPU runner
        # label is `linux-aarch64-a2-2` (2 cards); the cluster
        # device-plugin still passes both /dev/davinci* into the
        # container. The doc's I2V smoke is adapted to single-card
        # 32 GB so we only want to see one device — the doc's
        # `## 前置安装` `check-torch` step hardcodes `count: 1` in its
        # `#test-result` block (the doc itself sets
        # ASCEND_RT_VISIBLE_DEVICES=0 only inside the smoke step,
        # not at the check-torch step, so the doc would otherwise
        # see count=2 at step 4 and fail). Set the env at process
        # level here so the value is inherited by every doc
        # subprocess (MarkdownDocTestBase passes ``env=os.environ.copy()``
        # to each subprocess).
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        # 5) safetensors + cache validation: persistent host-side bind
        # mounts can hold truncated safetensors from interrupted
        # downloads. Walk every shard under the modelscope cache
        # ($PROJECT_ROOT/models, which is the bind-mount target) and
        # purge on failure; modelscope will re-download cleanly on
        # next access.
        # ensure_safetensors pulls in safetensors (transitive of
        # torch); install defensively in case the CANN base image
        # rolls forward.
        ensure_safetensors()
        try:
            purge_corrupt_models(resolve_modelscope_cache())
            # Also walk the bind-mounted $PROJECT_ROOT/models since
            # the doc's --local_dir outputs land there, not in the
            # default modelscope cache.
            from pathlib import Path
            proj_models = Path(cls._PROJECT_ROOT) / 'models'
            if proj_models.is_dir():
                purge_corrupt_models(proj_models)
        except Exception as exc:
            # purge_corrupt_models is best-effort: a permission error
            # or missing dir shouldn't abort the test. Log and
            # continue; the doc's `modelscope download --local_dir`
            # will surface real download failures via its own rc.
            print(f'setup: cache purge skipped ({exc})')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA
        constraints + uv + torch probe + PROJECT_ROOT + cache validation.

        ``lightx2v`` is NOT installed here — see ``prepare_environment``
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