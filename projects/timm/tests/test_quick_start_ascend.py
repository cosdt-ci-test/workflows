"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/timm/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``timm-quick-start.yml``):
    ``MONITORED_DOC_URL``         Used by the engine's monitor step (ubuntu,
                                  not the NPU runner) for doc hash checking.
                                  The test's ``pre_process`` reads the doc
                                  from the local checkout instead, because
                                  ``raw.githubusercontent.com`` is not
                                  reachable from the NPU runner's cluster.
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

    Scope: install (binary + source) + 5 core-chain ``#test`` smoke
    commands, constructing the model on ``device='npu:0'`` with
    ``pretrained=False`` (no weight download). Covers the upstream
    quickstart in order: list_models / create_model + forward /
    forward_features / resolve_data_config / create_transform. The doc
    body explicitly leaves pretrained weight loading out of scope: the
    NPU runner's cluster firewall cannot reach HuggingFace Hub / GitHub
    Releases, so every executable block avoids the network.
    """

    # git clone + uv pip install -e . + five #test smoke commands. The
    # stack is small (torch / torchvision / pyyaml / huggingface_hub /
    # safetensors, no transformers / modelscope / accelerate on the test
    # side), so 20 min covers cold cache + first-time wheel pulls for the
    # doc's own install blocks + the five ~1s smoke commands comfortably.
    DEFAULT_COMMAND_TIMEOUT = 1200

    # Monitored source is the cosdt-ci-test/workflows fork (this repo):
    # the doc lives at projects/timm/docs/Quick-start-Ascend.md and the
    # engine sets MONITORED_DOC_URL to the raw.githubusercontent.com URL
    # for the same path. Upstream timm's tutorial lives at
    # huggingface/pytorch-image-models README and is referenced from
    # inside the doc body, but it is NOT the file under test.
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
    # (subprocess.run inherits parent env by default) see it. timm's own
    # dep tree (torch / torchvision / pyyaml / huggingface_hub /
    # safetensors) shouldn't pull CUDA, but the source install goes
    # through `uv pip install -e .` which reads pyproject.toml and could
    # resolve dev/test extras that drag nvidia-* in via transitive deps.
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
    _CONSTRAINTS_FILE = '/tmp/timm_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``timm-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # pre_process: read doc from local checkout instead of fetching
    # raw.githubusercontent.com — the NPU runner sits behind a cluster
    # firewall that cannot reach it (known limitation, see
    # quick-start-template.yml). The checkout step always has the
    # latest doc for the branch under test, so there is no stale drift.
    # ----------------------------------------------------------

    def pre_process(self) -> str:
        """Read ``Quick-start-Ascend.md`` from the local checkout.

        Overrides the base ``MarkdownDocTestBase.pre_process`` which
        fetches ``MONITORED_DOC_URL`` via ``urllib`` — that endpoint
        (``raw.githubusercontent.com``) is not reachable from the NPU
        runner's cluster network. The workflow checkout already places
        the doc at ``projects/timm/docs/Quick-start-Ascend.md`` relative
        to this file, so reading it locally is both reliable and always
        in sync with the branch under test.
        """
        doc_path = (
            Path(__file__).resolve().parent.parent
            / 'docs'
            / 'Quick-start-Ascend.md'
        )
        if not doc_path.is_file():
            raise RuntimeError(
                f'doc not found in local checkout: {doc_path}'
            )
        return doc_path.read_text(encoding='utf-8')

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + torch stack probe
    # + torchvision (timm depends on torchvision)
    # (timm itself is installed by the doc's `## 安装 timm` blocks; no
    #  model download, so no modelscope cache binding)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv + torch stack probe
        + torchvision.

        The doc's ``## 安装 timm`` section is the single source of
        truth for which timm version gets installed; this class only
        handles ``torch`` / ``torch_npu`` / ``torchvision`` here (via
        the cluster cache + Huawei ascend dual-source). ``timm`` itself
        installs itself in document order via the ``#test`` machinery
        (``timm-install-binary`` / ``timm-install-source``).

        ``torchvision==0.24.0`` is the version paired with
        ``torch==2.9.0`` (see ``projects/ms-swift/constraints-npu.txt``).
        It is installed defensively with ``--no-deps`` so the doc's
        ``uv pip install timm`` finds ``torchvision`` already satisfied
        and does not try to resolve a torch-mismatched version. Importing
        torchvision's native lib only needs torch 2.9.0's CPU symbols,
        which the image's ``torch==2.9.0+cpu`` already provides.

        No modelscope cache bind-mount: the test does not download any
        model (every executable block uses ``pretrained=False``), so the
        host cache is not touched.

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

        # 2) uv: the doc's ``timm-install-binary`` / ``timm-install-source``
        # blocks call ``uv pip install`` which handles PEP 517 build deps
        # more reliably than pip. Inherit ``PIP_INDEX_URL`` +
        # ``PIP_TRUSTED_HOST`` from the yml job-level env (cluster cache
        # path + trusted-host).
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

        # 4) torchvision: timm depends on torchvision, and
        # torchvision==0.24.0 is the version paired with torch==2.9.0
        # (see projects/ms-swift/constraints-npu.txt). --no-deps avoids
        # pip re-resolving torch into a CUDA build; the image's
        # torch==2.9.0+cpu already provides the CPU symbols torchvision
        # needs at import time.
        subprocess.run(
            ['python', '-m', 'pip', 'install', '--no-deps', 'torchvision==0.24.0'],
            check=True,
        )

        # 5) timm itself is NOT installed here — the doc's
        # ``## 安装 timm`` blocks ``timm-install-binary`` /
        # ``timm-install-source`` exercise both binary and source install
        # paths against the upstream release tag injected by the
        # workflow, so a broken install block surfaces here as a fuzzy
        # mismatch against ``timm xxx`` rather than being masked by a
        # pre-installed copy.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv +
        torch stack + torchvision.

        ``timm`` is NOT installed here — see ``prepare_environment``
        for why.

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