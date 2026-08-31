"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/xllm/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``xllm-quick-start.yml``):
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

import glob
import os
import subprocess
import unittest

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SRC = os.path.join(_REPO_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 minutes: long enough for xllm inference
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + verify torch/torch_npu stack in one go.

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

        # 1) torch stack probe: when version matches the image's
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
                    '--index-url', 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple',
                    '--extra-index-url', 'https://repo.huaweicloud.com/ascend/repos/pypi',
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 2) Clone xllm source tree (with submodules) and build from source.
        # We no longer rely on a xllm-provided image; the CANN base image is
        # used and xllm's C++ extension is compiled here. xllm's setup.py does
        # NOT ship the `examples` package, so the repo must also be on
        # PYTHONPATH for `python -m examples.generate`.
        xllm_src = '/tmp/xllm-ai'
        if not os.path.isdir(xllm_src):
            upstream_ref = os.environ.get('UPSTREAM_REF') or 'main'
            print(f'setup: cloning xllm@{upstream_ref} (recursive) to {xllm_src}')
            subprocess.run(
                [
                    'git', 'clone', '--depth', '1', '--branch', upstream_ref,
                    '--recurse-submodules', '--shallow-submodules',
                    'https://github.com/xLLM-AI/xllm.git', xllm_src,
                ],
                check=True,
            )
        # Make `examples` importable in every doc-run subprocess.
        os.environ['PYTHONPATH'] = xllm_src + os.pathsep + os.environ.get('PYTHONPATH', '')

        # 3) Install build dependencies (CANN base image lacks cmake/rust/vcpkg
        # and the third_party submodules are needed by the build).
        print('setup: installing build dependencies')
        subprocess.run(
            ['bash', '-c',
             'apt-get update -qq && apt-get install -y --no-install-recommends '
             'python3-dev libssl-dev pkg-config git '
             'curl ca-certificates'],
            check=True,
        )
        subprocess.run(
            ['python', '-m', 'pip', 'install', '-q', 'cmake>=3.27', 'ninja'],
            check=True,
        )
        cmake_bin_dir = os.path.dirname(sys.executable)
        os.environ['PATH'] = cmake_bin_dir + os.pathsep + os.environ.get('PATH', '')
        subprocess.run(['cmake', '--version'], check=True)
        subprocess.run(
            ['bash', '-c',
             'command -v cargo >/dev/null 2>&1 || '
             '(curl -fsSL https://rsproxy.cn/rustup-init.sh -o /tmp/rustup-init.sh && '
             'sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable)'],
            check=True,
        )
        os.environ['PATH'] = os.pathsep.join(
            [os.path.expanduser('~/.cargo/bin'), os.environ.get('PATH', '')])
        os.environ['RUSTUP_DIST_SERVER'] = 'https://rsproxy.cn'
        os.environ['RUSTUP_UPDATE_ROOT'] = 'https://rsproxy.cn/rustup'
        subprocess.run(
            ['git', 'config', '--global',
             'url."https://gitcode.com/xLLM-AI/vcpkg.git".insteadOf',
             'https://github.com/microsoft/vcpkg.git'],
            check=True,
        )

        build_log = '/tmp/xllm-build.log'
        print(f'setup: building xllm wheel (device=npu); full log -> {build_log}')
        os.environ['SKIP_TEST'] = '1'
        with open(build_log, 'w') as lf:
            try:
                subprocess.run(
                    ['python', 'setup.py', 'bdist_wheel', '--device', 'npu'],
                    cwd=xllm_src, check=True, stdout=lf, stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError:
                print(f'!! xllm build failed; tail of {build_log}:')
                subprocess.run(['tail', '-n', '300', build_log])
                raise
        print(f'setup: build done; tail of {build_log}:')
        subprocess.run(['tail', '-n', '300', build_log])
        wheels = sorted(glob.glob(os.path.join(xllm_src, 'dist', '*.whl')))
        if not wheels:
            raise RuntimeError('xllm wheel was not produced by the build')
        subprocess.run(
            ['python', '-m', 'pip', 'install', '--force-reinstall', '--no-deps', wheels[0]],
            check=True,
        )

        # 5) Verify the freshly built xllm imports.
        subprocess.run(
            ['python', '-c', 'import xllm; print("xllm:", xllm.__version__)'],
            check=True,
        )

        # 6) Download the example model once into the mounted CI cache.
        model_dir = '/root/.cache/modelscope/Qwen2-7B-Instruct'
        if os.path.isdir(model_dir) and any(os.scandir(model_dir)):
            print(f'setup: model already cached at {model_dir}; skipping download')
        else:
            print(f'setup: downloading Qwen2-7B-Instruct to {model_dir}')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install', '-q', 'modelscope',
                ],
                check=False,
            )
            subprocess.run(
                [
                    'python', '-c',
                    "from modelscope import snapshot_download; "
                    "snapshot_download('Qwen/Qwen2-7B-Instruct', "
                    f"local_dir='{model_dir}')",
                ],
                check=True,
            )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + torch stack + xllm.

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