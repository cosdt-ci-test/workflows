"""End-to-end test for the FastChat Ascend quick-start document."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


_PROJECT_DIR = Path(__file__).resolve().parent.parent
_WORK_DIR = Path('/tmp/fastchat-quick-start')
_SERVICE_DIR = _WORK_DIR / '.fastchat'
_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
_OWNED_SERVICES = (
    ('api.pid', 'fastchat.serve.openai_api_server'),
    ('worker.pid', 'fastchat.serve.model_worker'),
    ('controller.pid', 'fastchat.serve.controller'),
)
_FSCHAT_PIN_RE = re.compile(
    r'fschat\[model_worker\]==(?P<version>[0-9][0-9A-Za-z.!+_-]*)'
)
_RELEASE_VERSION_RE = re.compile(r'[0-9]+(?:\.[0-9]+)+(?:[0-9A-Za-z.!+_-]*)?')


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() == 'true')


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


def _documented_fschat_version(text: str) -> str:
    versions = set(_FSCHAT_PIN_RE.findall(text))
    if len(versions) != 1:
        raise RuntimeError(
            'quick-start must contain exactly one fschat[model_worker] version pin; '
            f'found {sorted(versions)}'
        )
    return versions.pop()


def _release_version(upstream_ref: str) -> str:
    ref = upstream_ref.strip()
    version = ref[1:] if ref.startswith('v') else ref
    if not _RELEASE_VERSION_RE.fullmatch(version):
        raise RuntimeError(
            f'UPSTREAM_REF {upstream_ref!r} is not a release tag that can be '
            'mapped to a PyPI version'
        )
    return version


def _assert_version_alignment(text: str, upstream_ref: str) -> None:
    documented = _documented_fschat_version(text)
    monitored = _release_version(upstream_ref)
    if documented != monitored:
        raise RuntimeError(
            'FastChat version mismatch: '
            f'UPSTREAM_REF={upstream_ref!r} resolves to {monitored!r}, '
            f'but the quick-start installs fschat=={documented}'
        )


def _merge_sourced_env(script: str) -> None:
    """Source an environment script and merge its exported variables."""

    merged = subprocess.run(
        [
            'bash',
            '-c',
            f'source {shlex.quote(script)} >/dev/null 2>&1; env -0',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for entry in merged.stdout.split('\0'):
        if not entry or '=' not in entry:
            continue
        key, _, value = entry.partition('=')
        os.environ[key] = value


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 1 else None


def _process_cmdline(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(
            'utf-8', errors='replace'
        )
    except OSError:
        return ''


def _stop_owned_process(pid_file: Path, expected_module: str) -> None:
    """Stop only the process recorded by the document and matching its module."""

    pid = _read_pid(pid_file)
    if pid is None:
        pid_file.unlink(missing_ok=True)
        return

    cmdline = _process_cmdline(pid)
    if not cmdline:
        pid_file.unlink(missing_ok=True)
        return
    if expected_module not in cmdline:
        print(
            f'cleanup: refusing to stop pid {pid}; expected {expected_module!r}, '
            f'got {cmdline!r}'
        )
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return

    for _ in range(50):
        if not _process_cmdline(pid):
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)


def _cleanup_services() -> None:
    for pid_name, expected_module in _OWNED_SERVICES:
        _stop_owned_process(_SERVICE_DIR / pid_name, expected_module)


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """Run the documented CLI and OpenAI-compatible API flows on one NPU."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
        'Address already in use',
        'No available worker',
    )

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
    _CONSTRAINTS_FILE = '/tmp/fastchat_npu_constraints.txt'
    _CLUSTER_INDEX = (
        'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    )
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    def pre_process(self) -> str:
        text = super().pre_process()
        upstream_ref = os.environ.get('UPSTREAM_REF', '')
        if not upstream_ref:
            raise RuntimeError('UPSTREAM_REF is required for FastChat version alignment')
        _assert_version_alignment(text, upstream_ref)
        return text

    @classmethod
    def prepare_environment(cls) -> None:
        if not os.path.isfile(_CANN_SET_ENV):
            raise RuntimeError(f'required CANN environment script missing: {_CANN_SET_ENV}')
        _merge_sourced_env(_CANN_SET_ENV)

        Path(cls._CONSTRAINTS_FILE).write_text(
            '\n'.join(cls._CUDA_CONSTRAINTS) + '\n', encoding='utf-8'
        )
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        probe = subprocess.run(
            [
                sys.executable,
                '-c',
                'import torch, torch_npu; '
                "raise SystemExit(0 if torch.__version__.startswith('2.9.0') "
                "and torch_npu.__version__.startswith('2.9.0') else 1)",
            ],
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--index-url',
                    cls._CLUSTER_INDEX,
                    '--extra-index-url',
                    cls._ASCEND_EXTRA,
                    'torch==2.9.0',
                    'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

        _cleanup_services()
        if _WORK_DIR.exists():
            shutil.rmtree(_WORK_DIR)
        _WORK_DIR.mkdir(parents=True)
        os.chdir(_WORK_DIR)

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    def post_process(self) -> None:
        _cleanup_services()
        os.chdir(_PROJECT_DIR)
        if _WORK_DIR.exists():
            shutil.rmtree(_WORK_DIR, ignore_errors=True)

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


class TestVersionAlignment(unittest.TestCase):
    def test_matching_release_and_document_pin(self) -> None:
        _assert_version_alignment(
            'python -m pip install "fschat[model_worker]==0.2.36"',
            'v0.2.36',
        )

    def test_mismatch_fails_before_document_execution(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
            _assert_version_alignment(
                'python -m pip install "fschat[model_worker]==0.2.36"',
                'v0.2.37',
            )

    def test_non_release_ref_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'not a release tag'):
            _assert_version_alignment(
                'python -m pip install "fschat[model_worker]==0.2.36"',
                'main',
            )


if __name__ == '__main__':
    unittest.main()
