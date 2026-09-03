"""Quick-start-Ascend documentation test for llm-d.

Document under test: ``projects/llm-d/docs/Quick-start-Ascend.md``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.modelscope_cache import (
    ensure_safetensors,
    purge_corrupt_models,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


# Stop only process groups named in these files, and only if /proc/pid
# still looks like that process. Never pkill.
# The runner may host other inference jobs.
_OWNED_PROCESSES = (
    (Path('/root/llm-d/envoy.pid'), 'envoy'),
    (Path('/root/llm-d/epp.pid'), 'epp'),
    (Path('/root/llm-d/vllm.pid'), 'vllm'),
)


def _cmdline_of(pid: int) -> str:
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return ''
    return raw.replace(b'\x00', b' ').decode('utf-8', errors='replace')


def _stop_pid_file(pid_file: Path, needle: str, wait_s: float = 15.0) -> None:
    if not pid_file.is_file():
        return
    raw = pid_file.read_text(encoding='utf-8').strip()
    if not raw.isdigit():
        pid_file.unlink(missing_ok=True)
        return
    pid = int(raw)
    if needle not in _cmdline_of(pid):
        pid_file.unlink(missing_ok=True)
        return

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return

    _signal_group(signal.SIGTERM)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.2)
    _signal_group(signal.SIGKILL)
    pid_file.unlink(missing_ok=True)


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _NNAL_SET_ENV = '/usr/local/Ascend/nnal/atb/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        missing = [
            path for path in (cls._CANN_SET_ENV, cls._NNAL_SET_ENV)
            if not os.path.isfile(path)
        ]
        if missing:
            raise RuntimeError(
                'required env scripts missing: ' + ', '.join(missing)
            )

        # Overwrite, do not setdefault. The container already has
        # LD_LIBRARY_PATH, and setdefault would keep the pre-NNAL value.
        merged = subprocess.run(
            [
                'bash', '-c',
                f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; '
                f'source {cls._NNAL_SET_ENV} >/dev/null 2>&1; env',
            ],
            capture_output=True, text=True, check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        print('setup: sourced CANN and NNAL env')

        # Header check uses framework='numpy' and should not import torch.
        # Keep autoload off around the purge so an unexpected torch import
        # cannot dlopen torch_npu in this already-running process.
        # Child bash blocks inherit the merged env and load the plugin.
        prev_autoload = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        try:
            ensure_safetensors()
            try:
                purge_corrupt_models(resolve_modelscope_cache())
            except ModuleNotFoundError as exc:
                # safetensors.safe_open(..., framework='numpy') imports numpy.
                # The install #test has not run yet. A warm ModelScope
                # volume would otherwise crash setUpClass.
                print(f'setup: skip model cache purge ({exc})')
        finally:
            if prev_autoload is None:
                os.environ.pop('TORCH_DEVICE_BACKEND_AUTOLOAD', None)
            else:
                os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = prev_autoload

    def post_process(self) -> None:
        for pid_file, needle in _OWNED_PROCESSES:
            _stop_pid_file(pid_file, needle)

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()
