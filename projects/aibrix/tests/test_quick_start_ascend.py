"""Quick-start-Ascend documentation test for AIBrix.

Document under test: ``projects/aibrix/docs/Quick-start-Ascend.md``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.modelscope_cache import (
    ensure_safetensors,
    purge_corrupt_models,
    resolve_modelscope_cache,
)

_WORK_DIR = Path('.aibrix-quick-start')
_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
_ATB_SET_ENV = '/usr/local/Ascend/nnal/atb/latest/atb/set_env.sh'


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


def _read_pids(path: Path) -> list[int]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return []
    pids: list[int] = []
    for token in text.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid > 1:
            pids.append(pid)
    return pids


def _stop_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def cleanup_aibrix_workdir(root: Path) -> None:
    for pid in _read_pids(root / 'vllm.pid'):
        _stop_pid(pid)
    local_pids = root / 'aibrix' / 'deployment' / 'local' / '.pids'
    for pid in _read_pids(local_pids):
        _stop_pid(pid)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def _merge_sourced_env(*scripts: str) -> None:
    sourced = ' && '.join(f'source {shlex.quote(script)}' for script in scripts)
    merged = subprocess.run(
        ['bash', '-c', f'set +u; {sourced} >/dev/null; env'],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in merged.stdout.splitlines():
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key, value)


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1200
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'RuntimeError',
        'ERR99999',
        'Invalid device ID',
        'Address already in use',
        'libatb.so',
        'no healthy upstream',
    )

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        missing = [p for p in (_CANN_SET_ENV, _ATB_SET_ENV) if not os.path.isfile(p)]
        if missing:
            raise RuntimeError(
                'required Ascend env scripts missing: ' + ', '.join(missing)
            )
        _merge_sourced_env(_CANN_SET_ENV, _ATB_SET_ENV)
        print('setup: sourced CANN and ATB env')

        ensure_safetensors()
        purge_corrupt_models(resolve_modelscope_cache())

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    def post_process(self) -> None:
        cleanup_aibrix_workdir(_WORK_DIR)

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


class TestPostProcessCleanup(unittest.TestCase):
    def test_stops_recorded_pid_when_infer_did_not_run(self) -> None:
        root = Path(tempfile.mkdtemp(prefix='aibrix-qs-'))
        proc = subprocess.Popen(['sleep', '120'])
        try:
            (root / 'vllm.pid').write_text(str(proc.pid), encoding='utf-8')
            cleanup_aibrix_workdir(root)
            proc.wait(timeout=5)
            self.assertIsNotNone(proc.returncode)
            self.assertFalse(root.exists())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
