"""Guard projects/roll/docs/Quick-start-Ascend.md on an NPU runner."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _TENSORBOARD_DIR = Path(
        'output/tensorboard/roll-quick-start-npu'
    )
    _STACK_PROBE_STAGES = {
        'verify-vllm-triton': 'after vLLM-Ascend install',
        'verify-example-deps': 'after example dependency install',
    }

    def _log_torch_stack(self, stage, env, cwd) -> None:
        """Inspect each installed stack in a fresh process before training."""
        probe = """
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
import sys

print('python:', sys.executable)
for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend'):
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        print(f'{name}: not installed')
        continue
    print(f'{name}: {dist.version} at {dist.locate_file("")}')
    if name in ('vllm', 'vllm-ascend'):
        requirements = [req for req in dist.requires or [] if req.lower().startswith('torch')]
        print(f'{name} torch dependencies: {", ".join(requirements)}')

for name in ('torch', 'torch_npu'):
    spec = find_spec(name)
    print(f'{name} import path: {spec.origin if spec else "not found"}')

import torch
print('loaded torch:', torch.__version__, torch.__file__)
import torch_npu
print('loaded torch_npu:', torch_npu.__version__, torch_npu.__file__)
"""
        self.log(f'PyTorch stack {stage}')
        try:
            result = subprocess.run(
                [sys.executable, '-u', '-c', probe], env=env, cwd=cwd,
                capture_output=True, encoding='utf-8', errors='replace',
                check=False, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log(f'PyTorch stack probe could not finish: {exc}')
            return
        if result.stdout:
            self.log_block('installed PyTorch stack', result.stdout.splitlines())
        if result.stderr:
            self.log_block('PyTorch import stderr', result.stderr.splitlines())
        self.log(f'PyTorch stack probe exit code: {result.returncode}')

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        super()._run_one(cmd, results, env, cwd, timeout, idx)
        stage = self._STACK_PROBE_STAGES.get(getattr(cmd, 'id', None))
        if stage is not None:
            self._log_torch_stack(stage, env, cwd)

    def _verify_tensorboard_events(self) -> None:
        """Require a nonempty TensorBoard event file after training."""
        events = list(self._TENSORBOARD_DIR.glob('*/events.out.tfevents.*'))
        if not events:
            raise AssertionError(
                'no tensorboard event file found under '
                f'{self._TENSORBOARD_DIR}'
            )
        empty = [p for p in events if p.stat().st_size == 0]
        if empty:
            raise AssertionError(
                'tensorboard event file is empty: '
                + ', '.join(str(p) for p in empty)
            )
        self.log(
            '[Step] verified tensorboard events '
            f'({len(events)} file(s), first={events[0].stat().st_size}B): '
            f'{self._TENSORBOARD_DIR}'
        )

    def pre_process(self) -> str:
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

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env merge, card pin, PYTHONNOUSERSITE, PATH prefix, and
        model-cache sanity. Dependency installs live in the doc itself
        (that is the user path); only CI-side hygiene happens here."""
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(
                f'CANN environment script is missing: {cls._CANN_SET_ENV}'
            )

        merged = subprocess.run(
            ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        venv_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{venv_bin}:{path_dirs}:{os.environ.get("PATH", "")}'
        print('setup: sourced CANN environment')

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

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
        self._verify_tensorboard_events()


if __name__ == '__main__':
    unittest.main()
