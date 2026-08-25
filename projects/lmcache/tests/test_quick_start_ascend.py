"""Quick-start-Ascend documentation test for LMCache-Ascend.

Document under test: ``projects/lmcache/docs/Quick-start-Ascend.md``.
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
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 5400
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'HostRegisterError',
        'ERR99999',
    )
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _ATB_SET_ENV = '/usr/local/Ascend/nnal/atb/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        source_cmd = f'source {cls._CANN_SET_ENV} >/dev/null 2>&1'
        if os.path.isfile(cls._ATB_SET_ENV):
            # ATB set_env.sh reads unbound ZSH_VERSION; nounset would abort.
            source_cmd += (
                f'; set +u; source {cls._ATB_SET_ENV} >/dev/null 2>&1'
            )
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'{source_cmd}; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN toolkit and ATB env')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        ensure_safetensors()
        purge_corrupt_models(resolve_modelscope_cache())

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
