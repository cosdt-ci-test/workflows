import os
import re
import shutil
import subprocess
import time
import traceback
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / 'projects' / 'deepspeed' / 'docs' / 'Quick-start-Ascend.md'
WORK_DIR = REPO_ROOT / 'output' / 'deepspeed-quick-start'

_SUPPORTED_FENCES = (
    '```shell (run the fence body), '
    '```shell skip (log and continue), '
    '```text (expected output immediately after a run ```shell fence)'
)

_FENCE_RE = re.compile(r'^```([^\n]*)\n(.*?)(?:^```)', re.MULTILINE | re.DOTALL)


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fetch_doc_text() -> tuple[str, str]:
    url = os.environ.get('MONITORED_DOC_URL')
    if url:
        last_err = None
        headers = {'User-Agent': 'cosdt-ci-test/quick-start'}
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        if 'api.github.com' in url:
            headers['Accept'] = 'application/vnd.github.raw'
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read().decode('utf-8'), url
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                _log(f'fetch_doc_text: attempt {attempt+1}/2 failed for {url}')
                _log(traceback.format_exc())
                time.sleep(2)
        raise RuntimeError(
            f'failed to fetch {url} after 2 attempts: {last_err!r}')
    if DOC.exists():
        return DOC.read_text(encoding='utf-8'), f'local:{DOC}'
    raise FileNotFoundError(
        f'Quick Start doc not found at {DOC} '
        '(and MONITORED_DOC_URL is unset)')


def _reject_doctest_prompts(body: str) -> None:
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('>>>') or stripped.startswith('... '):
            raise ValueError(
                'shell fences must be pasteable bash, not doctest. '
                'A line starts with >>> or "... " after lstrip. '
                f'Supported fence styles: {_SUPPORTED_FENCES}')


def parse_blocks(doc_text: str) -> list[dict]:
    fences: list[dict] = []
    for match in _FENCE_RE.finditer(doc_text):
        tokens = match.group(1).split()
        lang = tokens[0] if tokens else ''
        markers = tokens[1:]
        body = match.group(2)
        if body.endswith('\n'):
            body = body[:-1]
        fences.append({
            'start': match.start(),
            'end': match.end(),
            'lang': lang,
            'markers': markers,
            'body': body,
        })

    steps: list[dict] = []
    for i, fence in enumerate(fences):
        if fence['lang'] != 'shell':
            continue
        _reject_doctest_prompts(fence['body'])
        kind = 'skip' if 'skip' in fence['markers'] else 'run'
        expected: list[str] = []
        if kind == 'run' and i + 1 < len(fences):
            nxt = fences[i + 1]
            between = doc_text[fence['end']:nxt['start']]
            if nxt['lang'] == 'text' and between.strip() == '':
                expected = nxt['body'].splitlines()
                while expected and expected[-1].strip() == '':
                    expected.pop()
        steps.append({
            'kind': kind,
            'cmd': fence['body'],
            'expected': expected,
        })
    return steps


_PLACEHOLDER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b\d+\.\d+\.x\b'), r'\d+\.\d+\.\d+'),
    (re.compile(r'\bxxx\b'), r'[^,\s]+?'),
]


def _sentinel(i: int) -> str:
    return f'\x01S{i}\x01'


def substitute_placeholders(expected: str) -> tuple[str, dict]:
    mapping: dict = {}
    text = expected
    counter = 0
    for pat, repl in _PLACEHOLDER_PATTERNS:
        token = _sentinel(counter)
        mapping[counter] = repl
        counter += 1
        text = pat.sub(token, text)
    return text, mapping


def _dump_merged(merged: str) -> None:
    data = merged.encode('utf-8', errors='replace')
    if len(data) <= 128_000:
        _log(f'CMD output ({len(data)}B):\n{merged.rstrip()}')
        return
    _log(f'CMD output (head 4000 of {len(data)}B):\n{merged[:4000].rstrip()}')
    _log(f'CMD output (tail 4000):\n{merged[-4000:].rstrip()}')


def run_command(cmd: str, env: dict, cwd: Path, timeout: int) -> tuple[int, str]:
    t0 = time.time()
    _log(f'CMD start (timeout={timeout}s): {cmd[:2000]}')
    proc = subprocess.run(
        ['bash', '-c', cmd],
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    merged = proc.stdout.decode('utf-8', errors='replace')
    elapsed = time.time() - t0
    _log(f'CMD done in {elapsed:.1f}s rc={proc.returncode} '
         f'(output={len(merged)}B)')
    if proc.returncode != 0:
        _dump_merged(merged)
    return proc.returncode, merged


BLOCK_TIMEOUTS = {
    'install': 900,
    'train': 300,
    'default': 300,
}


def block_kind(cmd: str) -> str:
    if 'pip install' in cmd:
        return 'install'
    if 'git clone' in cmd:
        return 'install'
    if 'cmake' in cmd:
        return 'install'
    if 'python -c' in cmd and 'Net' in cmd:
        return 'train'
    return 'default'


_SKIP_E2E = os.environ.get('NPU_READY', '') != 'true'


@unittest.skipIf(
    _SKIP_E2E,
    'end-to-end tests require NPU runner; set NPU_READY=true to run')
class TestQuickStartAscendEndToEnd(unittest.TestCase):
    doc_path: str
    steps: list[dict]

    @classmethod
    def setUpClass(cls):
        _log(f'setUpClass: fetching doc from {os.environ.get("MONITORED_DOC_URL", "<unset>")}')
        cls.doc_text, cls.doc_path = fetch_doc_text()
        _log(f'setUpClass: fetched doc ({len(cls.doc_text)} bytes)')
        cls.upstream_ref = os.environ.get('UPSTREAM_REF', '')
        if not cls.upstream_ref:
            raise unittest.SkipTest(
                'end-to-end requires UPSTREAM_REF '
                '(set by the CI workflow)')
        os.environ.setdefault('UPSTREAM_REF', cls.upstream_ref)
        _log(f'setUpClass: upstream ref={cls.upstream_ref}')
        cls.doc_text = cls.doc_text.replace(
            '<UPSTREAM_REF>', cls.upstream_ref)
        cls.steps = parse_blocks(cls.doc_text)
        _log(f'setUpClass: parsed {len(cls.steps)} steps')
        if not cls.steps:
            raise unittest.SkipTest(
                f'No shell code blocks found in {cls.doc_path}')

    def test_runs_quick_start(self):
        env = os.environ.copy()
        env['UPSTREAM_REF'] = self.upstream_ref

        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
        WORK_DIR.mkdir(parents=True)

        captures: dict = {}
        _log(f'test_runs_quick_start: starting {len(self.steps)} steps in {WORK_DIR}')
        for i, step in enumerate(self.steps):
            if step['kind'] == 'skip':
                _log(f'[step {i}/{len(self.steps)-1}] skip')
                self._log_block('skipped cmd', step['cmd'].splitlines(), cap=10)
                continue
            kind = block_kind(step['cmd'])
            timeout = BLOCK_TIMEOUTS.get(kind, BLOCK_TIMEOUTS['default'])
            cmd = step['cmd']
            for key, value in captures.items():
                cmd = cmd.replace(f'<{key}>', value)
            _log(f'[step {i}/{len(self.steps)-1}] kind={kind} timeout={timeout}s')
            rc, out = run_command(cmd, env, WORK_DIR, timeout)
            self.assertEqual(
                rc, 0,
                f'step #{i} ({kind}) command failed (rc={rc}):\n'
                f'  cmd: {cmd!r}\n'
                f'  output:\n{out}')
            expected = step['expected']
            if not expected:
                continue
            self._compare_lines(
                i, 0, kind, expected, out.splitlines(), captures,
            )
        _log('test_runs_quick_start: all steps done')

    def _compare_lines(self, block_idx, step_idx, kind, expected, actual, captures):
        expected_text = '\n'.join(expected)
        actual_text = '\n'.join(actual)

        text, mapping = substitute_placeholders(expected_text)
        escaped = re.escape(text)
        for i, frag in mapping.items():
            escaped = escaped.replace(_sentinel(i), frag)
        escaped = escaped.replace(r'\.\.\.', r'.*?')
        pattern = re.compile(escaped, re.DOTALL)
        m = pattern.search(actual_text)

        if m:
            for k, v in m.groupdict().items():
                if v is not None:
                    captures[k] = v

        if m:
            _log(f'[Test block {block_idx}] step {step_idx} ({kind}): OK '
                 f'({len(expected)} expected, {len(actual)} actual)')
            self._log_block('actual', actual, cap=30)
            return

        _log(f'[Test block {block_idx}] step {step_idx} ({kind}): MISMATCH '
             f'({len(expected)} expected, {len(actual)} actual)')
        self._log_block('expected', expected)
        self._log_block('actual', actual)
        self.fail(
            f'block #{block_idx} step #{step_idx} ({kind}) output '
            'mismatch; see summary above')

    @staticmethod
    def _log_block(label: str, lines, cap: int = 30) -> None:
        _log(f'  --- {label} (head + tail if huge) ---')
        if len(lines) <= cap:
            for i, ln in enumerate(lines, 1):
                _log(f'  {i:>3}. {ln}')
            return
        half = cap // 2
        head = lines[:half]
        tail = lines[-half:]
        for i, ln in enumerate(head, 1):
            _log(f'  {i:>3}. {ln}')
        elided = len(lines) - 2 * half
        _log(f'  ... [{elided} line(s) elided] ...')
        tail_start = len(lines) - half + 1
        for offset, ln in enumerate(tail):
            _log(f'  {tail_start + offset:>3}. {ln}')


class TestParseBlocks(unittest.TestCase):
    def test_parses_local_doc(self):
        text = DOC.read_text(encoding='utf-8')
        steps = parse_blocks(text)
        self.assertGreaterEqual(len(steps), 4)

        kinds = [s['kind'] for s in steps]
        self.assertIn('run', kinds)

        cmds = [s['cmd'] for s in steps]
        self.assertTrue(any('set_env.sh' in c for c in cmds))
        self.assertTrue(any('npu-smi info' in c.strip() for c in cmds))
        self.assertTrue(any('git clone' in c for c in cmds))
        self.assertTrue(any('pip install -e' in c for c in cmds))
        self.assertTrue(any('get_accelerator' in c for c in cmds))
        self.assertTrue(any('HelloDeepSpeed' in c or 'train_bert_ds' in c for c in cmds))

    def test_rejects_doctest_prompts_in_shell(self):
        fake = (
            '```shell\n'
            '>>> echo hi\n'
            '... --flag\n'
            '```\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_blocks(fake)
        self.assertIn('```shell', str(ctx.exception))
        self.assertIn('```shell skip', str(ctx.exception))
        self.assertIn('```text', str(ctx.exception))

    def test_prose_between_fences_does_not_attach(self):
        fake = (
            '```shell\n'
            'echo only-exit-code\n'
            '```\n'
            '\n'
            'A prose sentence sits between the command and the output.\n'
            '\n'
            '```text\n'
            'must_not_attach\n'
            '```\n'
        )
        steps = parse_blocks(fake)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['kind'], 'run')
        self.assertEqual(steps[0]['cmd'], 'echo only-exit-code')
        self.assertEqual(steps[0]['expected'], [])


if __name__ == '__main__':
    unittest.main()