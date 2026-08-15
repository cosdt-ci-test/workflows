# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests that walk through the Ascend Quick Start documentation.

The Quick Start is structured as a sequence of ``shell`` fenced code
blocks. Each block contains one or more ``>>>`` REPL commands followed
by their expected output. This test:

  1. Parses every ``shell`` block in
     ``docs/source/GetStarted/Quick-start-Ascend.md`` (or the English
     twin) into ``(command, expected_output_lines)`` pairs.
  2. Executes each command in a fresh subshell.
  3. Compares the actual output against the expected output line by
     line, using regex matching so that placeholders in the doc can
     stand in for dynamic values (PIDs, version build numbers, etc.).

If the Quick Start doc is changed in a way that breaks any block, this
test fails.

Placeholder syntax
------------------
* ``...``  - match any number of characters on this line (wildcard).
* ``<pid>`` - match a number and capture it; the captured value is
              substituted into later commands that contain ``<pid>``.
* ``xxx``   - match any non-whitespace run.
* ``2.9.0.x`` / ``2.9.0.postX`` / ``3.11.x`` - version placeholders.
* ``x.y.z`` - match ``d.d.d``.
* Plain lines that are not ``...`` are matched exactly.

The test runner is expected to be an NPU CI container
(``linux-aarch64-a2-1`` in this repo's ``citest_npu.yaml``) where
CANN, torch, torch_npu and ms-swift are already available.
"""

import os
import re
import subprocess
import time
import traceback
import unittest
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / 'projects' / 'ms-swift' / 'docs' / 'Quick-start-Ascend.md'


def _log(msg: str) -> None:
    """Print a timestamped, flushed log line so CI output shows progress
    even when subprocess buffers haven't been flushed yet."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

def fetch_doc_text() -> tuple[str, str]:
    """Fetch the Quick Start doc from MONITORED_DOC_URL.

    The URL must match what the monitor step hashed, so the doc under
    test is exactly the doc that triggered this run. Failure to fetch
    raises - the test fails (this is a CI failure, not a silent skip).

    Returns (text, url). Falls back to the local checkout copy when
    MONITORED_DOC_URL is not set, so the test can still be exercised
    by hand without the workflow.
    """
    url = os.environ.get('MONITORED_DOC_URL')
    if url:
        # urllib's default has no timeout - on flaky networks the
        # call can hang indefinitely. Cap each attempt at 30s and
        # retry once before giving up.
        last_err = None
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'cosdt-ci-test/quick-start'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read().decode('utf-8'), url
            except Exception as e:
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


def resolve_doc_path() -> Path:
    """Backwards-compatible shim for callers that still want a Path."""
    return DOC


def parse_blocks(doc_text: str) -> list[list[dict]]:
    """Parse every ``shell`` block into a list of ``{cmd, expected}``.

    Returns one inner list per fenced block. Each inner list contains
    the REPL commands and their expected output in source order. If a
    block contains no ``>>>`` command, the inner list is empty and the
    raw body is returned in the ``raw`` field of a sentinel dict so the
    runner can syntax-check the hand-written command.
    """
    fence_re = re.compile(r'```shell\s*\n(.*?)```', re.DOTALL)
    blocks: list[list[dict]] = []
    for m in fence_re.finditer(doc_text):
        body = m.group(1)
        block: list[dict] = []
        cur_cmd: list[str] = []
        cur_exp: list[str] = []
        for raw in body.splitlines():
            stripped = raw.lstrip()
            if stripped.startswith('>>> '):
                if cur_cmd or cur_exp:
                    block.append({
                        'cmd': '\n'.join(cur_cmd).rstrip(),
                        'expected': cur_exp,
                    })
                cur_cmd = [stripped[4:]]
                cur_exp = []
            elif stripped.startswith('... '):
                cur_cmd.append(stripped[4:])
            elif stripped.startswith('<<< '):
                cur_cmd.append(f': <<< {stripped[4:]}')
            elif stripped.startswith('#'):
                # Drop comment lines entirely (neither a command nor
                # expected output). Lets the doc author sprinkle plain
                # `# ...` explanations inside a shell block.
                continue
            else:
                cur_exp.append(raw)
        if cur_cmd or cur_exp:
            block.append({
                'cmd': '\n'.join(cur_cmd).rstrip(),
                'expected': cur_exp,
            })
        # Trim trailing blank expected lines.
        for c in block:
            while c['expected'] and c['expected'][-1].strip() == '':
                c['expected'].pop()
        # If no ``>>>`` was found anywhere, the whole body is a hand-written
        # command. Encode it as a single sentinel step with empty cmd and
        # the raw body in ``expected`` so the runner can pick it up.
        if not any(c['cmd'].strip() for c in block):
            block = [{'cmd': '', 'expected': [body], 'raw': body}]
        blocks.append(block)
    return blocks


# --------------------------------------------------------------------------- #
# Placeholder handling                                                        #
# --------------------------------------------------------------------------- #

# (pattern, regex_fragment). Named-capture fragments use (?P<name>...).
_PLACEHOLDER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'<pid>'),         r'(?P<pid>\d+)'),
    (re.compile(r'<x\.y\.z>'),     r'\d+\.\d+\.\d+'),
    (re.compile(r'\b2\.9\.0\.postX\b'), r'2\.9\.0\.post\d+'),
    (re.compile(r'\b2\.9\.0\.x\b'),     r'2\.9\.0\.\d+'),
    (re.compile(r'\b3\.11\.x\b'),       r'3\.11\.\d+'),
    (re.compile(r'v\d+-xxx'),      r'v\d+-\S+'),
    (re.compile(r'checkpoint-xxx'), r'checkpoint-\S+'),
    (re.compile(r'chatcmpl-xxx'),  r'chatcmpl-\S+'),
    (re.compile(r'\bxxx\b'),       r'\S+'),
    (re.compile(r'"created":\d+'), r'"created":\d+'),
]


def _sentinel(i: int) -> str:
    """A control-character token that ``re.escape`` will not touch."""
    return f'\x01S{i}\x01'


def substitute_placeholders(expected: str) -> tuple[str, dict]:
    """Swap placeholders for sentinels, returning (escaped_text, mapping)."""
    mapping: dict = {}
    text = expected
    counter = 0

    # Named-capture placeholders go first so the group name survives.
    for pat, repl in _PLACEHOLDER_PATTERNS:
        if '(?P<' not in repl:
            continue
        token = _sentinel(counter)
        mapping[counter] = repl
        counter += 1
        text = pat.sub(token, text)

    for pat, repl in _PLACEHOLDER_PATTERNS:
        if '(?P<' in repl:
            continue
        token = _sentinel(counter)
        mapping[counter] = repl
        counter += 1
        text = pat.sub(token, text)

    return text, mapping


def expected_line_to_regex(expected: str) -> re.Pattern:
    """Build a regex that matches a single line of actual output."""
    text, mapping = substitute_placeholders(expected)
    escaped = re.escape(text)
    for i, frag in mapping.items():
        escaped = escaped.replace(_sentinel(i), frag)
    # A literal ``...`` in the expected line becomes a wildcard.
    escaped = escaped.replace(r'\.\.\.', '.*')
    return re.compile('^' + escaped + '$')


# --------------------------------------------------------------------------- #
# Command execution                                                           #
# --------------------------------------------------------------------------- #

def run_command(cmd: str, env: dict, cwd: Path, timeout: int) -> tuple[int, str]:
    """Run ``cmd`` in bash; return ``(returncode, stdout+stderr)``."""
    t0 = time.time()
    # Truncate to 2000 chars so a multi-line swift sft invocation
    # stays visible in the log without flooding it.
    _log(f'CMD start (timeout={timeout}s): {cmd[:2000]}')
    proc = subprocess.run(
        ['bash', '-c', cmd],
        env=env,
        cwd=cwd,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    elapsed = time.time() - t0
    _log(f'CMD done in {elapsed:.1f}s rc={proc.returncode} '
         f'(stdout={len(out)}B stderr={len(err)}B)')
    # Surface the actual output when the command didn't return what we
    # expected, so the GitHub step log shows the failure mode without
    # having to download an artifact. Truncate to keep the log sane
    # (swift sft training can easily produce hundreds of MB of stdout).
    if proc.returncode != 0 or (out.strip() == '' and err.strip() != ''):
        head_out = out[:2000]
        tail_out = out[-2000:] if len(out) > 2000 else ''
        head_err = err[:2000]
        tail_err = err[-2000:] if len(err) > 2000 else ''
        if head_out:
            _log(f'CMD stdout (head):\n{head_out.rstrip()}')
        if tail_out and tail_out != head_out:
            _log(f'CMD stdout (tail):\n{tail_out.rstrip()}')
        if head_err:
            _log(f'CMD stderr (head):\n{head_err.rstrip()}')
        if tail_err and tail_err != head_err:
            _log(f'CMD stderr (tail):\n{tail_err.rstrip()}')
    # Return only stdout - stderr is noise (init banners, warnings)
    # that would otherwise get folded into 'expected output' and
    # cause spurious mismatches. Stderr is still logged above for
    # debugging when the command fails.
    return proc.returncode, out


# Per-block wall-clock budgets (seconds). The full doc takes the sum.
BLOCK_TIMEOUTS = {
    'install': 600,   # pip install ms-swift (CI image is pre-installed)
    'train': 1800,    # swift sft on 1000 samples
    'merge': 600,     # swift export --merge_lora
    'infer': 600,     # swift infer with piped input
    'deploy': 900,    # swift deploy + chat completion + kill
    'default': 300,
}


def block_kind(block: list[dict]) -> str:
    """Guess the kind of a block from its first command.

    Strips a leading ``VAR=...`` prefix (e.g. ``ASCEND_RT_VISIBLE_DEVICES=0``)
    before matching so the timeout bucket reflects what the command
    actually does, not how it was prefixed.
    """
    if not block:
        return 'default'
    head = block[0]['cmd'].lstrip().splitlines()[0] if block[0]['cmd'] else ''
    # Strip `KEY=VALUE ` prefix(es) that often precede the real command.
    while '=' in head.split(' ', 1)[0]:
        head = head.split(' ', 1)[1] if ' ' in head else ''
    if 'pip install' in head:
        return 'install'
    if head.startswith('swift sft'):
        return 'train'
    if head.startswith('swift export'):
        return 'merge'
    if head.startswith('swift infer'):
        return 'infer'
    if head.startswith(('nohup swift deploy', 'swift deploy')):
        return 'deploy'
    return 'default'


# End-to-end tests are only run on the NPU runner. Set SWIFT_NPU_E2E=1 to
# actually execute the test; otherwise the class is skipped.
_SKIP_E2E = os.environ.get('SWIFT_NPU_E2E', '0') != '1'


@unittest.skipIf(_SKIP_E2E,
                 'end-to-end tests require NPU runner; set SWIFT_NPU_E2E=1 to run')
class TestQuickStartAscendEndToEnd(unittest.TestCase):
    """End-to-end: actually run every block on a real NPU runner.

    These tests are **only** meant to run on a self-hosted NPU runner
    (the ``linux-aarch64-a2-1`` runner in this repo's
    ``citest_npu.yaml``). They execute real ``swift sft`` /
    ``swift infer`` commands and compare stdout against the expected
    output declared in the doc. They are skipped by default in any
    other environment.
    """

    doc_path: str = None
    blocks: list[list[dict]] = None

    @classmethod
    def setUpClass(cls):
        _log(f'setUpClass: fetching doc from {os.environ.get("MONITORED_DOC_URL", "<unset>")}')
        cls.doc_text, cls.doc_path = fetch_doc_text()
        _log(f'setUpClass: fetched doc ({len(cls.doc_text)} bytes)')
        # Record the upstream ref / commit being tested. The CI
        # workflow sets these before invoking unittest; when running
        # outside CI both are unset and the test is skipped below.
        cls.upstream_ref = os.environ.get('UPSTREAM_REF', '')
        cls.upstream_commit = os.environ.get('UPSTREAM_COMMIT', '')
        if not cls.upstream_ref or not cls.upstream_commit:
            raise unittest.SkipTest(
                'end-to-end requires UPSTREAM_REF and UPSTREAM_COMMIT '
                '(set by the CI workflow)')
        os.environ.setdefault('UPSTREAM_REF', cls.upstream_ref)
        os.environ.setdefault('UPSTREAM_COMMIT', cls.upstream_commit)
        _log(f'setUpClass: upstream ref={cls.upstream_ref} commit={cls.upstream_commit[:12]}')
        # Substitute <UPSTREAM_REF> in the doc with the exact ref/SHA
        # the monitor triggered on, then parse. The doc's
        # `## install ms-swift` block uses this placeholder to do the
        # source install in-band; we no longer install ms-swift here.
        cls.doc_text = cls.doc_text.replace(
            '<UPSTREAM_REF>', cls.upstream_commit)
        cls.blocks = parse_blocks(cls.doc_text)
        total_steps = sum(len(b) for b in cls.blocks)
        _log(f'setUpClass: parsed {total_steps} steps across '
             f'{len(cls.blocks)} blocks')
        if not cls.blocks:
            raise unittest.SkipTest(
                f'No shell code blocks found in {cls.doc_path}')

    def test_runs_quick_start(self):
        """Walk every block and execute, comparing actual vs expected.

        Fail fast on the first mismatch - subsequent blocks likely
        depend on previous ones (e.g. swift sft needs swift CLI from
        the install block).
        """
        env = os.environ.copy()
        env['UPSTREAM_REF'] = self.upstream_ref
        env['UPSTREAM_COMMIT'] = self.upstream_commit
        env.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
        env.setdefault('MODELSCOPE_CACHE', str(Path.home() / '.cache'))
        work_dir = REPO_ROOT / 'output' / 'npu-quick-start-lora'
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)
        env['WORK_DIR'] = str(work_dir)

        captures: dict = {}

        _log(f'test_runs_quick_start: starting {len(self.blocks)} blocks')
        for bi, block in enumerate(self.blocks):
            kind = block_kind(block)
            timeout = BLOCK_TIMEOUTS.get(kind, BLOCK_TIMEOUTS['default'])
            _log(f'BLOCK {bi}/{len(self.blocks)-1} kind={kind} timeout={timeout}s '
                 f'steps={len(block)}')
            if len(block) == 1 and not block[0]['cmd'].strip():
                _log(f'BLOCK {bi}: sentinel, skip')
                continue
            self._run_block(block, kind, env, captures, timeout, bi)
        _log('test_runs_quick_start: all blocks done')

    def _run_block(self, block, kind, env, captures, timeout, block_idx):
        actual_lines_per_step: list[list[str]] = []
        for si, step in enumerate(block):
            cmd = step['cmd']
            for k, v in captures.items():
                cmd = cmd.replace(f'<{k}>', v)
            if kind == 'train' and '--max_steps' not in cmd:
                cmd = cmd.rstrip() + (
                    ' \\\n    --max_steps 5'
                    ' \\\n    --save_strategy steps'
                    ' \\\n    --save_steps 5'
                    ' \\\n    --logging_steps 1'
                    ' \\\n    --eval_strategy no'
                    ' \\\n    --report_to none'
                )
            rc, out = run_command(cmd, env, REPO_ROOT, timeout)
            self.assertEqual(
                rc, 0,
                f'block #{block_idx} ({kind}) command failed (rc={rc}):\n'
                f'  cmd: {cmd!r}\n'
                f'  output:\n{out}')
            actual_lines_per_step.append(out.splitlines())

        for si, (step, actual_lines) in enumerate(zip(block, actual_lines_per_step)):
            expected = step['expected']
            if not expected:
                continue
            self._compare_lines(
                block_idx, si, kind, expected, actual_lines, captures,
            )

    def _compare_lines(self, block_idx, step_idx, kind, expected, actual, captures):
        """Walk expected and actual in lockstep, matching each line.

        A literal ``...`` on the expected side skips ONE actual line.
        Use consecutive ``...`` lines (or a trailing ``...``) to drop
        several, which is the typical pattern for variable-length
        output (training logs, model-generated text, ``npu-smi`` rows)
        where only the tail matters.

        Log policy:
          - When everything matches, log a single 'OK' line.
          - When something mismatches, dump the full expected and
            actual blocks once (so the reader sees them side by side)
            and mark each line OK/BAD.
        """
        a_iter: list[str] = list(actual)
        e_iter: list[str] = list(expected)
        mismatches: list[tuple[int, str, str | None]] = []  # (ei, exp, act)
        line_summary: list[tuple[str, str | None, bool]] = []
        for ei, line in enumerate(e_iter):
            if line == '...':
                # Wildcard: skip ALL remaining actual lines (and
                # mark this expected line as consumed). Use as a
                # trailing '...' to swallow variable-length tail
                # output (progress bars, init banners). Subsequent
                # expected entries are ignored - so place '...' only
                # AFTER all the lines you actually want to verify.
                while a_iter:
                    consumed = a_iter.pop(0)
                    line_summary.append(('...', consumed, True))
                continue
            actual_line = a_iter.pop(0) if a_iter else None
            if actual_line is None:
                mismatches.append((ei, line, None))
                line_summary.append((line, None, False))
                continue
            pat = expected_line_to_regex(line)
            m = pat.match(actual_line)
            ok = m is not None
            line_summary.append((line, actual_line, ok))
            if not ok:
                mismatches.append((ei, line, actual_line))
                continue
            # Pick up named captures from this actual line.
            for k, v in m.groupdict().items():
                if v is not None:
                    captures[k] = v
        leftover_count = sum(1 for _ in a_iter)

        if not mismatches:
            # Compact success: one line, plus leftover count if any.
            suffix = (f', {leftover_count} leftover line(s) ignored'
                      if leftover_count else '')
            _log(f'BLOCK {block_idx} step {step_idx} ({kind}): OK '
                 f'({len(expected)} expected, {len(actual)} actual{suffix})')
            # When there are leftovers (even on OK), dump the actual
            # content of each one - so the next person who wonders
            # 'why is actual bigger than expected' can see exactly
            # what got swallowed.
            if leftover_count:
                start_idx = len(expected)
                for i, ln in enumerate(actual[start_idx:], 1):
                    _log(f'    leftover {i}: {ln!r}')
            return

        # Mismatch path: dump the full expected and actual blocks once,
        # then highlight which lines failed.
        _log(f'BLOCK {block_idx} step {step_idx} ({kind}): MISMATCH '
             f'({len(expected)} expected, {len(actual)} actual, '
             f'{leftover_count} leftover)')
        _log('  --- expected ---')
        for i, ln in enumerate(e_iter, 1):
            _log(f'  {i:>3}. {ln}')
        _log('  --- actual (head + tail if huge) ---')
        if len(actual) <= 20:
            for i, ln in enumerate(actual, 1):
                _log(f'  {i:>3}. {ln}')
        else:
            # Truncate huge actual blocks (e.g. swift sft training
            # emits hundreds of progress lines).
            for i, ln in enumerate(actual[:10], 1):
                _log(f'  {i:>3}. {ln}')
            _log(f'  ... [{len(actual) - 20} line(s) elided] ...')
            for i, ln in enumerate(actual[-10:], len(actual) - 9):
                _log(f'  {i:>3}. {ln}')
        _log('  --- per-line status ---')
        for i, (exp, act, ok) in enumerate(line_summary, 1):
            mark = 'OK ' if ok else 'BAD'
            exp_disp = repr(exp) if exp else '<skip>'
            act_disp = repr(act) if act is not None else '<missing>'
            _log(f'  {i:>3}. [{mark}] exp={exp_disp}  act={act_disp}')
        msg = (f'block #{block_idx} step #{step_idx} ({kind}) output '
               'mismatch; see summary above')
        self.fail(msg)


if __name__ == '__main__':
    unittest.main()
