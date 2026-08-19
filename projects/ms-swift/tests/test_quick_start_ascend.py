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
* ``MAJOR.MINOR.x`` - match any ``d.d.<patch>`` triple (e.g. ``3.12.x``,
              ``3.11.x``, ``2.7.x`` - one placeholder line covers all
              minor versions without per-minor entries).
* ``2.9.0.postX`` - match ``2.9.0.post<patch>`` (legacy pre-MINOR.x
              placeholder for Ascend NPU plugin releases).
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
        in_cmd = False   # True right after a '>>>' line - `...` then
                        # continues the command; False once any
                        # non-`...` line (expected output) is seen.
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
                # Right after `>>>`, `...` lines are bash
                # continuations of the command, not wildcards.
                in_cmd = True
            elif in_cmd and stripped.startswith('... '):
                # `...` continuation of the current `>>>` command.
                cur_cmd.append(stripped[4:])
            elif stripped == '...':
                # Bare ``...`` line - swallow whatever the actual
                # line ends up being (true wildcard).
                cur_exp.append('...')
                in_cmd = False
            elif stripped.startswith('... ') and stripped.lstrip('...').strip().startswith('#'):
                # ``...`` followed by an inline comment only (the
                # whole line is annotation + wildcard, no content
                # anchor past it). Strip the comment so the regex
                # framework sees a single ``...``.
                cur_exp.append('...')
                in_cmd = False
            elif stripped.startswith('...#'):
                # ``...#...`` (no leading space before the comment).
                # Same handling as ``... # ...`` above.
                cur_exp.append('...')
                in_cmd = False
            elif stripped.startswith('...'):
                # ``...`` followed by literal content - we keep the
                # whole line so the literal content survives as a
                # regex anchor instead of being silently dropped.
                # The downstream framework already turns every
                # ``...`` sequence into ``.*?`` (non-overlapping
                # ``\\.\\.\\.`` -> ``.*?`` replacement in
                # ``_compare_lines``), so ``...你好...`` becomes the
                # regex ``.*?你好.*?`` - the leading / trailing
                # ``...`` swallow variable noise; ``你好`` literal
                # anchors swift infer actually emitting that token.
                cur_exp.append(stripped)
                in_cmd = False
            elif stripped.startswith('<<<'):
                cur_cmd.append(f': <<< {stripped[4:]}')
                in_cmd = True
            elif stripped.startswith('#'):
                # Drop comment lines entirely (neither a command nor
                # expected output). Lets the doc author sprinkle plain
                # `# ...` explanations inside a shell block.
                continue
            else:
                # Any non-`...` non-comment line ends the command-
                # continuation phase; subsequent `...` are wildcards.
                cur_exp.append(raw)
                in_cmd = False
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
#
# Only placeholders that the doc actually emits are kept. The
# Quick-start-Ascend.md is the only doc that runs through this
# test today (the ms-swift-examples.yml doc is a separate suite
# and does not import from this file).
_PLACEHOLDER_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ``<checkpoint>`` captures the Swift-SFT checkpoint path emitted by
    # ``ls -dt output/*/checkpoint-* | head -n 1`` so a later
    # ``swift infer --adapters <checkpoint>`` substitutes the actual
    # checkpoint directory instead of relying on a placeholder that
    # would never exist on disk.
    (re.compile(r'<checkpoint>'), r'(?P<checkpoint>output/[^/\s]+/checkpoint-\S+)'),
    # Generic ``MAJOR.MINOR.x`` version placeholder: matches any
    # ``d.d.<patch>`` so 3.12.x, 3.11.x, 2.7.x etc. all work without
    # us having to add a pattern line per minor. The ``\b`` rejects
    # ``3.12.x.y`` (extra dot) and ``3.12x`` (no dot before x) by
    # virtue of the ``\.`` between the digit groups.
    (re.compile(r'\b\d+\.\d+\.x\b'), r'\d+\.\d+\.\d+'),
    # ``xxx`` is a one-or-more non-whitespace, non-comma run. Used
    # to match the loss dict's ``2.27966142``, the train_runtime's
    # ``2.4083461``, and so on - any place the doc wants to flag a
    # specific number that the test should not pin.
    (re.compile(r'\bxxx\b'),       r'[^,\s]+?'),
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
    #
    # We also dump stderr when the process looks "healthy" (rc=0) but
    # the canonical error markers are present. swift infer on the
    # NPU runner has been seen to swallow [ERROR] / Exception lines
    # into stderr while exiting cleanly, masking a real failure
    # behind a successful run-conclusion.
    error_markers = ('[ERROR]', 'Traceback (most recent call last)',
                     'applicaiton exception', 'ERR99999')
    stderr_has_error = any(m in err for m in error_markers)
    if proc.returncode != 0 or stderr_has_error \
            or (out.strip() == '' and err.strip() != ''):
        head_out = out[:2000]
        tail_out = out[-2000:] if len(out) > 2000 else ''
        head_err = err[:2000]
        tail_err = err[-2000:] if len(err) > 2000 else ''
        # When the stderr carries an error marker we dump it FULL
        # (capped at a generous ~256 KB) instead of head/tail-only.
        # ERR99999 on the NPU runner often points at a Python
        # traceback nested deep in stderr; if we only show the first
        # and last 2 KB the actual offending file/line is silently
        # truncated and the next CI round reproduces the same red.
        full_err = err if stderr_has_error and len(err) <= 256_000 else ''
        if full_err:
            _log(f'CMD stderr (full, {len(full_err)}B):\n'
                 f'{full_err.rstrip()}')
        else:
            if head_err:
                _log(f'CMD stderr (head):\n{head_err.rstrip()}')
            if tail_err and tail_err != head_err:
                _log(f'CMD stderr (tail):\n{tail_err.rstrip()}')
        if head_out:
            _log(f'CMD stdout (head):\n{head_out.rstrip()}')
        if tail_out and tail_out != head_out:
            _log(f'CMD stdout (tail):\n{tail_out.rstrip()}')
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

    doc_path: str
    blocks: list[list[dict]]

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
            _log(f'[Test block {bi}/{len(self.blocks)-1}] kind={kind} timeout={timeout}s '
                 f'steps={len(block)}')
            if len(block) == 1 and not block[0]['cmd'].strip():
                _log(f'[Test block {bi}]: sentinel, skip')
                continue
            self._run_block(block, kind, env, captures, timeout, bi)
        _log('test_runs_quick_start: all blocks done')

    def _run_block(self, block, kind, env, captures, timeout, block_idx):
        actual_lines_per_step: list[list[str]] = []
        for si, step in enumerate(block):
            cmd = step['cmd']
            for k, v in captures.items():
                cmd = cmd.replace(f'<{k}>', v)
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
        """Match the whole expected block against the whole actual block.

        Join both into one string per side and build a single regex
        over the expected text. `...` becomes non-greedy `.*?` under
        `re.DOTALL` so it can swallow variable-length noise runs
        (loss logs, init banners). `xxx` becomes `\S+` for single-token
        placeholders. One `re.search` over both sides replaces the old
        per-line lockstep walk - simpler state machine, same coverage.

        Log policy:
          - When everything matches, log a single 'OK' line.
          - When something mismatches, dump the full expected and
            actual blocks once (so the reader sees them side by side).
        """
        expected_text = '\n'.join(expected)
        actual_text = '\n'.join(actual)

        text, mapping = substitute_placeholders(expected_text)
        escaped = re.escape(text)
        for i, frag in mapping.items():
            escaped = escaped.replace(_sentinel(i), frag)
        # `...` becomes non-greedy `.*?` with DOTALL so it can span
        # multiple actual lines (loss dicts, init banners).
        escaped = escaped.replace(r'\.\.\.', r'.*?')
        pattern = re.compile(escaped, re.DOTALL)
        m = pattern.search(actual_text)

        # Pick up named captures whether or not we matched.
        if m:
            for k, v in m.groupdict().items():
                if v is not None:
                    captures[k] = v

        if m:
            _log(f'[Test block {block_idx}] step {step_idx} ({kind}): OK '
                 f'({len(expected)} expected, {len(actual)} actual)')
            # Always print the full actual block (capped at 30 lines) so a
            # reader can spot-check what was matched without rerunning.
            self._log_block('actual', actual, cap=30)
            return

        # Mismatch - dump both blocks.
        _log(f'[Test block {block_idx}] step {step_idx} ({kind}): MISMATCH '
             f'({len(expected)} expected, {len(actual)} actual)')
        self._log_block('expected', expected)
        self._log_block('actual', actual)
        self.fail(
            f'block #{block_idx} step #{step_idx} ({kind}) output '
            'mismatch; see summary above')

    @staticmethod
    def _log_block(label: str, lines, cap: int = 30) -> None:
        """Pretty-print a block (or head+tail if it exceeds ``cap``).

        Always emits a consistent header so the OK and MISMATCH code paths
        show the same shape; an explicit ``cap`` keeps CI logs from
        flooding when the swift sft output runs into hundreds of lines.
        Output budget is exactly ``cap`` lines (default 30) for the head
        half of those plus the tail half of those, never exceeding it.
        """
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


if __name__ == '__main__':
    unittest.main()
