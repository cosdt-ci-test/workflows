"""Verify the summary contract for fresh runs and cache hits."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/summarize_example_results.py'
RUN_FIELDS = {
    'trigger': 'schedule',
    'target_repo': 'pytorch/tensordict',
    'target_ref': 'v0.13.0',
}
ZERO = {'total': 0, 'success': 0, 'failed': 0}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def detail(self, index, entry, status='success'):
        path = self.root / f'tensordict-examples-123-{index}' / 'result.json'
        path.parent.mkdir()
        path.write_text(json.dumps({**RUN_FIELDS, **entry, 'job_status': status}))
        return path

    def summary(self, matrix=(), **overrides):
        output = self.root / 'summary.json'
        env = {
            **os.environ, **{key.upper(): value for key, value in RUN_FIELDS.items()},
            'GITHUB_RUN_ID': '123',
            'PROJECT': 'tensordict',
            'REASON': 'release',
            'MANIFEST_RESULT': 'success',
            'PUBLISH_MATRIX': json.dumps(list(matrix)),
            'CONCLUSIONS': 'success skipped',
            **overrides,
        }
        subprocess.run(
            [sys.executable, str(SCRIPT), 'summarize', '--artifacts',
             str(self.root), '--output', str(output)],
            env=env, check=True,
        )
        return json.loads(output.read_text())

    def test_mixed_machine_results_include_failed_and_missing_details(self):
        a2 = {'path': 'examples/smoke.py', 'image': 'test:a2', 'machine_type': 'a2'}
        a3 = {**a2, 'image': 'test:a3', 'machine_type': 'a3'}
        missing = {**a3, 'path': 'examples/missing.py'}
        self.detail(0, a2)
        self.detail(1, a3, status='failure')
        summary = self.summary([a2, a3, missing])
        self.assertEqual(summary['machines'], {
            'a2': {'total': 1, 'success': 1, 'failed': 0},
            'a3': {'total': 2, 'success': 0, 'failed': 2},
        })
        self.assertEqual(summary['workflow_status'], 'failure')

    def test_a2_only_round_trips_through_cache_without_details(self):
        entry = {'path': 'examples/smoke.py', 'image': 'test:a2', 'machine_type': 'a2'}
        detail = self.detail(0, entry)
        executed = self.summary([entry])
        self.assertEqual(executed['machines']['a3'], ZERO)
        detail.unlink()
        cached = self.root / 'machines.json'
        cached.write_text(json.dumps(executed['machines']))
        machines = subprocess.check_output(
            [sys.executable, str(SCRIPT), 'cached', str(cached)], text=True
        )
        replayed = self.summary(
            REASON='cached', MANIFEST_RESULT='skipped', PUBLISH_MATRIX='',
            CACHED_MACHINES=machines,
        )
        self.assertEqual(replayed, {**executed, 'reason': 'cached'})

    def test_empty_plan_is_distinct_from_unavailable_plan(self):
        empty = self.summary()
        unavailable = self.summary(MANIFEST_RESULT='failure', PUBLISH_MATRIX='')
        self.assertEqual(empty['machines'], {'a2': ZERO, 'a3': ZERO})
        self.assertEqual(empty['workflow_status'], 'success')
        self.assertIsNone(unavailable['machines'])
        self.assertEqual(unavailable['workflow_status'], 'failure')

    def test_incomplete_cache_cannot_be_replayed(self):
        path = self.root / 'machines.json'
        for contents in (None, '{', json.dumps({'a2': ZERO}), json.dumps({
            'a2': {'total': 1, 'success': 0, 'failed': 1}, 'a3': ZERO,
        })):
            with self.subTest(contents=contents):
                if contents is not None:
                    path.write_text(contents)
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), 'cached', str(path)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, '')

    def test_preceding_failure_takes_priority_over_cancellation(self):
        cancelled = self.summary(CONCLUSIONS='success cancelled')
        failed = self.summary(CONCLUSIONS='failure cancelled')
        self.assertEqual(cancelled['workflow_status'], 'cancelled')
        self.assertEqual(failed['workflow_status'], 'failure')


if __name__ == '__main__':
    unittest.main()
