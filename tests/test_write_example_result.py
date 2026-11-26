"""Tests for scripts/write_example_result.py (result publishing)."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import write_example_result

ENV = {
    'GH_TOKEN': 't',
    'GITHUB_REPOSITORY': 'org/workflows',
    'GITHUB_RUN_ID': '123',
    'EXPECTED_JOB_NAME': 'run_peft',
    'TRIGGER': 'schedule',
    'TARGET_REPO': 'huggingface/peft',
    'TARGET_REF': 'v0.20.0',
    'EXAMPLE_PATH': 'examples/sft/run_peft.sh',
    'IMAGE': 'swr.example/cann:tag',
}


class JobMatchingTests(unittest.TestCase):

    def test_exact_name_matches(self) -> None:
        self.assertTrue(write_example_result.job_matches(
            'run_peft', 'run_peft'))

    def test_called_workflow_prefix_matches(self) -> None:
        self.assertTrue(write_example_result.job_matches(
            'peft-examples / run_peft', 'run_peft'))

    def test_nested_prefix_matches(self) -> None:
        self.assertTrue(write_example_result.job_matches(
            'outer / peft-examples / run_peft', 'run_peft'))

    def test_other_jobs_do_not_match(self) -> None:
        self.assertFalse(write_example_result.job_matches(
            'peft-examples / publish-result (run_peft)', 'run_peft'))


class ConclusionMappingTests(unittest.TestCase):

    def test_success_and_cancelled_pass_through(self) -> None:
        self.assertEqual(write_example_result.conclusion_to_status('success'),
                         'success')
        self.assertEqual(
            write_example_result.conclusion_to_status('cancelled'),
            'cancelled')

    def test_everything_else_is_failure(self) -> None:
        for conclusion in ('failure', 'startup_failure', 'skipped', ''):
            self.assertEqual(
                write_example_result.conclusion_to_status(conclusion),
                'failure', conclusion)


class JobsApiRetryTests(unittest.TestCase):
    @staticmethod
    def http_error(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError('https://api.github.com', code, 'error',
                                      None, io.BytesIO())

    def test_retries_gateway_errors_and_timeouts(self) -> None:
        for error in (self.http_error(502), self.http_error(503), self.http_error(504),
                      TimeoutError(), urllib.error.URLError(TimeoutError())):
            with self.subTest(error=error):
                response = io.StringIO(json.dumps({
                    'jobs': [{'name': 'peft-examples / run_peft', 'conclusion': 'success'}],
                }))
                with mock.patch.object(write_example_result.urllib.request, 'urlopen',
                                       side_effect=[error, response]) as urlopen, \
                        mock.patch.object(write_example_result.time, 'sleep') as sleep, \
                        mock.patch('sys.stderr', new_callable=io.StringIO):
                    result = write_example_result.fetch_conclusion('org/workflows', '123',
                                                                  'run_peft', 't')
                self.assertEqual(result, 'success')
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(urlopen.call_args_list[0], urlopen.call_args_list[1])
                self.assertEqual(urlopen.call_args.kwargs['timeout'], 30)
                sleep.assert_called_once_with(1)

    def test_stops_after_three_retries(self) -> None:
        errors = [self.http_error(502) for _ in range(4)]
        with mock.patch.object(write_example_result.urllib.request, 'urlopen',
                               side_effect=errors) as urlopen, \
                mock.patch.object(write_example_result.time, 'sleep') as sleep, \
                mock.patch('sys.stderr', new_callable=io.StringIO), \
                self.assertRaises(urllib.error.HTTPError) as raised:
            write_example_result.fetch_conclusion('org/workflows', '123', 'run_peft', 't')
        self.assertIs(raised.exception, errors[-1])
        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(sleep.call_args_list, [mock.call(1), mock.call(2), mock.call(4)])

    def test_does_not_retry_non_transient_http_errors(self) -> None:
        for code in (403, 404, 429):
            with self.subTest(code=code), \
                    mock.patch.object(write_example_result.urllib.request, 'urlopen',
                                       side_effect=self.http_error(code)) as urlopen, \
                    mock.patch.object(write_example_result.time, 'sleep') as sleep, \
                    self.assertRaises(urllib.error.HTTPError):
                write_example_result.fetch_conclusion('org/workflows', '123', 'run_peft', 't')
            urlopen.assert_called_once()
            sleep.assert_not_called()


class MainTests(unittest.TestCase):

    def test_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result.json'
            with mock.patch.dict(os.environ, ENV), \
                    mock.patch.object(write_example_result, 'fetch_conclusion',
                                      return_value='failure'), \
                    mock.patch.object(sys, 'argv',
                                      ['prog', '--output', str(out)]):
                write_example_result.main()
            data = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(data, {
            'trigger': 'schedule',
            'target_repo': 'huggingface/peft',
            'target_ref': 'v0.20.0',
            'path': 'examples/sft/run_peft.sh',
            'image': 'swr.example/cann:tag',
            'job_status': 'failure',
        })

    def test_project_case_result_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result.json'
            env = dict(
                ENV,
                EXAMPLE_SOURCE='project',
                EXAMPLE_PATH='example/test_npu_discovery.py',
            )
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(write_example_result, 'fetch_conclusion',
                                      return_value='success'), \
                    mock.patch.object(sys, 'argv',
                                      ['prog', '--output', str(out)]):
                write_example_result.main()
            data = json.loads(out.read_text(encoding='utf-8'))

        self.assertIn('source', data)
        self.assertNotIn('case_id', data)
        self.assertEqual(data['source'], 'project')
        self.assertEqual(data['target_repo'], 'huggingface/peft')

    def test_result_schema_has_no_case_id_extension(self) -> None:
        schema_path = SCRIPTS.parent / 'schemas' / 'result.schema.json'
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        self.assertNotIn('case_id', schema['properties'])

    def test_missing_conclusion_exits_1(self) -> None:
        with mock.patch.dict(os.environ, ENV), \
                mock.patch.object(write_example_result, 'fetch_conclusion',
                                  return_value=None), \
                mock.patch.object(sys, 'argv',
                                  ['prog', '--output', '/tmp/x.json']), \
                self.assertRaises(SystemExit) as ctx:
            write_example_result.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_missing_env_exits_1(self) -> None:
        env = {k: v for k, v in ENV.items() if k != 'IMAGE'}
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(sys, 'argv', ['prog']), \
                self.assertRaises(SystemExit) as ctx:
            write_example_result.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
