#!/usr/bin/env python3
"""Summarize example results and validate cached machine counts."""

import argparse
import json
import os
from pathlib import Path


MACHINE_TYPES = ('a2', 'a3')
COUNTERS = {'total', 'success', 'failed'}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def valid_cached_machines(machines):
    if not isinstance(machines, dict) or machines.keys() != set(MACHINE_TYPES):
        return False
    for counts in machines.values():
        if not isinstance(counts, dict) or counts.keys() != COUNTERS:
            return False
        if any(type(value) is not int or value < 0 for value in counts.values()):
            return False
        if counts['failed'] or counts['total'] != counts['success']:
            return False
    return True


def count_results(matrix, artifacts, artifact_prefix, run_fields):
    machines = {
        machine: {'total': 0, 'success': 0}
        for machine in MACHINE_TYPES
    }
    for index, entry in enumerate(matrix):
        counts = machines[entry['machine_type']]
        detail = read_json(artifacts / f'{artifact_prefix}-{index}' / 'result.json')
        expected = {
            **run_fields,
            'path': entry['path'],
            'image': entry['image'],
            'machine_type': entry['machine_type'],
            'job_status': 'success',
        }
        counts['total'] += 1
        if isinstance(detail, dict) and (
            all(detail.get(key) == value for key, value in expected.items())
            and detail.get('source') == entry.get('source')
        ):
            counts['success'] += 1
    for counts in machines.values():
        counts['failed'] = counts['total'] - counts['success']
    return machines


def workflow_status(conclusions, machines):
    if 'failure' in conclusions:
        return 'failure'
    if 'cancelled' in conclusions:
        return 'cancelled'
    if machines is None or any(counts['failed'] for counts in machines.values()):
        return 'failure'
    return 'success'


def summarize(artifacts, output):
    project = os.environ['PROJECT']
    reason = os.environ['REASON'] or 'monitor-error'
    if reason == 'cached':
        machines = json.loads(os.environ['CACHE_RESULT'])
    elif os.environ['MANIFEST_RESULT'] == 'success':
        matrix = json.loads(os.environ['PUBLISH_MATRIX'])
        prefix = f"{project}-examples-{os.environ['GITHUB_RUN_ID']}"
        run_fields = {
            key: os.environ[key.upper()]
            for key in ('trigger', 'target_repo', 'target_ref')
        }
        machines = count_results(matrix, artifacts, prefix, run_fields)
    else:
        machines = None

    summary = {
        'project': project,
        'trigger': os.environ['TRIGGER'],
        'target_ref': os.environ['TARGET_REF'] or None,
        'reason': reason,
        'machines': machines,
        'workflow_status': workflow_status(
            os.environ['CONCLUSIONS'].split(), machines
        ),
    }
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cached = commands.add_parser('cached')
    cached.add_argument('path', type=Path)
    collector = commands.add_parser('summarize')
    collector.add_argument('--artifacts', type=Path, required=True)
    collector.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.command == 'cached':
        machines = read_json(args.path)
        if not valid_cached_machines(machines):
            raise SystemExit(1)
        print(json.dumps(machines, separators=(',', ':')))
    else:
        summarize(args.artifacts, args.output)


if __name__ == '__main__':
    main()
