#!/usr/bin/env python3
"""Diff a target examples tree against examples_manifest.yaml.

The manifest's scan section (root, include_extensions) decides what to
scan. Writes a machine-readable JSON result (new/stale paths, supported
entries, target repo/ref) to --result-json and stdout.
Always exits 0. Writes supported_matrix and has_supported to
GITHUB_OUTPUT when set.

Requires PyYAML (preinstalled on ubuntu-latest runners).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

DEFAULT_SCAN_ROOT = 'examples'
DEFAULT_INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')


def load_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    scan = data.get('scan') or {}
    return {
        'scan_root': scan.get('root') or DEFAULT_SCAN_ROOT,
        'include_extensions': tuple(scan.get('include_extensions') or DEFAULT_INCLUDE_EXTENSIONS),
        'supported': data.get('supported') or [],
        'unsupported': data.get('unsupported') or [],
    }


def scan_examples(target_root: Path, scan_root: str,
                  include_extensions: tuple[str, ...]) -> list[str]:
    examples_root = target_root / scan_root
    if not examples_root.is_dir():
        raise SystemExit(f'{scan_root}/ not found under {target_root}')
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in include_extensions:
            found.append(path.relative_to(target_root).as_posix())
    return found


def write_result(result_json: str, trigger: str, target_repo: str,
                 target_ref: str, new_paths: list[str], stale_paths: list[str],
                 supported: list[dict]) -> None:
    result = {
        'trigger': trigger,
        'target_repo': target_repo,
        'target_ref': target_ref,
        'new_paths': new_paths,
        'stale_paths': stale_paths,
        'supported': supported,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    Path(result_json).write_text(text, encoding='utf-8')
    sys.stdout.write(text)


def write_github_output(supported: list[dict]) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if not output_path:
        return
    payload = json.dumps(supported, ensure_ascii=False)
    with open(output_path, 'a', encoding='utf-8') as handle:
        handle.write('supported_matrix<<EOF\n')
        handle.write(payload)
        handle.write('\nEOF\n')
        handle.write('has_supported={}\n'.format('true' if supported else 'false'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--result-json', required=True)
    parser.add_argument('--trigger', default='')
    parser.add_argument('--target-repo', default='')
    parser.add_argument('--target-ref', default='')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    manifest = load_manifest(Path(args.manifest))
    scanned = set(scan_examples(
        target_root, manifest['scan_root'], manifest['include_extensions']))
    listed = {item['path'] for item in manifest['supported']} | set(manifest['unsupported'])
    new_paths = sorted(scanned - listed)
    stale_paths = sorted(listed - scanned)
    write_result(args.result_json, args.trigger, args.target_repo,
                 args.target_ref, new_paths, stale_paths, manifest['supported'])
    write_github_output(manifest['supported'])
    raise SystemExit(0)


if __name__ == '__main__':
    main()
