#!/usr/bin/env python3
"""Diff a target examples/ tree against examples_manifest.yaml.

Writes a machine-readable JSON result (new/stale paths, supported entries,
target repo/ref) to --result-json and stdout.
Always exits 0. Writes supported_matrix JSON to GITHUB_OUTPUT when set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')


def scan_examples(target_root: Path) -> list[str]:
    examples_root = target_root / 'examples'
    if not examples_root.is_dir():
        raise SystemExit(f'examples/ not found under {target_root}')
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in INCLUDE_EXTENSIONS:
            found.append(path.relative_to(target_root).as_posix())
    return found


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_manifest(path: Path) -> dict:
    """Parse the bootstrap-generated manifest without a YAML dependency."""
    supported: list[dict] = []
    unsupported: list[str] = []
    current: dict | None = None
    section: str | None = None
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip(' '))
        if indent == 0 and stripped.endswith(':') and not stripped.startswith('-'):
            if current is not None and section == 'supported':
                supported.append(current)
                current = None
            key = stripped[:-1]
            section = key if key in {'scan', 'supported', 'unsupported'} else None
            continue
        if section == 'supported' and stripped.startswith('- path:'):
            if current is not None:
                supported.append(current)
            current = {'path': _unquote(stripped.split(':', 1)[1])}
            continue
        if section == 'supported' and current is not None and ':' in stripped and not stripped.startswith('-'):
            key, value = stripped.split(':', 1)
            parsed: str | int = _unquote(value)
            if key == 'timeout_minutes':
                parsed = int(parsed)
            current[key] = parsed
            continue
        if section == 'unsupported' and stripped.startswith('- '):
            unsupported.append(_unquote(stripped[2:]))
    if current is not None:
        supported.append(current)
    return {'supported': supported, 'unsupported': unsupported}


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
    scanned = set(scan_examples(target_root))
    listed = {item['path'] for item in manifest['supported']} | set(manifest['unsupported'])
    new_paths = sorted(scanned - listed)
    stale_paths = sorted(listed - scanned)
    write_result(args.result_json, args.trigger, args.target_repo,
                 args.target_ref, new_paths, stale_paths, manifest['supported'])
    write_github_output(manifest['supported'])
    raise SystemExit(0)


if __name__ == '__main__':
    main()
