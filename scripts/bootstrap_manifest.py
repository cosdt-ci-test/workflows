#!/usr/bin/env python3
"""Scan a target tree's examples/ and write examples_manifest.yaml.

Paths passed with --supported are written to the supported section. Every
other scanned example is written as unsupported. That classification is a
task rule, not a community judgment.

CI does not call this script. Use it once when onboarding a project, then
fill in runner / overlay / timeout on each supported entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')


def scan_examples(target_root: Path) -> list[str]:
    examples_root = target_root / 'examples'
    if not examples_root.is_dir():
        raise SystemExit(f'examples/ not found under {target_root}')
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in INCLUDE_EXTENSIONS:
            rel = path.relative_to(target_root).as_posix()
            found.append(rel)
    return found


def render_supported_entry(
    path: str,
    runner: str | None,
    npu_devices: str | None,
    overlay: str | None,
    timeout_minutes: int | None,
) -> list[str]:
    lines = [f'  - path: {path}']
    if runner is not None:
        lines.append(f'    runner: {runner}')
    else:
        lines.append('    # runner: <runner-label>')
    if npu_devices is not None:
        lines.append(f"    npu_devices: '{npu_devices}'")
    else:
        lines.append("    # npu_devices: '0,1'")
    if overlay is not None:
        lines.append(f'    overlay: {overlay}')
    else:
        lines.append('    # overlay: overlays/<name>.args')
    if timeout_minutes is not None:
        lines.append(f'    timeout_minutes: {timeout_minutes}')
    else:
        lines.append('    # timeout_minutes: 180')
    return lines


def render_manifest(
    paths: list[str],
    supported_paths: list[str],
    runner: str | None,
    npu_devices: str | None,
    overlay: str | None,
    timeout_minutes: int | None,
) -> str:
    missing = [p for p in supported_paths if p not in paths]
    if missing:
        raise SystemExit(f'supported example missing from scan: {missing[0]}')
    supported_set = set(supported_paths)
    unsupported = [p for p in paths if p not in supported_set]
    lines = [
        'version: 1',
        'scan:',
        '  root: examples',
        "  include_extensions: ['.sh', '.py', '.yaml']",
        'supported:',
    ]
    if supported_paths:
        for path in supported_paths:
            lines.extend(render_supported_entry(
                path, runner, npu_devices, overlay, timeout_minutes,
            ))
    else:
        lines.append('  []')
    lines.append('unsupported:')
    for path in unsupported:
        lines.append(f'  - {path}')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', required=True, help='Checkout of the target project tree')
    parser.add_argument('--output', required=True, help='Path to write examples_manifest.yaml')
    parser.add_argument(
        '--supported',
        action='append',
        default=[],
        help='Example path (relative to target root) to mark supported. Repeatable.',
    )
    parser.add_argument('--runner', default=None, help='Runner label written on every supported entry')
    parser.add_argument('--npu-devices', default=None, help="Value for npu_devices, e.g. 0,1")
    parser.add_argument('--overlay', default=None, help='Overlay path written on every supported entry')
    parser.add_argument('--timeout-minutes', type=int, default=None, help='Timeout written on every supported entry')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output)
    text = render_manifest(
        scan_examples(target_root),
        args.supported,
        args.runner,
        args.npu_devices,
        args.overlay,
        args.timeout_minutes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(f'wrote {output} ({text.count(chr(10))} lines)')


if __name__ == '__main__':
    main()
