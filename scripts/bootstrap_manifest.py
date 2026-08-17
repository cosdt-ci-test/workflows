#!/usr/bin/env python3
"""Scan a target tree's examples and write examples_manifest.yaml.

Paths passed with --supported are written to the supported section. Every
other scanned example is written as unsupported. That classification is a
task rule, not a community judgment.

--scan-root and --include-extension control what is scanned and are
recorded in the manifest's scan section, which the CI-side
check_examples_manifest.py replays. --runner / --npu-devices / --image /
--timeout-minutes / --profile apply to every supported
entry; entries that need different values must be edited by hand
afterwards. overlay_args is optional and always left as a comment
for hand editing.

CI does not call this script. Use it once when onboarding a project, then
fill in the scheduling fields on each supported entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SCAN_ROOT = 'examples'
DEFAULT_INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')


def normalize_extension(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise SystemExit('empty value passed to --include-extension')
    return ext if ext.startswith('.') else f'.{ext}'


def scan_examples(target_root: Path, scan_root: str,
                  include_extensions: tuple[str, ...]) -> list[str]:
    examples_root = target_root / scan_root
    if not examples_root.is_dir():
        raise SystemExit(f'{scan_root}/ not found under {target_root}')
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in include_extensions:
            rel = path.relative_to(target_root).as_posix()
            found.append(rel)
    return found


def render_supported_entry(
    path: str,
    runner: str | None,
    npu_devices: str | None,
    image: str | None,
    timeout_minutes: int | None,
    profile: str | None,
) -> list[str]:
    lines = [f'  - path: {path}']
    if profile is not None:
        lines.append(f'    profile: {profile}')
    else:
        lines.append('    # profile: <setup-profile>')
    if runner is not None:
        lines.append(f'    runner: {runner}')
    else:
        lines.append('    # runner: <runner-label>')
    if npu_devices is not None:
        lines.append(f"    npu_devices: '{npu_devices}'")
    else:
        lines.append("    # npu_devices: '0,1'")
    if image is not None:
        lines.append(f'    image: {image}')
    else:
        lines.append('    # image: <swr-image>')
    lines.append('    # overlay_args: []')
    if timeout_minutes is not None:
        lines.append(f'    timeout_minutes: {timeout_minutes}')
    else:
        lines.append('    # timeout_minutes: 180')
    return lines


def render_manifest(
    paths: list[str],
    supported_paths: list[str],
    scan_root: str,
    include_extensions: tuple[str, ...],
    runner: str | None,
    npu_devices: str | None,
    image: str | None,
    timeout_minutes: int | None,
    profile: str | None,
) -> str:
    missing = [p for p in supported_paths if p not in paths]
    if missing:
        raise SystemExit(f'supported example missing from scan: {missing[0]}')
    supported_set = set(supported_paths)
    unsupported = [p for p in paths if p not in supported_set]
    rendered_extensions = ', '.join(f"'{ext}'" for ext in include_extensions)
    lines = [
        'version: 1',
        'scan:',
        f'  root: {scan_root}',
        f'  include_extensions: [{rendered_extensions}]',
        'supported:',
    ]
    if supported_paths:
        for path in supported_paths:
            lines.extend(render_supported_entry(
                path, runner, npu_devices, image, timeout_minutes,
                profile,
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
    parser.add_argument(
        '--scan-root',
        default=DEFAULT_SCAN_ROOT,
        help='Directory under the target root to scan (default: examples)',
    )
    parser.add_argument(
        '--include-extension',
        action='append',
        default=None,
        help="File extension to scan, with or without the leading dot "
             "(default: .sh .py .yaml). Repeatable.",
    )
    parser.add_argument('--runner', default=None, help='Runner label written on every supported entry')
    parser.add_argument('--npu-devices', default=None, help="Value for npu_devices, e.g. 0,1")
    parser.add_argument('--image', default=None, help='Container image written on every supported entry')
    parser.add_argument('--timeout-minutes', type=int, default=None, help='Timeout written on every supported entry')
    parser.add_argument('--profile', default=None, help='Setup profile written on every supported entry')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output)
    if args.include_extension is None:
        include_extensions = DEFAULT_INCLUDE_EXTENSIONS
    else:
        include_extensions = tuple(
            normalize_extension(ext) for ext in args.include_extension)
    text = render_manifest(
        scan_examples(target_root, args.scan_root, include_extensions),
        args.supported,
        args.scan_root,
        include_extensions,
        args.runner,
        args.npu_devices,
        args.image,
        args.timeout_minutes,
        args.profile,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(f'wrote {output} ({text.count(chr(10))} lines)')


if __name__ == '__main__':
    main()
