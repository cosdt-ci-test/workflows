#!/usr/bin/env python3
"""Scan a target tree's examples and write examples_manifest.yaml.

Paths passed with --supported are written to the supported section. Every
other scanned example is written as unsupported. That classification is a
task rule, not a community judgment.

--scan-root, --include-extension, --unit, --marker, and --max-depth
control what is scanned and are recorded in the manifest's scan
section, which the CI-side check_examples_manifest.py replays.
Default unit is files (.sh / .py / .yaml). unit=directories treats
each child directory as one example. --runner / --npu-devices /
--image / --timeout-minutes / --profile apply to every supported
entry; entries that need different values must be edited by hand
afterwards. overlay_args and exec are optional and left as comments
for hand editing.

CI does not call this script. Use it once when onboarding a project, then
fill in the scheduling fields on each supported entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SCAN_ROOT = 'examples'
DEFAULT_INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')
DEFAULT_SCAN_UNIT = 'files'
DEFAULT_DIR_MARKER = 'CMakeLists.txt'
DEFAULT_DIR_MAX_DEPTH = 1
SCAN_UNITS = ('files', 'directories')


def normalize_extension(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise SystemExit('empty value passed to --include-extension')
    return ext if ext.startswith('.') else f'.{ext}'


def scan_file_units(examples_root: Path, target_root: Path,
                    include_extensions: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in include_extensions:
            found.append(path.relative_to(target_root).as_posix())
    return found


def scan_directory_units(examples_root: Path, target_root: Path,
                         marker: str, max_depth: int) -> list[str]:
    found: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if (child / marker).is_file():
                found.append(child.relative_to(target_root).as_posix())
            visit(child, depth + 1)

    visit(examples_root, 0)
    return found


def scan_examples(target_root: Path, scan_root: str, unit: str,
                  include_extensions: tuple[str, ...], marker: str,
                  max_depth: int) -> list[str]:
    examples_root = target_root / scan_root
    if not examples_root.is_dir():
        raise SystemExit(f'{scan_root}/ not found under {target_root}')
    if unit == 'directories':
        return scan_directory_units(
            examples_root, target_root, marker, max_depth)
    return scan_file_units(examples_root, target_root, include_extensions)


def render_supported_entry(
    path: str,
    runner: str | None,
    npu_devices: str | None,
    image: str | None,
    timeout_minutes: int | None,
    profile: str | None,
    unit: str,
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
    if unit == 'directories':
        lines.append('    # exec: build/bin/<binary>')
    lines.append('    # overlay_args: []')
    if timeout_minutes is not None:
        lines.append(f'    timeout_minutes: {timeout_minutes}')
    else:
        lines.append('    # timeout_minutes: 180')
    return lines


def render_scan_section(
    scan_root: str,
    unit: str,
    include_extensions: tuple[str, ...],
    marker: str,
    max_depth: int,
) -> list[str]:
    lines = [
        'scan:',
        f'  root: {scan_root}',
    ]
    if unit == 'directories':
        lines.append('  unit: directories')
        lines.append(f'  marker: {marker}')
        lines.append(f'  max_depth: {max_depth}')
        return lines
    rendered_extensions = ', '.join(f"'{ext}'" for ext in include_extensions)
    lines.append(f'  include_extensions: [{rendered_extensions}]')
    return lines


def render_manifest(
    paths: list[str],
    supported_paths: list[str],
    scan_root: str,
    unit: str,
    include_extensions: tuple[str, ...],
    marker: str,
    max_depth: int,
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
    lines = [
        'version: 1',
        *render_scan_section(
            scan_root, unit, include_extensions, marker, max_depth),
        'supported:',
    ]
    if supported_paths:
        for path in supported_paths:
            lines.extend(render_supported_entry(
                path, runner, npu_devices, image, timeout_minutes,
                profile, unit,
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
             "(default: .sh .py .yaml). Repeatable. Ignored when "
             "--unit directories.",
    )
    parser.add_argument(
        '--unit',
        choices=SCAN_UNITS,
        default=DEFAULT_SCAN_UNIT,
        help='Example unit to scan (default: files)',
    )
    parser.add_argument(
        '--marker',
        default=None,
        help='File that must exist in a directory unit '
             f'(default: {DEFAULT_DIR_MARKER} when --unit directories)',
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=DEFAULT_DIR_MAX_DEPTH,
        help='How many directory levels under --scan-root to treat as '
             'example units (default: 1)',
    )
    parser.add_argument('--runner', default=None, help='Runner label written on every supported entry')
    parser.add_argument('--npu-devices', default=None, help="Value for npu_devices, e.g. 0,1")
    parser.add_argument('--image', default=None, help='Container image written on every supported entry')
    parser.add_argument('--timeout-minutes', type=int, default=None, help='Timeout written on every supported entry')
    parser.add_argument('--profile', default=None, help='Setup profile written on every supported entry')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output)
    if args.max_depth < 1:
        raise SystemExit('--max-depth must be >= 1')
    if args.unit == 'directories':
        marker = (args.marker or DEFAULT_DIR_MARKER).strip()
        if not marker:
            raise SystemExit('--marker must be a non-empty string')
        include_extensions = DEFAULT_INCLUDE_EXTENSIONS
    else:
        marker = DEFAULT_DIR_MARKER
        if args.include_extension is None:
            include_extensions = DEFAULT_INCLUDE_EXTENSIONS
        else:
            include_extensions = tuple(
                normalize_extension(ext) for ext in args.include_extension)
    text = render_manifest(
        scan_examples(
            target_root, args.scan_root, args.unit, include_extensions,
            marker, args.max_depth),
        args.supported,
        args.scan_root,
        args.unit,
        include_extensions,
        marker,
        args.max_depth,
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
